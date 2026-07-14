from __future__ import annotations

import unittest

import numpy as np

from scheduler.graph import Graph, GraphNode
from scheduler.graph_passes import GraphPassPipeline
from scheduler.strategy import SchedulingStrategy
from scheduler.types import HardwareSpec, TensorInfo


def conv2d_nchw(
    x: np.ndarray,
    weight: np.ndarray,
    bias: np.ndarray | None = None,
    *,
    stride: tuple[int, int] = (1, 1),
    pads: tuple[int, int, int, int] = (0, 0, 0, 0),
    groups: int = 1,
) -> np.ndarray:
    n, c_in, h, w = x.shape
    c_out, c_per_group, kh, kw = weight.shape
    sh, sw = stride
    pt, pl, pb, pr = pads

    x_pad = np.pad(
        x,
        ((0, 0), (0, 0), (pt, pb), (pl, pr)),
        mode="constant",
    )
    out_h = (h + pt + pb - kh) // sh + 1
    out_w = (w + pl + pr - kw) // sw + 1
    y = np.zeros((n, c_out, out_h, out_w), dtype=np.float32)

    out_per_group = c_out // groups
    in_per_group = c_in // groups

    for batch in range(n):
        for group in range(groups):
            in_start = group * in_per_group
            out_start = group * out_per_group
            for oc_local in range(out_per_group):
                oc = out_start + oc_local
                for oy in range(out_h):
                    for ox in range(out_w):
                        region = x_pad[
                            batch,
                            in_start:in_start + c_per_group,
                            oy * sh:oy * sh + kh,
                            ox * sw:ox * sw + kw,
                        ]
                        y[batch, oc, oy, ox] = np.sum(
                            region * weight[oc]
                        )
                        if bias is not None:
                            y[batch, oc, oy, ox] += bias[oc]
    return y


def batch_norm_inference(
    x: np.ndarray,
    gamma: np.ndarray,
    beta: np.ndarray,
    mean: np.ndarray,
    variance: np.ndarray,
    epsilon: float,
) -> np.ndarray:
    scale_shape = (1, -1, 1, 1)
    return (
        (x - mean.reshape(scale_shape))
        / np.sqrt(variance.reshape(scale_shape) + epsilon)
        * gamma.reshape(scale_shape)
        + beta.reshape(scale_shape)
    )


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    shifted = x - np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(shifted)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def layer_norm(
    x: np.ndarray,
    scale: np.ndarray,
    bias: np.ndarray,
    epsilon: float = 1e-5,
) -> np.ndarray:
    mean = np.mean(x, axis=-1, keepdims=True)
    variance = np.mean((x - mean) ** 2, axis=-1, keepdims=True)
    normalized = (x - mean) / np.sqrt(variance + epsilon)
    return normalized * scale + bias


class C33NumericalCorrectnessTests(unittest.TestCase):
    def setUp(self) -> None:
        self.rng = np.random.default_rng(20260713)

    def test_conv_bn_fold_matches_unfused_reference(self) -> None:
        x = self.rng.normal(size=(2, 3, 7, 7)).astype(np.float32)
        weight = self.rng.normal(size=(4, 3, 3, 3)).astype(np.float32)
        conv_bias = self.rng.normal(size=(4,)).astype(np.float32)
        gamma = self.rng.normal(size=(4,)).astype(np.float32)
        beta = self.rng.normal(size=(4,)).astype(np.float32)
        mean = self.rng.normal(size=(4,)).astype(np.float32)
        variance = (
            np.abs(self.rng.normal(size=(4,))).astype(np.float32)
            + 0.5
        )
        epsilon = 1e-5

        graph = Graph(
            nodes=[
                GraphNode(
                    "conv",
                    "Conv",
                    ["x", "w", "conv_b"],
                    ["conv_y"],
                    {
                        "kernel_shape": (3, 3),
                        "strides": (1, 1),
                        "pads": (1, 1, 1, 1),
                        "group": 1,
                    },
                ),
                GraphNode(
                    "bn",
                    "BatchNormalization",
                    ["conv_y", "gamma", "beta", "mean", "var"],
                    ["y"],
                    {"epsilon": epsilon},
                ),
            ],
            inputs=["x"],
            outputs=["y"],
            initializers={
                "w": weight,
                "conv_b": conv_bias,
                "gamma": gamma,
                "beta": beta,
                "mean": mean,
                "var": variance,
            },
            tensors={
                "w": TensorInfo("w", "FLOAT", weight.shape),
            },
        )

        reference_conv = conv2d_nchw(
            x,
            weight,
            conv_bias,
            stride=(1, 1),
            pads=(1, 1, 1, 1),
        )
        reference = batch_norm_inference(
            reference_conv,
            gamma,
            beta,
            mean,
            variance,
            epsilon,
        )

        optimized, results = GraphPassPipeline(
            enable_fusion=True
        ).run(graph)

        self.assertEqual(len(optimized.nodes), 1)
        fused = optimized.nodes[0]
        self.assertEqual(fused.op_type, "FusedConv2dBatchNorm")
        self.assertTrue(fused.attributes["weights_folded"])

        merged_weight = optimized.initializers[fused.inputs[1]]
        merged_bias = optimized.initializers[fused.inputs[2]]
        actual = conv2d_nchw(
            x,
            np.asarray(merged_weight),
            np.asarray(merged_bias),
            stride=(1, 1),
            pads=(1, 1, 1, 1),
        )

        np.testing.assert_allclose(
            actual,
            reference,
            rtol=2e-5,
            atol=2e-5,
        )

        log = results["Fusion"]["stats"]["fusion_log"]
        self.assertEqual(log[0]["pattern"], "FusedConv2dBatchNorm")
        self.assertTrue(log[0]["weights_folded"])

    def test_conv_bn_fold_without_original_bias(self) -> None:
        x = self.rng.normal(size=(1, 2, 5, 5)).astype(np.float32)
        weight = self.rng.normal(size=(3, 2, 3, 3)).astype(np.float32)
        gamma = self.rng.normal(size=(3,)).astype(np.float32)
        beta = self.rng.normal(size=(3,)).astype(np.float32)
        mean = self.rng.normal(size=(3,)).astype(np.float32)
        variance = (
            np.abs(self.rng.normal(size=(3,))).astype(np.float32)
            + 0.2
        )
        epsilon = 1e-4

        graph = Graph(
            nodes=[
                GraphNode(
                    "conv",
                    "Conv",
                    ["x", "w"],
                    ["conv_y"],
                    {
                        "kernel_shape": (3, 3),
                        "strides": (1, 1),
                        "pads": (0, 0, 0, 0),
                    },
                ),
                GraphNode(
                    "bn",
                    "BatchNormalization",
                    ["conv_y", "gamma", "beta", "mean", "var"],
                    ["y"],
                    {"epsilon": epsilon},
                ),
            ],
            inputs=["x"],
            outputs=["y"],
            initializers={
                "w": weight,
                "gamma": gamma,
                "beta": beta,
                "mean": mean,
                "var": variance,
            },
            tensors={
                "w": TensorInfo("w", "FLOAT", weight.shape),
            },
        )

        reference = batch_norm_inference(
            conv2d_nchw(x, weight, None),
            gamma,
            beta,
            mean,
            variance,
            epsilon,
        )

        optimized, _ = GraphPassPipeline(
            enable_fusion=True
        ).run(graph)
        fused = optimized.nodes[0]

        actual = conv2d_nchw(
            x,
            np.asarray(optimized.initializers[fused.inputs[1]]),
            np.asarray(optimized.initializers[fused.inputs[2]]),
        )

        np.testing.assert_allclose(
            actual,
            reference,
            rtol=2e-5,
            atol=2e-5,
        )

    def test_matmul_bias_fused_semantics(self) -> None:
        x = self.rng.normal(size=(5, 7)).astype(np.float32)
        weight = self.rng.normal(size=(7, 4)).astype(np.float32)
        bias = self.rng.normal(size=(4,)).astype(np.float32)

        graph = Graph(
            nodes=[
                GraphNode("mm", "MatMul", ["x", "w"], ["t"]),
                GraphNode("bias", "Add", ["t", "b"], ["y"]),
            ],
            inputs=["x"],
            outputs=["y"],
            initializers={"w": weight, "b": bias},
        )

        reference = x @ weight + bias
        optimized, _ = GraphPassPipeline(
            enable_fusion=True
        ).run(graph)
        fused = optimized.nodes[0]

        self.assertEqual(fused.op_type, "FusedMatMulBias")
        actual = x @ optimized.initializers[fused.inputs[1]]
        actual = actual + optimized.initializers[fused.inputs[2]]

        np.testing.assert_allclose(
            actual,
            reference,
            rtol=1e-6,
            atol=1e-6,
        )

    def test_softmax_dropout_inference_semantics(self) -> None:
        x = self.rng.normal(size=(3, 11)).astype(np.float32)

        graph = Graph(
            nodes=[
                GraphNode(
                    "softmax",
                    "Softmax",
                    ["x"],
                    ["p"],
                    {"axis": -1},
                ),
                GraphNode(
                    "dropout",
                    "Dropout",
                    ["p"],
                    ["y"],
                ),
            ],
            inputs=["x"],
            outputs=["y"],
        )

        reference = softmax(x, axis=-1)
        optimized, _ = GraphPassPipeline(
            enable_fusion=True
        ).run(graph)

        fused = optimized.nodes[0]
        self.assertEqual(fused.op_type, "FusedSoftmaxDropout")

        # ONNX Dropout is identity in inference mode.
        actual = softmax(x, axis=-1)
        np.testing.assert_allclose(
            actual,
            reference,
            rtol=1e-7,
            atol=1e-7,
        )

    def test_residual_norm_fused_semantics(self) -> None:
        x = self.rng.normal(size=(2, 3, 8)).astype(np.float32)
        residual = self.rng.normal(size=(2, 3, 8)).astype(np.float32)
        scale = self.rng.normal(size=(8,)).astype(np.float32)
        bias = self.rng.normal(size=(8,)).astype(np.float32)
        epsilon = 1e-5

        graph = Graph(
            nodes=[
                GraphNode(
                    "residual_add",
                    "Add",
                    ["x", "residual"],
                    ["r"],
                ),
                GraphNode(
                    "norm",
                    "LayerNormalization",
                    ["r", "scale", "bias"],
                    ["y"],
                    {"epsilon": epsilon},
                ),
            ],
            inputs=["x", "residual"],
            outputs=["y"],
            initializers={
                "scale": scale,
                "bias": bias,
            },
        )

        reference = layer_norm(
            x + residual,
            scale,
            bias,
            epsilon,
        )

        optimized, _ = GraphPassPipeline(
            enable_fusion=True
        ).run(graph)
        fused = optimized.nodes[0]

        self.assertEqual(fused.op_type, "FusedResidualNorm")

        actual = layer_norm(
            x + residual,
            scale,
            bias,
            epsilon,
        )
        np.testing.assert_allclose(
            actual,
            reference,
            rtol=1e-6,
            atol=1e-6,
        )

    def test_elementwise_chain_fused_semantics(self) -> None:
        x = self.rng.normal(size=(4, 6)).astype(np.float32)
        addend = self.rng.normal(size=(4, 6)).astype(np.float32)
        multiplier = self.rng.normal(size=(4, 6)).astype(np.float32)

        graph = Graph(
            nodes=[
                GraphNode("add", "Add", ["x", "a"], ["t0"]),
                GraphNode("mul", "Mul", ["t0", "b"], ["t1"]),
                GraphNode("relu", "Relu", ["t1"], ["y"]),
            ],
            inputs=["x", "a", "b"],
            outputs=["y"],
        )

        reference = np.maximum((x + addend) * multiplier, 0.0)

        optimized, _ = GraphPassPipeline(
            enable_fusion=True
        ).run(graph)
        fused = optimized.nodes[0]

        self.assertEqual(fused.op_type, "FusedEWChain")
        self.assertEqual(
            tuple(fused.attributes["original_op_types"]),
            ("Add", "Mul", "Relu"),
        )

        actual = np.maximum((x + addend) * multiplier, 0.0)
        np.testing.assert_allclose(
            actual,
            reference,
            rtol=1e-7,
            atol=1e-7,
        )

    def test_multi_consumer_intermediate_is_not_fused(self) -> None:
        graph = Graph(
            nodes=[
                GraphNode("add", "Add", ["x", "a"], ["t"]),
                GraphNode("relu", "Relu", ["t"], ["y0"]),
                GraphNode("mul", "Mul", ["t", "b"], ["y1"]),
            ],
            inputs=["x", "a", "b"],
            outputs=["y0", "y1"],
        )

        optimized, results = GraphPassPipeline(
            enable_fusion=True
        ).run(graph)

        patterns = {
            entry["pattern"]
            for entry in results["Fusion"]["stats"]["fusion_log"]
        }
        self.assertNotIn("FusedEWChain", patterns)
        self.assertEqual(len(optimized.nodes), 3)
        self.assertTrue(optimized.validate())

    def test_graph_output_intermediate_is_not_fused(self) -> None:
        graph = Graph(
            nodes=[
                GraphNode("add", "Add", ["x", "a"], ["t"]),
                GraphNode("relu", "Relu", ["t"], ["y"]),
            ],
            inputs=["x", "a"],
            outputs=["t", "y"],
        )

        optimized, results = GraphPassPipeline(
            enable_fusion=True
        ).run(graph)

        patterns = {
            entry["pattern"]
            for entry in results["Fusion"]["stats"]["fusion_log"]
        }
        self.assertNotIn("FusedEWChain", patterns)
        self.assertEqual(len(optimized.nodes), 2)
        self.assertEqual(optimized.outputs, ["t", "y"])
        self.assertTrue(optimized.validate())

    def test_fused_nodes_schedule_to_dedicated_kernels(self) -> None:
        graph = Graph(
            nodes=[
                GraphNode("add", "Add", ["x", "a"], ["t0"]),
                GraphNode("mul", "Mul", ["t0", "b"], ["t1"]),
                GraphNode("relu", "Relu", ["t1"], ["y"]),
            ],
            inputs=["x", "a", "b"],
            outputs=["y"],
        )

        optimized, _ = GraphPassPipeline(
            enable_fusion=True
        ).run(graph)

        strategy = SchedulingStrategy(
            hardware=HardwareSpec(),
            autotune_mode="off",
        )
        fused = optimized.nodes[0]
        precision = strategy.select_precision(fused, optimized)
        kernels = strategy.decompose(
            fused,
            optimized,
            precision,
        )

        self.assertTrue(kernels)
        self.assertTrue(
            all(ref.name.startswith("fused_") for ref in kernels)
        )
        self.assertTrue(
            all(not ref.name.startswith("generic_") for ref in kernels)
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
