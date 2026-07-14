from __future__ import annotations

import unittest

import numpy as np

from scheduler.graph import Graph, GraphNode
from scheduler.graph_passes import GraphPassPipeline
from scheduler.types import TensorInfo


def run_pattern(graph: Graph, expected: str) -> tuple[Graph, dict]:
    optimized, results = GraphPassPipeline(enable_fusion=True).run(graph)
    log = results["Fusion"]["stats"]["fusion_log"]
    patterns = {item["pattern"] for item in log}
    assert expected in patterns, (expected, patterns)
    assert len(optimized.nodes) < len(graph.nodes)
    assert optimized.inputs == graph.inputs
    assert optimized.outputs == graph.outputs
    assert optimized.validate()
    return optimized, results


class FusionPatternTests(unittest.TestCase):
    def test_fused_matmul_bias(self) -> None:
        graph = Graph(
            nodes=[
                GraphNode("mm", "MatMul", ["x", "w"], ["mm_y"]),
                GraphNode("bias", "Add", ["mm_y", "b"], ["y"]),
            ],
            inputs=["x"],
            outputs=["y"],
            initializers={
                "w": np.ones((4, 4), np.float32),
                "b": np.ones((4,), np.float32),
            },
        )
        run_pattern(graph, "FusedMatMulBias")

    def test_fused_conv2d_batchnorm_with_weight_fold(self) -> None:
        weight = np.arange(8, dtype=np.float32).reshape(2, 1, 2, 2)
        gamma = np.array([1.5, 0.5], np.float32)
        beta = np.array([0.2, -0.1], np.float32)
        mean = np.array([0.3, -0.4], np.float32)
        var = np.array([1.2, 0.8], np.float32)

        graph = Graph(
            nodes=[
                GraphNode(
                    "conv",
                    "Conv",
                    ["x", "w"],
                    ["conv_y"],
                    {"kernel_shape": (2, 2)},
                ),
                GraphNode(
                    "bn",
                    "BatchNormalization",
                    ["conv_y", "gamma", "beta", "mean", "var"],
                    ["y"],
                    {"epsilon": 1e-5},
                ),
            ],
            inputs=["x"],
            outputs=["y"],
            initializers={
                "w": weight,
                "gamma": gamma,
                "beta": beta,
                "mean": mean,
                "var": var,
            },
            tensors={
                "w": TensorInfo("w", "FLOAT", weight.shape),
            },
        )
        optimized, results = run_pattern(
            graph,
            "FusedConv2dBatchNorm",
        )
        fused = optimized.nodes[0]
        self.assertEqual(fused.op_type, "FusedConv2dBatchNorm")
        self.assertEqual(len(fused.inputs), 3)
        self.assertTrue(fused.attributes["weights_folded"])

        log = results["Fusion"]["stats"]["fusion_log"][0]
        self.assertIn(log["merged_weight"], optimized.initializers)
        self.assertIn(log["merged_bias"], optimized.initializers)

    def test_fused_ew_chain(self) -> None:
        graph = Graph(
            nodes=[
                GraphNode("add", "Add", ["x", "a"], ["t0"]),
                GraphNode("mul", "Mul", ["t0", "b"], ["t1"]),
                GraphNode("relu", "Relu", ["t1"], ["y"]),
            ],
            inputs=["x", "a", "b"],
            outputs=["y"],
        )
        run_pattern(graph, "FusedEWChain")

    def test_fused_softmax_dropout(self) -> None:
        graph = Graph(
            nodes=[
                GraphNode("sm", "Softmax", ["x"], ["p"]),
                GraphNode("drop", "Dropout", ["p"], ["y"]),
            ],
            inputs=["x"],
            outputs=["y"],
        )
        run_pattern(graph, "FusedSoftmaxDropout")

    def test_fused_residual_norm(self) -> None:
        graph = Graph(
            nodes=[
                GraphNode("skip_add", "Add", ["x", "skip"], ["r"]),
                GraphNode(
                    "ln",
                    "LayerNormalization",
                    ["r", "scale", "bias"],
                    ["y"],
                ),
            ],
            inputs=["x", "skip"],
            outputs=["y"],
            initializers={
                "scale": np.ones((4,), np.float32),
                "bias": np.zeros((4,), np.float32),
            },
        )
        run_pattern(graph, "FusedResidualNorm")


if __name__ == "__main__":
    unittest.main(verbosity=2)
