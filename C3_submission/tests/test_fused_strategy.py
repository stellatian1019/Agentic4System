from __future__ import annotations

import unittest

import numpy as np

from scheduler.graph import Graph, GraphNode
from scheduler.graph_passes import GraphPassPipeline
from scheduler.strategy import SchedulingStrategy
from scheduler.types import HardwareSpec


class FusedStrategyIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.strategy = SchedulingStrategy(
            hardware=HardwareSpec(),
            autotune_mode="off",
        )

    def _schedule(self, graph: Graph):
        optimized, _ = GraphPassPipeline(enable_fusion=True).run(graph)
        results = []
        for node in optimized.nodes:
            precision = self.strategy.select_precision(node, optimized)
            kernels = self.strategy.decompose(
                node,
                optimized,
                precision,
            )
            self.assertTrue(kernels)
            for ref in kernels:
                self.assertFalse(ref.name.startswith("generic_"))
                params = self.strategy.tune_kernel(
                    ref,
                    precision,
                    4096,
                )
                self.assertGreater(params.block_x, 0)
                self.assertGreater(params.grid_x, 0)
            results.append((node, precision, kernels))
        return results

    def test_matmul_bias_uses_fused_kernel(self) -> None:
        graph = Graph(
            nodes=[
                GraphNode("mm", "MatMul", ["x", "w"], ["t"]),
                GraphNode("add", "Add", ["t", "b"], ["y"]),
            ],
            inputs=["x"],
            outputs=["y"],
            initializers={
                "w": np.ones((4, 4), np.float32),
                "b": np.ones((4,), np.float32),
            },
        )
        results = self._schedule(graph)
        self.assertEqual(results[0][0].op_type, "FusedMatMulBias")
        self.assertTrue(
            results[0][2][0].name.startswith("fused_matmul_bias_")
        )

    def test_conv_bn_uses_fused_kernel(self) -> None:
        graph = Graph(
            nodes=[
                GraphNode(
                    "conv",
                    "Conv",
                    ["x", "w"],
                    ["t"],
                    {"kernel_shape": (2, 2)},
                ),
                GraphNode(
                    "bn",
                    "BatchNormalization",
                    ["t", "gamma", "beta", "mean", "var"],
                    ["y"],
                ),
            ],
            inputs=["x"],
            outputs=["y"],
            initializers={
                "w": np.ones((2, 1, 2, 2), np.float32),
                "gamma": np.ones((2,), np.float32),
                "beta": np.zeros((2,), np.float32),
                "mean": np.zeros((2,), np.float32),
                "var": np.ones((2,), np.float32),
            },
        )
        results = self._schedule(graph)
        self.assertEqual(
            results[0][2][0].name.split("_fp")[0],
            "fused_conv2d_batchnorm",
        )

    def test_sensitive_fused_ops_are_fp32(self) -> None:
        for op_type in (
            "FusedSoftmaxDropout",
            "FusedResidualNorm",
        ):
            graph = Graph(
                nodes=[
                    GraphNode(op_type, op_type, ["x"], ["y"]),
                ],
                inputs=["x"],
                outputs=["y"],
            )
            profile = self.strategy.select_precision(
                graph.nodes[0],
                graph,
            )
            self.assertEqual(profile.precision, "fp32")


if __name__ == "__main__":
    unittest.main(verbosity=2)
