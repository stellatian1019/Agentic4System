from __future__ import annotations

import unittest

import numpy as np

from scheduler.graph import Graph, GraphNode
from scheduler.memory import ExecutionPlanBuilder
from scheduler.types import TensorInfo


class ExecutionPlanBuilderTests(unittest.TestCase):
    def _graph(self) -> Graph:
        return Graph(
            nodes=[
                GraphNode(
                    "mm",
                    "MatMul",
                    ["x", "w"],
                    ["t0"],
                ),
                GraphNode(
                    "relu",
                    "Relu",
                    ["t0"],
                    ["t1"],
                ),
                GraphNode(
                    "add",
                    "Add",
                    ["t1", "b"],
                    ["y"],
                ),
            ],
            inputs=["x"],
            outputs=["y"],
            initializers={
                "w": np.arange(
                    16,
                    dtype=np.float32,
                ).reshape(4, 4),
                "b": np.arange(
                    4,
                    dtype=np.float32,
                ),
            },
            tensors={
                "x": TensorInfo(
                    "x",
                    "FLOAT",
                    (2, 4),
                ),
                "w": TensorInfo(
                    "w",
                    "FLOAT",
                    (4, 4),
                ),
                "b": TensorInfo(
                    "b",
                    "FLOAT",
                    (4,),
                ),
                "t0": TensorInfo(
                    "t0",
                    "FLOAT",
                    (2, 4),
                ),
                "t1": TensorInfo(
                    "t1",
                    "FLOAT",
                    (2, 4),
                ),
                "y": TensorInfo(
                    "y",
                    "FLOAT",
                    (2, 4),
                ),
            },
        )

    def test_build_complete_plan(self) -> None:
        builder = ExecutionPlanBuilder(
            prefer_gpu=False,
        )
        result = builder.build(self._graph())

        self.assertTrue(result.validate())
        self.assertGreater(
            result.execution_plan.num_steps,
            0,
        )
        self.assertEqual(
            result.execution_plan.streams,
            ("compute",),
        )

    def test_weights_are_preloaded(self) -> None:
        builder = ExecutionPlanBuilder(
            prefer_gpu=False,
        )
        result = builder.build(self._graph())

        self.assertTrue(
            result.weight_store.is_resident("w")
        )
        self.assertTrue(
            result.weight_store.is_resident("b")
        )
        self.assertEqual(
            result.weight_store.copy_count,
            2,
        )

        np.testing.assert_array_equal(
            result.weight_store.backend.to_host(
                result.weight_store.get("w")
            ),
            self._graph().initializers["w"],
        )

    def test_activation_allocations_exist(self) -> None:
        builder = ExecutionPlanBuilder(
            prefer_gpu=False,
        )
        result = builder.build(self._graph())

        allocations = (
            result.execution_plan.allocations
        )
        self.assertIn("t0", allocations)
        self.assertIn("t1", allocations)
        self.assertIn("y", allocations)
        self.assertNotIn("w", allocations)
        self.assertNotIn("x", allocations)

    def test_kernel_steps_have_valid_tuning(self) -> None:
        builder = ExecutionPlanBuilder(
            prefer_gpu=False,
        )
        result = builder.build(self._graph())

        for step in result.execution_plan.steps:
            self.assertGreater(
                step.attributes["block_x"],
                0,
            )
            self.assertGreater(
                step.attributes["grid_x"],
                0,
            )
            self.assertGreaterEqual(
                step.attributes["smem_bytes"],
                0,
            )

    def test_stats_are_complete(self) -> None:
        builder = ExecutionPlanBuilder(
            prefer_gpu=False,
        )
        result = builder.build(self._graph())
        stats = result.stats()

        self.assertIn(
            "activation_memory",
            stats,
        )
        self.assertIn("weights", stats)
        self.assertEqual(
            stats["weights"][
                "preloaded_weight_count"
            ],
            2,
        )




class StreamPlanIntegrationTests(unittest.TestCase):
    def test_builder_attaches_optional_stream_schedule(self) -> None:
        graph = ExecutionPlanBuilderTests()._graph()
        result = ExecutionPlanBuilder(
            prefer_gpu=False,
            enable_stream_schedule=True,
            num_compute_streams=2,
        ).build(graph)

        metadata = result.execution_plan.metadata
        self.assertIn("stream_schedule", metadata)
        self.assertIn("stream_schedule_stats", metadata)
        self.assertIn("prefetch_plan", metadata)
        self.assertTrue(
            set(result.execution_plan.streams)
            <= {"compute_0", "compute_1", "transfer"}
        )

if __name__ == "__main__":
    unittest.main(verbosity=2)
