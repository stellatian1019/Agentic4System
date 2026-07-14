from __future__ import annotations

import unittest

import numpy as np

from scheduler.graph import Graph, GraphNode
from scheduler.memory import (
    DeviceMemoryPool,
    ExecutionPlan,
    ExecutionStep,
    TensorLifetime,
    analyze_tensor_lifetimes,
)
from scheduler.types import TensorInfo


class DeviceMemoryPoolTests(unittest.TestCase):
    def test_allocate_free_and_reuse(self) -> None:
        pool = DeviceMemoryPool(
            4096,
            alignment_bytes=256,
        )

        a = pool.allocate("a", 300)
        b = pool.allocate("b", 500)

        self.assertEqual(a.offset_bytes, 0)
        self.assertEqual(a.size_bytes, 512)
        self.assertEqual(b.offset_bytes, 512)
        self.assertEqual(b.size_bytes, 512)

        pool.free("a")
        c = pool.allocate("c", 128)

        self.assertEqual(c.offset_bytes, 0)
        self.assertTrue(pool.validate())

    def test_free_blocks_are_coalesced(self) -> None:
        pool = DeviceMemoryPool(
            2048,
            alignment_bytes=256,
        )

        pool.allocate("a", 256)
        pool.allocate("b", 256)
        pool.allocate("c", 256)

        pool.free("b")
        pool.free("a")

        self.assertEqual(
            pool.blocks[0].size_bytes,
            512,
        )
        self.assertTrue(pool.blocks[0].is_free)
        self.assertTrue(pool.validate())


class LifetimeTests(unittest.TestCase):
    def test_lifetime_overlap(self) -> None:
        a = TensorLifetime("a", 0, 2, 128)
        b = TensorLifetime("b", 3, 4, 128)
        c = TensorLifetime("c", 2, 5, 128)

        self.assertFalse(a.overlaps(b))
        self.assertTrue(a.overlaps(c))

    def test_analyze_graph_lifetimes(self) -> None:
        graph = Graph(
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
                "w": np.ones((4, 4), np.float32),
                "b": np.ones((4,), np.float32),
            },
            tensors={
                "x": TensorInfo("x", "FLOAT", (2, 4)),
                "w": TensorInfo("w", "FLOAT", (4, 4)),
                "b": TensorInfo("b", "FLOAT", (4,)),
                "t0": TensorInfo("t0", "FLOAT", (2, 4)),
                "t1": TensorInfo("t1", "FLOAT", (2, 4)),
                "y": TensorInfo("y", "FLOAT", (2, 4)),
            },
        )

        lifetimes = analyze_tensor_lifetimes(graph)

        self.assertEqual(
            (lifetimes["t0"].birth_step,
             lifetimes["t0"].death_step),
            (0, 1),
        )
        self.assertEqual(
            (lifetimes["t1"].birth_step,
             lifetimes["t1"].death_step),
            (1, 2),
        )
        self.assertTrue(lifetimes["w"].is_weight)
        self.assertEqual(
            lifetimes["w"].death_step,
            2,
        )
        self.assertTrue(lifetimes["y"].is_graph_output)


class ExecutionPlanTests(unittest.TestCase):
    def test_execution_plan_validation(self) -> None:
        plan = ExecutionPlan()
        plan.add_step(
            ExecutionStep(
                step_id=0,
                node_name="mm",
                op_type="MatMul",
                kernel_name="matmul_fp16",
                inputs=("x", "w"),
                outputs=("y",),
            )
        )
        plan.add_lifetime(
            TensorLifetime(
                "y",
                birth_step=0,
                death_step=0,
                size_bytes=256,
                is_graph_output=True,
            )
        )
        plan.set_allocation("y", {"offset_bytes": 0})

        self.assertTrue(plan.validate())
        self.assertEqual(plan.num_steps, 1)
        self.assertEqual(plan.streams, ("compute",))


if __name__ == "__main__":
    unittest.main(verbosity=2)
