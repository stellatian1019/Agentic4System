from __future__ import annotations

import unittest

import numpy as np

from scheduler.graph import Graph, GraphNode
from scheduler.memory import (
    ExecutionPlan,
    TensorLifetime,
    analyze_tensor_lifetimes,
    attach_allocation_plan,
    plan_lifetime_reuse,
)
from scheduler.types import TensorInfo


class LifetimeReusePlannerTests(unittest.TestCase):
    def test_non_overlapping_tensors_reuse_offset(self) -> None:
        lifetimes = {
            "a": TensorLifetime(
                "a",
                birth_step=0,
                death_step=1,
                size_bytes=300,
            ),
            "b": TensorLifetime(
                "b",
                birth_step=2,
                death_step=3,
                size_bytes=200,
            ),
        }

        plan = plan_lifetime_reuse(
            lifetimes,
            alignment_bytes=256,
        )

        self.assertEqual(
            plan.allocations["a"].offset_bytes,
            plan.allocations["b"].offset_bytes,
        )
        self.assertEqual(plan.reuse_count, 1)
        self.assertGreater(
            plan.bytes_saved_by_reuse,
            0,
        )
        self.assertTrue(plan.validate(lifetimes))

    def test_overlapping_tensors_do_not_share_memory(self) -> None:
        lifetimes = {
            "a": TensorLifetime(
                "a",
                birth_step=0,
                death_step=3,
                size_bytes=256,
            ),
            "b": TensorLifetime(
                "b",
                birth_step=1,
                death_step=2,
                size_bytes=256,
            ),
        }

        plan = plan_lifetime_reuse(
            lifetimes,
            alignment_bytes=256,
        )

        a = plan.allocations["a"]
        b = plan.allocations["b"]

        self.assertNotEqual(
            a.offset_bytes,
            b.offset_bytes,
        )
        self.assertTrue(plan.validate(lifetimes))

    def test_graph_lifetimes_enable_intermediate_reuse(self) -> None:
        graph = Graph(
            nodes=[
                GraphNode(
                    "op0",
                    "Relu",
                    ["x"],
                    ["t0"],
                ),
                GraphNode(
                    "op1",
                    "Relu",
                    ["t0"],
                    ["t1"],
                ),
                GraphNode(
                    "op2",
                    "Relu",
                    ["t1"],
                    ["t2"],
                ),
                GraphNode(
                    "op3",
                    "Add",
                    ["t2", "bias"],
                    ["y"],
                ),
            ],
            inputs=["x"],
            outputs=["y"],
            initializers={
                "bias": np.ones((64,), np.float32),
            },
            tensors={
                name: TensorInfo(
                    name,
                    "FLOAT",
                    (64,),
                )
                for name in (
                    "x",
                    "bias",
                    "t0",
                    "t1",
                    "t2",
                    "y",
                )
            },
        )

        lifetimes = analyze_tensor_lifetimes(graph)
        plan = plan_lifetime_reuse(
            lifetimes,
            alignment_bytes=256,
        )

        self.assertEqual(
            plan.allocations["t0"].offset_bytes,
            plan.allocations["t2"].offset_bytes,
        )
        self.assertGreaterEqual(
            plan.reuse_count,
            1,
        )
        self.assertLess(
            plan.peak_bytes,
            plan.total_tensor_bytes,
        )
        self.assertTrue(plan.validate(
            {
                name: lifetime
                for name, lifetime in lifetimes.items()
                if not lifetime.is_weight
                and not lifetime.is_graph_input
            }
        ))

    def test_attach_plan_to_execution_plan(self) -> None:
        lifetimes = {
            "a": TensorLifetime(
                "a",
                birth_step=0,
                death_step=0,
                size_bytes=256,
            ),
            "b": TensorLifetime(
                "b",
                birth_step=1,
                death_step=1,
                size_bytes=256,
            ),
        }

        allocation_plan = plan_lifetime_reuse(
            lifetimes,
            alignment_bytes=256,
        )

        execution_plan = ExecutionPlan()
        execution_plan.add_lifetime(lifetimes["a"])
        execution_plan.add_lifetime(lifetimes["b"])

        attach_allocation_plan(
            execution_plan,
            allocation_plan,
        )

        self.assertIn("a", execution_plan.allocations)
        self.assertIn(
            "memory_reuse",
            execution_plan.metadata,
        )
        self.assertTrue(execution_plan.validate())

    def test_tight_capacity_succeeds_due_to_reuse(self) -> None:
        lifetimes = {
            "a": TensorLifetime(
                "a",
                birth_step=0,
                death_step=0,
                size_bytes=512,
            ),
            "b": TensorLifetime(
                "b",
                birth_step=1,
                death_step=1,
                size_bytes=512,
            ),
            "c": TensorLifetime(
                "c",
                birth_step=2,
                death_step=2,
                size_bytes=512,
            ),
        }

        plan = plan_lifetime_reuse(
            lifetimes,
            capacity_bytes=512,
            alignment_bytes=256,
        )

        self.assertEqual(plan.peak_bytes, 512)
        self.assertEqual(plan.reuse_count, 2)
        self.assertTrue(plan.validate(lifetimes))


if __name__ == "__main__":
    unittest.main(verbosity=2)
