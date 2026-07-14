from __future__ import annotations

import unittest

import numpy as np

from scheduler.graph import Graph, GraphNode
from scheduler.memory import (
    ExecutionPlanBuilder,
    StreamScheduler,
    WeightPrefetchPlanner,
)
from scheduler.types import TensorInfo


class PrefetchAndStreamSchedulerTests(unittest.TestCase):
    def _graph(self) -> Graph:
        return Graph(
            nodes=[
                GraphNode(
                    "mm0",
                    "MatMul",
                    ["x", "w0"],
                    ["t0"],
                ),
                GraphNode(
                    "relu",
                    "Relu",
                    ["t0"],
                    ["t1"],
                ),
                GraphNode(
                    "mm1",
                    "MatMul",
                    ["t1", "w1"],
                    ["y"],
                ),
            ],
            inputs=["x"],
            outputs=["y"],
            initializers={
                "w0": np.ones((4, 4), np.float32),
                "w1": np.ones((4, 4), np.float32),
            },
            tensors={
                "x": TensorInfo("x", "FLOAT", (2, 4)),
                "w0": TensorInfo("w0", "FLOAT", (4, 4)),
                "w1": TensorInfo("w1", "FLOAT", (4, 4)),
                "t0": TensorInfo("t0", "FLOAT", (2, 4)),
                "t1": TensorInfo("t1", "FLOAT", (2, 4)),
                "y": TensorInfo("y", "FLOAT", (2, 4)),
            },
        )

    def test_prefetch_plan_contains_weight_markers(self) -> None:
        built = ExecutionPlanBuilder(
            prefer_gpu=False,
        ).build(self._graph())

        planner = WeightPrefetchPlanner(
            lookahead_steps=1,
            include_resident_markers=True,
        )
        plan = planner.build(
            built.execution_plan,
            built.weight_store,
        )

        names = {step.weight_name for step in plan.steps}
        self.assertEqual(names, {"w0", "w1"})
        self.assertTrue(plan.validate())

    def test_prefetch_only_first_consumer(self) -> None:
        built = ExecutionPlanBuilder(
            prefer_gpu=False,
        ).build(self._graph())

        planner = WeightPrefetchPlanner(
            lookahead_steps=2,
            include_resident_markers=True,
        )
        plan = planner.build(
            built.execution_plan,
            built.weight_store,
        )

        self.assertEqual(
            sum(step.weight_name == "w0" for step in plan.steps),
            1,
        )
        self.assertEqual(
            sum(step.weight_name == "w1" for step in plan.steps),
            1,
        )

    def test_stream_scheduler_attaches_wait_events(self) -> None:
        built = ExecutionPlanBuilder(
            prefer_gpu=False,
        ).build(self._graph())

        prefetch_plan = WeightPrefetchPlanner(
            lookahead_steps=1,
            include_resident_markers=True,
        ).build(
            built.execution_plan,
            built.weight_store,
        )

        schedule = StreamScheduler(
            num_compute_streams=2,
        ).build(
            built.execution_plan,
            prefetch_plan,
        )

        for prefetch in prefetch_plan.steps:
            consumer = schedule.execution_plan.get_step(
                prefetch.consumer_step
            )
            self.assertIn(
                prefetch.ready_event,
                consumer.wait_for_events,
            )

        self.assertTrue(schedule.validate())

    def test_dependent_steps_remain_on_same_compute_stream(self) -> None:
        built = ExecutionPlanBuilder(
            prefer_gpu=False,
        ).build(self._graph())

        prefetch_plan = WeightPrefetchPlanner(
            include_resident_markers=True,
        ).build(
            built.execution_plan,
            built.weight_store,
        )

        schedule = StreamScheduler(
            num_compute_streams=2,
        ).build(
            built.execution_plan,
            prefetch_plan,
        )

        steps = schedule.execution_plan.steps
        producer_for = {}
        for step in steps:
            for output_name in step.outputs:
                producer_for[output_name] = step

        for step in steps:
            for input_name in step.inputs:
                producer = producer_for.get(input_name)
                if producer is not None:
                    self.assertEqual(
                        producer.stream,
                        step.stream,
                    )

    def test_schedule_stats(self) -> None:
        built = ExecutionPlanBuilder(
            prefer_gpu=False,
        ).build(self._graph())

        prefetch_plan = WeightPrefetchPlanner(
            include_resident_markers=True,
        ).build(
            built.execution_plan,
            built.weight_store,
        )
        schedule = StreamScheduler(
            num_compute_streams=2,
        ).build(
            built.execution_plan,
            prefetch_plan,
        )

        stats = schedule.stats()
        self.assertEqual(
            stats["compute_streams"],
            ("compute_0", "compute_1"),
        )
        self.assertEqual(
            stats["transfer_stream"],
            "transfer",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
