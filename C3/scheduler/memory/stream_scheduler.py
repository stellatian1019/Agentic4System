from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .execution_plan import ExecutionPlan, ExecutionStep
from .prefetch import PrefetchPlan, PrefetchStep


@dataclass(frozen=True)
class EventDependency:
    event_name: str
    producer_stream: str
    consumer_stream: str
    producer_step_id: int | None
    consumer_step_id: int


@dataclass
class StreamSchedule:
    execution_plan: ExecutionPlan
    prefetch_plan: PrefetchPlan
    dependencies: list[EventDependency] = field(
        default_factory=list
    )
    compute_streams: tuple[str, ...] = ("compute_0",)
    transfer_stream: str = "transfer"

    def validate(self) -> bool:
        self.execution_plan.validate()
        self.prefetch_plan.validate()

        known_step_ids = {
            step.step_id
            for step in self.execution_plan.steps
        }
        known_events = {
            prefetch.ready_event
            for prefetch in self.prefetch_plan.steps
        }

        for dependency in self.dependencies:
            if dependency.consumer_step_id not in known_step_ids:
                raise ValueError(
                    "Dependency references unknown consumer step"
                )
            if dependency.event_name not in known_events:
                raise ValueError(
                    "Dependency references unknown event"
                )

        return True

    @property
    def streams(self) -> tuple[str, ...]:
        streams = set(self.compute_streams)
        streams.add(self.transfer_stream)
        return tuple(sorted(streams))

    def stats(self) -> dict[str, Any]:
        return {
            "num_execution_steps": (
                self.execution_plan.num_steps
            ),
            "num_prefetch_steps": (
                self.prefetch_plan.num_prefetches
            ),
            "num_event_dependencies": (
                len(self.dependencies)
            ),
            "compute_streams": self.compute_streams,
            "transfer_stream": self.transfer_stream,
            "all_streams": self.streams,
        }


class StreamScheduler:
    """
    Assign compute kernels to streams and attach transfer-event waits.

    Current policy:
    - one dedicated transfer stream;
    - N compute streams;
    - kernels with direct tensor dependencies stay ordered by events;
    - independent nodes may round-robin across compute streams;
    - prefetch ready events are added to the first consumer step.
    """

    def __init__(
        self,
        *,
        num_compute_streams: int = 2,
        transfer_stream: str = "transfer",
    ) -> None:
        if num_compute_streams <= 0:
            raise ValueError(
                "num_compute_streams must be positive"
            )
        if not transfer_stream:
            raise ValueError(
                "transfer_stream must be non-empty"
            )

        self.compute_streams = tuple(
            f"compute_{index}"
            for index in range(num_compute_streams)
        )
        self.transfer_stream = transfer_stream

    @staticmethod
    def _producer_map(
        execution_plan: ExecutionPlan,
    ) -> dict[str, int]:
        producers: dict[str, int] = {}
        for step in execution_plan.steps:
            for output_name in step.outputs:
                producers[output_name] = step.step_id
        return producers

    def _assign_compute_streams(
        self,
        execution_plan: ExecutionPlan,
    ) -> None:
        producers = self._producer_map(execution_plan)
        step_stream: dict[int, str] = {}
        round_robin = 0

        for step in execution_plan.steps:
            dependency_steps = {
                producers[input_name]
                for input_name in step.inputs
                if input_name in producers
            }

            if dependency_steps:
                latest_dependency = max(dependency_steps)
                stream = step_stream.get(
                    latest_dependency,
                    self.compute_streams[0],
                )
            else:
                stream = self.compute_streams[
                    round_robin % len(self.compute_streams)
                ]
                round_robin += 1

            step.stream = stream
            step_stream[step.step_id] = stream

    @staticmethod
    def _append_wait_event(
        step: ExecutionStep,
        event_name: str,
    ) -> None:
        if event_name in step.wait_for_events:
            return
        step.wait_for_events = (
            *step.wait_for_events,
            event_name,
        )

    def build(
        self,
        execution_plan: ExecutionPlan,
        prefetch_plan: PrefetchPlan,
    ) -> StreamSchedule:
        self._assign_compute_streams(execution_plan)

        dependencies: list[EventDependency] = []

        for prefetch in prefetch_plan.steps:
            consumer = execution_plan.get_step(
                prefetch.consumer_step
            )
            self._append_wait_event(
                consumer,
                prefetch.ready_event,
            )

            dependencies.append(
                EventDependency(
                    event_name=prefetch.ready_event,
                    producer_stream=prefetch.stream,
                    consumer_stream=consumer.stream,
                    producer_step_id=None,
                    consumer_step_id=consumer.step_id,
                )
            )

        execution_plan.metadata["stream_schedule"] = {
            "compute_streams": self.compute_streams,
            "transfer_stream": self.transfer_stream,
            "num_dependencies": len(dependencies),
        }

        schedule = StreamSchedule(
            execution_plan=execution_plan,
            prefetch_plan=prefetch_plan,
            dependencies=dependencies,
            compute_streams=self.compute_streams,
            transfer_stream=self.transfer_stream,
        )
        schedule.validate()
        return schedule
