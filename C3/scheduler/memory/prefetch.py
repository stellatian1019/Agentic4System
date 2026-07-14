from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

from .execution_plan import ExecutionPlan, ExecutionStep
from .weight_store import WeightStore


@dataclass(frozen=True)
class PrefetchStep:
    """
    Host-to-device prefetch request for one weight.

    issue_before_step:
        Compute step before which the copy should be issued.
    consumer_step:
        First compute step that needs the weight.
    ready_event:
        Event recorded after the transfer completes.
    """
    prefetch_id: int
    weight_name: str
    issue_before_step: int
    consumer_step: int
    stream: str
    ready_event: str
    size_bytes: int
    already_resident: bool = False


@dataclass
class PrefetchPlan:
    steps: list[PrefetchStep] = field(default_factory=list)
    lookahead_steps: int = 1
    transfer_stream: str = "transfer"

    @property
    def num_prefetches(self) -> int:
        return len(self.steps)

    @property
    def total_bytes(self) -> int:
        return sum(
            step.size_bytes
            for step in self.steps
            if not step.already_resident
        )

    @property
    def skipped_resident_count(self) -> int:
        return sum(
            1 for step in self.steps if step.already_resident
        )

    def for_consumer(
        self,
        step_id: int,
    ) -> tuple[PrefetchStep, ...]:
        return tuple(
            step
            for step in self.steps
            if step.consumer_step == step_id
        )

    def issued_at(
        self,
        step_id: int,
    ) -> tuple[PrefetchStep, ...]:
        return tuple(
            step
            for step in self.steps
            if step.issue_before_step == step_id
        )

    def validate(self) -> bool:
        seen_ids: set[int] = set()
        seen_events: set[str] = set()

        for step in self.steps:
            if step.prefetch_id in seen_ids:
                raise ValueError(
                    f"Duplicate prefetch_id: {step.prefetch_id}"
                )
            seen_ids.add(step.prefetch_id)

            if step.ready_event in seen_events:
                raise ValueError(
                    f"Duplicate ready event: {step.ready_event}"
                )
            seen_events.add(step.ready_event)

            if step.issue_before_step < 0:
                raise ValueError(
                    "issue_before_step must be non-negative"
                )
            if step.consumer_step < step.issue_before_step:
                raise ValueError(
                    "consumer_step must not precede issue step"
                )
            if not step.weight_name:
                raise ValueError(
                    "weight_name must be non-empty"
                )
            if not step.stream:
                raise ValueError(
                    "stream must be non-empty"
                )

        return True

    def stats(self) -> dict[str, int | str]:
        return {
            "num_prefetches": self.num_prefetches,
            "total_bytes": self.total_bytes,
            "skipped_resident_count": (
                self.skipped_resident_count
            ),
            "lookahead_steps": self.lookahead_steps,
            "transfer_stream": self.transfer_stream,
        }


class WeightPrefetchPlanner:
    """
    Build a deterministic weight-prefetch plan from ExecutionPlan inputs.

    The planner identifies weights by checking whether a step input exists in
    WeightStore records. Each unique weight is scheduled once, before its first
    consumer.
    """

    def __init__(
        self,
        *,
        lookahead_steps: int = 1,
        transfer_stream: str = "transfer",
        include_resident_markers: bool = False,
    ) -> None:
        if lookahead_steps < 0:
            raise ValueError(
                "lookahead_steps must be non-negative"
            )
        if not transfer_stream:
            raise ValueError(
                "transfer_stream must be non-empty"
            )

        self.lookahead_steps = int(lookahead_steps)
        self.transfer_stream = transfer_stream
        self.include_resident_markers = bool(
            include_resident_markers
        )

    @staticmethod
    def _first_consumers(
        execution_plan: ExecutionPlan,
        weight_store: WeightStore,
    ) -> dict[str, int]:
        first: dict[str, int] = {}

        for step in execution_plan.steps:
            for input_name in step.inputs:
                if not weight_store.is_resident(input_name):
                    continue
                first.setdefault(input_name, step.step_id)

        return first

    def build(
        self,
        execution_plan: ExecutionPlan,
        weight_store: WeightStore,
    ) -> PrefetchPlan:
        first_consumers = self._first_consumers(
            execution_plan,
            weight_store,
        )

        steps: list[PrefetchStep] = []
        prefetch_id = 0

        for weight_name, consumer_step in sorted(
            first_consumers.items(),
            key=lambda item: (item[1], item[0]),
        ):
            record = weight_store.get_record(weight_name)
            issue_step = max(
                0,
                consumer_step - self.lookahead_steps,
            )

            already_resident = (
                weight_store.is_resident(weight_name)
            )

            if already_resident and not self.include_resident_markers:
                # The current WeightStore preloads all model weights. In that
                # case no extra transfer is needed, but the planner can still
                # expose markers when code review wants explicit overlap intent.
                continue

            steps.append(
                PrefetchStep(
                    prefetch_id=prefetch_id,
                    weight_name=weight_name,
                    issue_before_step=issue_step,
                    consumer_step=consumer_step,
                    stream=self.transfer_stream,
                    ready_event=(
                        f"__c34_weight_ready_{prefetch_id}__"
                    ),
                    size_bytes=record.nbytes,
                    already_resident=already_resident,
                )
            )
            prefetch_id += 1

        plan = PrefetchPlan(
            steps=steps,
            lookahead_steps=self.lookahead_steps,
            transfer_stream=self.transfer_stream,
        )
        plan.validate()
        return plan

    def build_for_lazy_weights(
        self,
        execution_plan: ExecutionPlan,
        weight_store: WeightStore,
    ) -> PrefetchPlan:
        """
        Build explicit prefetch markers even when the WeightStore already
        contains the weights.

        This is useful because the current C3.4 design preloads all weights.
        A later lazy-loading variant can execute the same plan as real copies.
        """
        original = self.include_resident_markers
        try:
            self.include_resident_markers = True
            return self.build(
                execution_plan,
                weight_store,
            )
        finally:
            self.include_resident_markers = original
