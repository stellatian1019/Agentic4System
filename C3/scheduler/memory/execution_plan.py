from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable


@dataclass(frozen=True)
class TensorLifetime:
    """
    Inclusive execution-step interval during which a tensor must stay alive.
    """
    tensor_name: str
    birth_step: int
    death_step: int
    size_bytes: int
    is_weight: bool = False
    is_graph_input: bool = False
    is_graph_output: bool = False

    def __post_init__(self) -> None:
        if self.birth_step < 0:
            raise ValueError("birth_step must be non-negative")
        if self.death_step < self.birth_step:
            raise ValueError(
                "death_step must be >= birth_step"
            )
        if self.size_bytes <= 0:
            raise ValueError("size_bytes must be positive")

    def overlaps(self, other: "TensorLifetime") -> bool:
        return not (
            self.death_step < other.birth_step
            or other.death_step < self.birth_step
        )


@dataclass
class ExecutionStep:
    step_id: int
    node_name: str
    op_type: str
    kernel_name: str
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    stream: str = "compute"
    wait_for_events: tuple[str, ...] = ()
    record_event: str | None = None
    attributes: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> bool:
        if self.step_id < 0:
            raise ValueError("step_id must be non-negative")
        if not self.node_name:
            raise ValueError("node_name must be non-empty")
        if not self.kernel_name:
            raise ValueError("kernel_name must be non-empty")
        if not self.stream:
            raise ValueError("stream must be non-empty")
        return True


@dataclass
class ExecutionPlan:
    steps: list[ExecutionStep] = field(default_factory=list)
    lifetimes: dict[str, TensorLifetime] = field(
        default_factory=dict
    )
    allocations: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def add_step(self, step: ExecutionStep) -> None:
        if any(
            existing.step_id == step.step_id
            for existing in self.steps
        ):
            raise ValueError(
                f"Duplicate step_id: {step.step_id}"
            )
        self.steps.append(step)
        self.steps.sort(key=lambda item: item.step_id)

    def add_lifetime(
        self,
        lifetime: TensorLifetime,
    ) -> None:
        if lifetime.tensor_name in self.lifetimes:
            raise ValueError(
                f"Duplicate tensor lifetime: "
                f"{lifetime.tensor_name}"
            )
        self.lifetimes[lifetime.tensor_name] = lifetime

    def set_allocation(
        self,
        tensor_name: str,
        allocation: Any,
    ) -> None:
        self.allocations[tensor_name] = allocation

    def get_step(self, step_id: int) -> ExecutionStep:
        for step in self.steps:
            if step.step_id == step_id:
                return step
        raise KeyError(f"Unknown step_id: {step_id}")

    def validate(self) -> bool:
        expected_ids = list(range(len(self.steps)))
        actual_ids = [step.step_id for step in self.steps]
        if actual_ids != expected_ids:
            raise ValueError(
                "Execution step IDs must be contiguous from zero"
            )

        for step in self.steps:
            step.validate()

        for tensor_name, lifetime in self.lifetimes.items():
            if tensor_name != lifetime.tensor_name:
                raise ValueError(
                    "Lifetime dictionary key mismatch"
                )

        unknown_allocations = (
            set(self.allocations) - set(self.lifetimes)
        )
        if unknown_allocations:
            raise ValueError(
                "Allocations exist without lifetimes: "
                f"{sorted(unknown_allocations)}"
            )

        return True

    @property
    def num_steps(self) -> int:
        return len(self.steps)

    @property
    def streams(self) -> tuple[str, ...]:
        return tuple(
            sorted({step.stream for step in self.steps})
        )

    def steps_for_stream(
        self,
        stream: str,
    ) -> tuple[ExecutionStep, ...]:
        return tuple(
            step
            for step in self.steps
            if step.stream == stream
        )

    def iter_steps(self) -> Iterable[ExecutionStep]:
        return iter(self.steps)
