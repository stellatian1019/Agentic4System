from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from math import prod
from typing import Any

from ..graph import Graph
from .execution_plan import ExecutionPlan, TensorLifetime
from .pool import DeviceMemoryPool, TensorAllocation


_DTYPE_BYTES = {
    "FLOAT": 4,
    "FLOAT32": 4,
    "FP32": 4,
    "FLOAT16": 2,
    "FP16": 2,
    "BFLOAT16": 2,
    "BF16": 2,
    "FLOAT8": 1,
    "FP8": 1,
    "INT64": 8,
    "INT32": 4,
    "INT16": 2,
    "INT8": 1,
    "UINT8": 1,
    "BOOL": 1,
}


@dataclass(frozen=True)
class ReuseDecision:
    tensor_name: str
    offset_bytes: int
    size_bytes: int
    reused: bool
    reused_from: str | None
    birth_step: int
    death_step: int


@dataclass
class AllocationPlan:
    allocations: dict[str, TensorAllocation]
    decisions: list[ReuseDecision]
    peak_bytes: int
    total_tensor_bytes: int
    bytes_saved_by_reuse: int
    reuse_count: int
    pool_capacity_bytes: int

    @property
    def reuse_ratio(self) -> float:
        if not self.decisions:
            return 0.0
        return self.reuse_count / len(self.decisions)

    @property
    def memory_saving_ratio(self) -> float:
        if self.total_tensor_bytes <= 0:
            return 0.0
        return self.bytes_saved_by_reuse / self.total_tensor_bytes

    def validate(
        self,
        lifetimes: dict[str, TensorLifetime],
    ) -> bool:
        for left_name, left_alloc in self.allocations.items():
            left_lifetime = lifetimes[left_name]

            for right_name, right_alloc in self.allocations.items():
                if left_name >= right_name:
                    continue

                right_lifetime = lifetimes[right_name]

                memory_overlaps = not (
                    left_alloc.end_offset_bytes
                    <= right_alloc.offset_bytes
                    or right_alloc.end_offset_bytes
                    <= left_alloc.offset_bytes
                )

                if memory_overlaps and left_lifetime.overlaps(
                    right_lifetime
                ):
                    raise ValueError(
                        "Overlapping lifetimes share overlapping "
                        f"memory: {left_name!r}, {right_name!r}"
                    )

        return True

    def stats(self) -> dict[str, int | float]:
        return {
            "peak_bytes": self.peak_bytes,
            "total_tensor_bytes": self.total_tensor_bytes,
            "bytes_saved_by_reuse": self.bytes_saved_by_reuse,
            "reuse_count": self.reuse_count,
            "reuse_ratio": self.reuse_ratio,
            "memory_saving_ratio": self.memory_saving_ratio,
            "pool_capacity_bytes": self.pool_capacity_bytes,
        }


def _tensor_size_bytes(
    graph: Graph,
    tensor_name: str,
) -> int:
    tensor = graph.tensors.get(tensor_name)

    shape = getattr(tensor, "shape", None)
    dtype = str(
        getattr(tensor, "dtype", "FLOAT")
    ).upper()

    if shape:
        known_dims = [
            int(value)
            for value in shape
            if isinstance(value, int) and value > 0
        ]
        numel = prod(known_dims) if known_dims else 1
    else:
        numel = graph.tensor_numel(tensor_name) or 1

    return max(
        1,
        int(numel) * _DTYPE_BYTES.get(dtype, 4),
    )


def analyze_tensor_lifetimes(
    graph: Graph,
) -> dict[str, TensorLifetime]:
    node_count = len(graph.nodes)
    final_step = max(0, node_count - 1)

    producer_step: dict[str, int] = {}
    consumer_steps: dict[str, list[int]] = defaultdict(list)

    for step_id, node in enumerate(graph.nodes):
        for tensor_name in node.outputs:
            if tensor_name:
                producer_step[tensor_name] = step_id

        for tensor_name in node.inputs:
            if tensor_name:
                consumer_steps[tensor_name].append(step_id)

    all_tensors = (
        set(graph.inputs)
        | set(graph.outputs)
        | set(graph.initializers)
        | set(producer_step)
        | set(consumer_steps)
    )

    lifetimes: dict[str, TensorLifetime] = {}

    for tensor_name in sorted(all_tensors):
        is_weight = tensor_name in graph.initializers
        is_graph_input = tensor_name in graph.inputs
        is_graph_output = tensor_name in graph.outputs

        if is_weight or is_graph_input:
            birth = 0
        else:
            birth = producer_step.get(tensor_name, 0)

        consumers = consumer_steps.get(tensor_name, [])
        death = max(consumers, default=birth)

        if is_weight or is_graph_output:
            death = max(death, final_step)

        lifetimes[tensor_name] = TensorLifetime(
            tensor_name=tensor_name,
            birth_step=birth,
            death_step=death,
            size_bytes=_tensor_size_bytes(
                graph,
                tensor_name,
            ),
            is_weight=is_weight,
            is_graph_input=is_graph_input,
            is_graph_output=is_graph_output,
        )

    return lifetimes


def _align(
    size_bytes: int,
    alignment_bytes: int,
) -> int:
    return (
        (int(size_bytes) + alignment_bytes - 1)
        // alignment_bytes
        * alignment_bytes
    )


def estimate_pool_capacity(
    lifetimes: dict[str, TensorLifetime],
    *,
    alignment_bytes: int = 256,
    include_weights: bool = True,
    include_graph_inputs: bool = True,
) -> int:
    """
    Conservative upper bound used to initialize the logical pool.

    This is the sum of aligned sizes, so allocation cannot fail even before
    reuse is applied.
    """
    selected = [
        lifetime
        for lifetime in lifetimes.values()
        if (include_weights or not lifetime.is_weight)
        and (
            include_graph_inputs
            or not lifetime.is_graph_input
        )
    ]

    total = sum(
        _align(lifetime.size_bytes, alignment_bytes)
        for lifetime in selected
    )
    return max(alignment_bytes, total)


def plan_lifetime_reuse(
    lifetimes: dict[str, TensorLifetime],
    *,
    capacity_bytes: int | None = None,
    alignment_bytes: int = 256,
    include_weights: bool = False,
    include_graph_inputs: bool = False,
) -> AllocationPlan:
    """
    Allocate tensors in birth-step order and free them immediately after their
    death step, enabling lifetime-based memory reuse.

    By default this plans intermediate/output tensors only. Weights and graph
    inputs are excluded because they are handled by persistent/preloaded
    storage in later C3.4 steps.
    """
    selected = [
        lifetime
        for lifetime in lifetimes.values()
        if (include_weights or not lifetime.is_weight)
        and (
            include_graph_inputs
            or not lifetime.is_graph_input
        )
    ]

    selected.sort(
        key=lambda item: (
            item.birth_step,
            -item.size_bytes,
            item.tensor_name,
        )
    )

    if capacity_bytes is None:
        capacity_bytes = estimate_pool_capacity(
            lifetimes,
            alignment_bytes=alignment_bytes,
            include_weights=include_weights,
            include_graph_inputs=include_graph_inputs,
        )

    pool = DeviceMemoryPool(
        capacity_bytes,
        alignment_bytes=alignment_bytes,
        pool_name="activation",
    )

    active: dict[str, TensorLifetime] = {}
    allocations: dict[str, TensorAllocation] = {}
    last_owner_by_offset: dict[int, str] = {}
    decisions: list[ReuseDecision] = []

    peak_bytes = 0
    reuse_count = 0
    total_tensor_bytes = 0

    for lifetime in selected:
        expired = [
            tensor_name
            for tensor_name, active_lifetime in active.items()
            if active_lifetime.death_step
            < lifetime.birth_step
        ]

        for tensor_name in sorted(expired):
            pool.free(tensor_name)
            active.pop(tensor_name)

        previous_offsets = {
            block.offset_bytes
            for block in pool.blocks
            if block.is_free
        }

        allocation = pool.allocate(
            lifetime.tensor_name,
            lifetime.size_bytes,
        )
        allocations[lifetime.tensor_name] = allocation
        active[lifetime.tensor_name] = lifetime

        reused_from = last_owner_by_offset.get(
            allocation.offset_bytes
        )
        reused = (
            allocation.offset_bytes in previous_offsets
            and reused_from is not None
        )

        if reused:
            reuse_count += 1

        decisions.append(
            ReuseDecision(
                tensor_name=lifetime.tensor_name,
                offset_bytes=allocation.offset_bytes,
                size_bytes=allocation.size_bytes,
                reused=reused,
                reused_from=reused_from,
                birth_step=lifetime.birth_step,
                death_step=lifetime.death_step,
            )
        )

        last_owner_by_offset[
            allocation.offset_bytes
        ] = lifetime.tensor_name

        total_tensor_bytes += allocation.size_bytes
        peak_bytes = max(peak_bytes, pool.used_bytes)
        pool.validate()

    bytes_saved = max(
        0,
        total_tensor_bytes - peak_bytes,
    )

    plan = AllocationPlan(
        allocations=allocations,
        decisions=decisions,
        peak_bytes=peak_bytes,
        total_tensor_bytes=total_tensor_bytes,
        bytes_saved_by_reuse=bytes_saved,
        reuse_count=reuse_count,
        pool_capacity_bytes=capacity_bytes,
    )
    plan.validate(
        {
            item.tensor_name: item
            for item in selected
        }
    )
    return plan


def attach_allocation_plan(
    execution_plan: ExecutionPlan,
    allocation_plan: AllocationPlan,
) -> None:
    for tensor_name, allocation in (
        allocation_plan.allocations.items()
    ):
        execution_plan.set_allocation(
            tensor_name,
            allocation,
        )

    execution_plan.metadata[
        "memory_reuse"
    ] = allocation_plan.stats()
