from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..graph import Graph
from ..strategy import SchedulingStrategy
from ..types import HardwareSpec, PrecisionProfile
from .backend import DeviceMemoryBackend
from .execution_plan import (
    ExecutionPlan,
    ExecutionStep,
    TensorLifetime,
)
from .planner import (
    AllocationPlan,
    analyze_tensor_lifetimes,
    attach_allocation_plan,
    plan_lifetime_reuse,
)
from .pool import DeviceMemoryPool
from .prefetch import WeightPrefetchPlanner
from .stream_scheduler import StreamScheduler
from .weight_store import WeightStore


@dataclass
class BuiltExecutionPlan:
    """
    Complete C3.4 planning result.

    execution_plan:
        Ordered kernel-level execution steps.
    activation_plan:
        Lifetime-based reusable activation allocations.
    activation_pool:
        Logical activation memory pool.
    activation_backend:
        Backing byte buffer for activation allocations.
    weight_store:
        Persistent preloaded model weights.
    """
    execution_plan: ExecutionPlan
    activation_plan: AllocationPlan
    activation_pool: DeviceMemoryPool
    activation_backend: DeviceMemoryBackend
    weight_store: WeightStore

    def validate(self) -> bool:
        self.execution_plan.validate()
        self.activation_pool.validate()
        self.activation_plan.validate(
            {
                name: lifetime
                for name, lifetime in self.execution_plan.lifetimes.items()
                if name in self.activation_plan.allocations
            }
        )
        return True

    def stats(self) -> dict[str, Any]:
        return {
            "execution_steps": self.execution_plan.num_steps,
            "streams": self.execution_plan.streams,
            "activation_memory": self.activation_plan.stats(),
            "activation_pool": self.activation_pool.stats(),
            "weights": self.weight_store.stats(),
        }


class ExecutionPlanBuilder:
    """
    Build a C3.4 execution plan from a scheduled graph.

    Pipeline:
        Graph
        -> precision selection
        -> operator decomposition
        -> kernel tuning
        -> kernel-level ExecutionStep list
        -> tensor lifetime analysis
        -> activation-memory reuse plan
        -> persistent weight preload
    """

    def __init__(
        self,
        *,
        hardware: HardwareSpec | None = None,
        strategy: SchedulingStrategy | None = None,
        prefer_gpu: bool = True,
        activation_alignment_bytes: int = 256,
        weight_alignment_bytes: int = 256,
        enable_stream_schedule: bool = False,
        num_compute_streams: int = 2,
    ) -> None:
        self.hardware = hardware or HardwareSpec()
        self.strategy = strategy or SchedulingStrategy(
            hardware=self.hardware,
            autotune_mode="off",
        )
        self.prefer_gpu = bool(prefer_gpu)
        self.activation_alignment_bytes = int(
            activation_alignment_bytes
        )
        self.weight_alignment_bytes = int(
            weight_alignment_bytes
        )
        self.enable_stream_schedule = bool(enable_stream_schedule)
        self.num_compute_streams = max(1, int(num_compute_streams))

    @staticmethod
    def _precision_name(
        precision: PrecisionProfile | str,
    ) -> str:
        if isinstance(precision, PrecisionProfile):
            return precision.precision
        return str(precision)

    @staticmethod
    def _problem_size(node: Any, graph: Graph) -> int:
        for output_name in node.outputs:
            try:
                numel = graph.tensor_numel(output_name)
            except Exception:
                numel = None

            if isinstance(numel, int) and numel > 0:
                return numel

        return 1

    def _build_kernel_steps(
        self,
        graph: Graph,
    ) -> tuple[list[ExecutionStep], dict[str, Any]]:
        steps: list[ExecutionStep] = []
        node_summaries: list[dict[str, Any]] = []
        step_id = 0

        for node_index, node in enumerate(graph.nodes):
            precision = self.strategy.select_precision(
                node,
                graph,
            )
            kernels = self.strategy.decompose(
                node,
                graph,
                precision,
            )

            if not kernels:
                raise ValueError(
                    f"Node {node.name!r} produced no kernels"
                )

            problem_size = self._problem_size(node, graph)
            kernel_names: list[str] = []

            for kernel_index, ref in enumerate(kernels):
                tuning = self.strategy.tune_kernel(
                    ref,
                    precision,
                    problem_size,
                )
                tuning.validate(
                    max_threads_per_block=(
                        self.hardware.max_threads_per_block
                    ),
                    max_smem_bytes=self.hardware.smem_bytes,
                )

                step = ExecutionStep(
                    step_id=step_id,
                    node_name=node.name,
                    op_type=node.op_type,
                    kernel_name=ref.name,
                    inputs=tuple(ref.inputs),
                    outputs=tuple(ref.outputs),
                    stream="compute",
                    attributes={
                        "node_index": node_index,
                        "kernel_index": kernel_index,
                        "precision": self._precision_name(
                            precision
                        ),
                        "block_x": tuning.block_x,
                        "grid_x": tuning.grid_x,
                        "smem_bytes": tuning.smem_bytes,
                        "kernel_attributes": dict(
                            getattr(ref, "attributes", {})
                        ),
                    },
                )
                step.validate()
                steps.append(step)
                kernel_names.append(ref.name)
                step_id += 1

            node_summaries.append(
                {
                    "node_name": node.name,
                    "op_type": node.op_type,
                    "precision": self._precision_name(
                        precision
                    ),
                    "problem_size": problem_size,
                    "kernel_names": kernel_names,
                }
            )

        return steps, {
            "nodes": node_summaries,
            "graph_node_count": len(graph.nodes),
            "kernel_step_count": len(steps),
        }

    def _build_weight_store(
        self,
        graph: Graph,
    ) -> WeightStore:
        capacity = WeightStore.required_capacity(
            graph.initializers,
            alignment_bytes=self.weight_alignment_bytes,
        )

        pool = DeviceMemoryPool(
            capacity,
            alignment_bytes=self.weight_alignment_bytes,
            pool_name="weights",
        )
        backend = DeviceMemoryBackend(
            capacity,
            prefer_gpu=self.prefer_gpu,
        )
        store = WeightStore(pool, backend)
        store.preload(graph.initializers)
        return store

    def _build_activation_memory(
        self,
        lifetimes: dict[str, TensorLifetime],
    ) -> tuple[
        AllocationPlan,
        DeviceMemoryPool,
        DeviceMemoryBackend,
    ]:
        allocation_plan = plan_lifetime_reuse(
            lifetimes,
            alignment_bytes=self.activation_alignment_bytes,
            include_weights=False,
            include_graph_inputs=False,
        )

        pool = DeviceMemoryPool(
            allocation_plan.pool_capacity_bytes,
            alignment_bytes=self.activation_alignment_bytes,
            pool_name="activation",
        )
        backend = DeviceMemoryBackend(
            allocation_plan.pool_capacity_bytes,
            prefer_gpu=self.prefer_gpu,
        )
        backend.attach_to_pool(pool)

        return allocation_plan, pool, backend

    def build(
        self,
        graph: Graph,
    ) -> BuiltExecutionPlan:
        graph.validate()

        kernel_steps, scheduling_metadata = (
            self._build_kernel_steps(graph)
        )
        lifetimes = analyze_tensor_lifetimes(graph)

        execution_plan = ExecutionPlan()
        for step in kernel_steps:
            execution_plan.add_step(step)

        for lifetime in lifetimes.values():
            execution_plan.add_lifetime(lifetime)

        (
            activation_plan,
            activation_pool,
            activation_backend,
        ) = self._build_activation_memory(lifetimes)

        attach_allocation_plan(
            execution_plan,
            activation_plan,
        )

        weight_store = self._build_weight_store(graph)

        if self.enable_stream_schedule:
            prefetch_plan = WeightPrefetchPlanner().build_for_lazy_weights(
                execution_plan,
                weight_store,
            )
            stream_schedule = StreamScheduler(
                num_compute_streams=self.num_compute_streams,
            ).build(
                execution_plan,
                prefetch_plan,
            )
            execution_plan.metadata["prefetch_plan"] = prefetch_plan.stats()
            execution_plan.metadata["stream_schedule_stats"] = (
                stream_schedule.stats()
            )

        execution_plan.metadata.update(
            {
                "scheduling": scheduling_metadata,
                "weight_store": weight_store.stats(),
                "activation_backend": (
                    activation_backend.info.name
                ),
                "activation_backend_is_gpu": (
                    activation_backend.info.is_gpu
                ),
                "lifetime_granularity": "graph_node",
            }
        )

        result = BuiltExecutionPlan(
            execution_plan=execution_plan,
            activation_plan=activation_plan,
            activation_pool=activation_pool,
            activation_backend=activation_backend,
            weight_store=weight_store,
        )
        result.validate()
        return result
