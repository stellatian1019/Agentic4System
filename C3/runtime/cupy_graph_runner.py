from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import os
from pathlib import Path
import sys
from typing import Any

import cupy as cp
from cupyx.scipy.special import erf
import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from scheduler import import_onnx_graph
from scheduler.memory.plan_builder import ExecutionPlanBuilder


_GIB = 1 << 30


@dataclass(frozen=True)
class ExternalWeight:
    name: str
    path: Path
    dtype: np.dtype
    shape: tuple[int, ...]
    offset: int
    nbytes: int

    def host_view(self) -> np.ndarray:
        return np.memmap(
            self.path,
            mode="r",
            dtype=self.dtype,
            offset=self.offset,
            shape=self.shape,
            order="C",
        )


class LazyWeightStore:
    """Keep a bounded set of weights on GPU and stream the rest on demand."""

    def __init__(
        self,
        weights: dict[str, ExternalWeight],
        first_use_order: list[str],
        activation_reserve_bytes: int = 0,
    ) -> None:
        self.weights = weights
        self.transfer_stream = cp.cuda.Stream(non_blocking=True)
        self.resident: dict[str, cp.ndarray] = {}
        self.pending: dict[str, tuple[cp.ndarray, cp.cuda.Event]] = {}
        self.uploaded_bytes = 0
        self.streamed_bytes = 0

        free_bytes, total_bytes = cp.cuda.runtime.memGetInfo()
        reserve_setting = os.environ.get("C3_GPU_RESERVE_GIB")
        if reserve_setting is None:
            reserve_bytes = max(
                int(activation_reserve_bytes) + 2 * _GIB,
                total_bytes // 8,
            )
        else:
            reserve_bytes = max(
                int(float(reserve_setting) * _GIB),
                total_bytes // 8,
            )
        requested = os.environ.get("C3_WEIGHT_CACHE_GIB")
        if requested is None:
            budget = max(0, free_bytes - reserve_bytes)
        else:
            budget = min(
                max(0, int(float(requested) * _GIB)),
                max(0, free_bytes - reserve_bytes),
            )
        self.budget_bytes = budget
        self._preload(first_use_order)

    def _preload(self, names: list[str]) -> None:
        used = 0
        with self.transfer_stream:
            for name in names:
                if name in self.resident:
                    continue
                weight = self.weights[name]
                if used + weight.nbytes > self.budget_bytes:
                    continue
                self.resident[name] = cp.asarray(weight.host_view())
                used += weight.nbytes
                self.uploaded_bytes += weight.nbytes
        self.transfer_stream.synchronize()
        print(
            "[c3.4] weight cache "
            f"{used / _GIB:.2f} GiB / {self.budget_bytes / _GIB:.2f} GiB; "
            f"{len(self.resident)}/{len(self.weights)} tensors resident",
            file=sys.stderr,
            flush=True,
        )

    def is_resident(self, name: str) -> bool:
        return name in self.resident

    def prefetch(self, name: str) -> None:
        if name in self.resident or name in self.pending:
            return
        weight = self.weights[name]
        with self.transfer_stream:
            array = cp.asarray(weight.host_view())
            ready = cp.cuda.Event()
            ready.record(self.transfer_stream)
        self.pending[name] = (array, ready)
        self.streamed_bytes += weight.nbytes

    def get(
        self,
        name: str,
        compute_stream: cp.cuda.Stream,
    ) -> tuple[cp.ndarray, bool]:
        resident = self.resident.get(name)
        if resident is not None:
            return resident, False
        self.prefetch(name)
        array, ready = self.pending[name]
        compute_stream.wait_event(ready)
        return array, True

    def release_streamed(
        self,
        names: set[str],
        done: cp.cuda.Event,
    ) -> None:
        if not names:
            return
        done.synchronize()
        for name in names:
            item = self.pending.pop(name, None)
            if item is not None:
                array, _ = item
                del array
                del item
        cp.get_default_memory_pool().free_all_blocks()


class CuPyGraphRunner:
    """Memory-bounded ONNX executor for BigFormer-style external-data graphs."""

    SUPPORTED_OPS = {
        "Add",
        "Constant",
        "Div",
        "Erf",
        "Gather",
        "Identity",
        "LayerNormalization",
        "MatMul",
        "Mul",
        "Reshape",
        "Softmax",
        "Split",
        "Transpose",
    }

    def __init__(self, model_path: str | Path, batch_size: int = 32) -> None:
        # Streaming weights must be returned to CUDA immediately. The default
        # CuPy cache otherwise retains every differently sized weight block.
        cp.cuda.set_allocator(None)
        self.model_path = Path(model_path).resolve()
        self.batch_size = int(batch_size)

        scheduled_graph = import_onnx_graph(self.model_path)
        self.c34_plan = ExecutionPlanBuilder(
            prefer_gpu=False,
            enable_stream_schedule=True,
            num_compute_streams=max(
                1,
                int(os.getenv("C3_NUM_COMPUTE_STREAMS", "1")),
            ),
        ).build(scheduled_graph)
        self.plan_stats = self.c34_plan.stats()
        activation_stats = self.plan_stats["activation_memory"]
        self.c34_activation_reserve_bytes = int(
            activation_stats.get("peak_bytes", 0)
        )
        print(
            "[c3.4] plan ",
             f"{self.plan_stats['execution_steps']} steps; ",
             f"activation peak {self.c34_activation_reserve_bytes / (1 << 20):.1f} MiB; ",
            f"external weights {self.plan_stats["weights"].get("external_weight_count", 0)}",
            file=sys.stderr,
        )
        self.model = onnx.load(
            str(self.model_path),
            load_external_data=False,
        )
        self.nodes = list(self.model.graph.node)
        self.input_names = [value.name for value in self.model.graph.input]
        self.output_names = [value.name for value in self.model.graph.output]
        self.compute_stream = cp.cuda.Stream(non_blocking=True)
        stream_names = self.c34_plan.execution_plan.metadata.get(
            "stream_schedule", {}
        ).get("compute_streams", ("compute_0",))
        self.compute_streams = {
            name: (
                self.compute_stream
                if index == 0
                else cp.cuda.Stream(non_blocking=True)
            )
            for index, name in enumerate(stream_names)
        }
        self.node_stream_names: dict[str, str] = {}
        for step in self.c34_plan.execution_plan.steps:
            self.node_stream_names.setdefault(step.node_name, step.stream)

        unsupported = sorted(
            {node.op_type for node in self.nodes} - self.SUPPORTED_OPS
        )
        if unsupported:
            raise NotImplementedError(
                f"CuPy runner does not support operators: {unsupported}"
            )

        self.weights, self.embedded = self._load_initializers()
        self.aliases, self.skipped_identities = self._build_weight_aliases()
        first_use = self._first_weight_use_order()
        self.weight_store = LazyWeightStore(
            self.weights,
            first_use,
            self.c34_activation_reserve_bytes,
        )
        self.base_use_counts = self._build_use_counts()

    def _load_initializers(
        self,
    ) -> tuple[dict[str, ExternalWeight], dict[str, np.ndarray]]:
        external: dict[str, ExternalWeight] = {}
        embedded: dict[str, np.ndarray] = {}
        for tensor in self.model.graph.initializer:
            metadata = {item.key: item.value for item in tensor.external_data}
            if metadata:
                location = metadata["location"]
                path = (self.model_path.parent / location).resolve()
                dtype = np.dtype(helper.tensor_dtype_to_np_dtype(tensor.data_type))
                shape = tuple(int(dim) for dim in tensor.dims)
                default_size = int(np.prod(shape, dtype=np.int64)) * dtype.itemsize
                external[tensor.name] = ExternalWeight(
                    name=tensor.name,
                    path=path,
                    dtype=dtype,
                    shape=shape,
                    offset=int(metadata.get("offset", "0")),
                    nbytes=int(metadata.get("length", str(default_size))),
                )
            else:
                embedded[tensor.name] = np.asarray(
                    numpy_helper.to_array(tensor)
                )
        return external, embedded

    def _resolve(self, name: str) -> str:
        seen: set[str] = set()
        while name in self.aliases and name not in seen:
            seen.add(name)
            name = self.aliases[name]
        return name

    def _build_weight_aliases(self) -> tuple[dict[str, str], set[int]]:
        aliases: dict[str, str] = {}
        skipped: set[int] = set()
        self.aliases = aliases
        known = set(self.weights) | set(self.embedded)
        for index, node in enumerate(self.nodes):
            if node.op_type != "Identity" or len(node.input) != 1:
                continue
            source = self._resolve(node.input[0])
            if source not in known:
                continue
            for output in node.output:
                aliases[output] = source
            skipped.add(index)
        return aliases, skipped

    def _first_weight_use_order(self) -> list[str]:
        order: list[str] = []
        seen: set[str] = set()
        for index, node in enumerate(self.nodes):
            if index in self.skipped_identities:
                continue
            for input_name in node.input:
                name = self._resolve(input_name)
                if name in self.weights and name not in seen:
                    seen.add(name)
                    order.append(name)
        return order

    def _build_use_counts(self) -> Counter[str]:
        counts: Counter[str] = Counter()
        for index, node in enumerate(self.nodes):
            if index in self.skipped_identities:
                continue
            for name in node.input:
                counts[name] += 1
        for name in self.output_names:
            counts[name] += 1
        return counts

    @staticmethod
    def _attributes(node: onnx.NodeProto) -> dict[str, Any]:
        return {
            attribute.name: helper.get_attribute_value(attribute)
            for attribute in node.attribute
        }

    @staticmethod
    def _constant(attributes: dict[str, Any]) -> cp.ndarray:
        if "value" in attributes:
            return cp.asarray(numpy_helper.to_array(attributes["value"]))
        for key in ("value_float", "value_int", "value_floats", "value_ints"):
            if key in attributes:
                return cp.asarray(attributes[key.removeprefix("value_")])
        raise ValueError("Unsupported Constant node")

    @staticmethod
    def _reshape(x: cp.ndarray, shape_value: Any, allowzero: int) -> cp.ndarray:
        shape = np.asarray(cp.asnumpy(shape_value), dtype=np.int64).tolist()
        if not allowzero:
            shape = [
                x.shape[index] if dim == 0 else int(dim)
                for index, dim in enumerate(shape)
            ]
        return cp.reshape(x, tuple(shape))

    def _execute(
        self,
        node: onnx.NodeProto,
        inputs: list[Any],
    ) -> list[cp.ndarray]:
        op = node.op_type
        attributes = self._attributes(node)

        if op == "Constant":
            return [self._constant(attributes)]
        if op == "Identity":
            return [inputs[0]]
        if op == "Add":
            return [inputs[0] + inputs[1]]
        if op == "Mul":
            return [inputs[0] * inputs[1]]
        if op == "Div":
            return [inputs[0] / inputs[1]]
        if op == "MatMul":
            return [cp.matmul(inputs[0], inputs[1])]
        if op == "Gather":
            return [
                cp.take(
                    inputs[0],
                    inputs[1].astype(cp.int64, copy=False),
                    axis=int(attributes.get("axis", 0)),
                )
            ]
        if op == "Reshape":
            return [
                self._reshape(
                    inputs[0],
                    inputs[1],
                    int(attributes.get("allowzero", 0)),
                )
            ]
        if op == "Transpose":
            permutation = attributes.get("perm")
            return [
                cp.transpose(
                    inputs[0],
                    None if permutation is None else tuple(permutation),
                )
            ]
        if op == "Split":
            axis = int(attributes.get("axis", 0))
            if len(inputs) > 1:
                sizes = np.asarray(cp.asnumpy(inputs[1]), dtype=np.int64)
                indices = np.cumsum(sizes)[:-1].tolist()
                return list(cp.split(inputs[0], indices, axis=axis))
            outputs = int(attributes.get("num_outputs", len(node.output)))
            return list(cp.array_split(inputs[0], outputs, axis=axis))
        if op == "Softmax":
            axis = int(attributes.get("axis", -1))
            shifted = inputs[0] - cp.max(inputs[0], axis=axis, keepdims=True)
            exponents = cp.exp(shifted)
            return [exponents / cp.sum(exponents, axis=axis, keepdims=True)]
        if op == "Erf":
            return [erf(inputs[0])]
        if op == "LayerNormalization":
            axis = int(attributes.get("axis", -1))
            epsilon = float(attributes.get("epsilon", 1e-5))
            axes = tuple(range(axis % inputs[0].ndim, inputs[0].ndim))
            mean = cp.mean(inputs[0], axis=axes, keepdims=True)
            variance = cp.mean(
                cp.square(inputs[0] - mean),
                axis=axes,
                keepdims=True,
            )
            inv_std = cp.reciprocal(cp.sqrt(variance + epsilon))
            normalized = (inputs[0] - mean) * inv_std
            result = normalized * inputs[1]
            if len(inputs) > 2:
                result = result + inputs[2]
            outputs = [result]
            if len(node.output) > 1:
                outputs.append(mean)
            if len(node.output) > 2:
                outputs.append(inv_std)
            return outputs
        raise NotImplementedError(f"Unsupported operator: {op}")

    def _next_streamed_weight(self, start_index: int) -> str | None:
        for index in range(start_index, len(self.nodes)):
            if index in self.skipped_identities:
                continue
            for input_name in self.nodes[index].input:
                name = self._resolve(input_name)
                if (
                    name in self.weights
                    and not self.weight_store.is_resident(name)
                    and name not in self.weight_store.pending
                ):
                    return name
        return None

    def _run_batch(self, host_inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        values: dict[str, Any] = {
            name: cp.asarray(value)
            for name, value in host_inputs.items()
        }
        for name, value in self.embedded.items():
            values[name] = cp.asarray(value)
        use_counts = self.base_use_counts.copy()
        tensor_events: dict[str, cp.cuda.Event] = {}

        for index, node in enumerate(self.nodes):
            if index in self.skipped_identities:
                continue
            stream_name = self.node_stream_names.get(
                node.name,
                "compute_0",
            )
            compute_stream = self.compute_streams.get(
                stream_name,
                self.compute_stream,
            )
            node_inputs: list[Any] = []
            transient_weights: set[str] = set()
            for original_name in node.input:
                name = self._resolve(original_name)
                ready = tensor_events.get(name)
                if ready is not None:
                    compute_stream.wait_event(ready)
                if name in values:
                    value = values[name]
                elif name in self.weights:
                    value, transient = self.weight_store.get(
                        name,
                        self.compute_stream,
                    )
                    if transient:
                        transient_weights.add(name)
                else:
                    raise KeyError(
                        f"Tensor {original_name!r} for node {node.name!r} "
                        "is unavailable"
                    )
                node_inputs.append(value)

            # Keep exactly one future weight in flight. Without this guard,
            # non-weight operators would enqueue the whole remaining model.
            if len(self.weight_store.pending) <= len(transient_weights):
                next_weight = self._next_streamed_weight(index + 1)
                if next_weight is not None:
                    self.weight_store.prefetch(next_weight)

            with compute_stream:
                node_outputs = self._execute(node, node_inputs)
                for name, value in zip(node.output, node_outputs):
                    if name:
                        values[name] = value
                done = cp.cuda.Event()
                done.record(compute_stream)
            for output_name in node.output:
                if output_name:
                    tensor_events[output_name] = done

            for original_name in node.input:
                use_counts[original_name] -= 1
                resolved = self._resolve(original_name)
                if (
                    use_counts[original_name] <= 0
                    and resolved == original_name
                    and original_name in values
                    and original_name not in self.output_names
                    and original_name not in self.embedded
                ):
                    del values[original_name]

            node_inputs.clear()
            if "value" in locals():
                del value
            self.weight_store.release_streamed(transient_weights, done)

        for stream in self.compute_streams.values():
            stream.synchronize()
        return {
            name: cp.asnumpy(values[name]).astype(np.float32, copy=False)
            for name in self.output_names
        }

    def run(self, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        sample_count = int(next(iter(inputs.values())).shape[0])
        chunks: dict[str, list[np.ndarray]] = {
            name: [] for name in self.output_names
        }
        for start in range(0, sample_count, self.batch_size):
            end = min(start + self.batch_size, sample_count)
            outputs = self._run_batch(
                {name: value[start:end] for name, value in inputs.items()}
            )
            for name, value in outputs.items():
                chunks[name].append(value)
        return {
            name: np.concatenate(values, axis=0)
            for name, values in chunks.items()
        }
