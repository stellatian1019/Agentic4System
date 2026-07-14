from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from ..types import ExternalTensorReference
from .backend import DeviceMemoryBackend
from .pool import DeviceMemoryPool, TensorAllocation


@dataclass(frozen=True)
class WeightRecord:
    name: str
    allocation: TensorAllocation
    dtype: str
    shape: tuple[int, ...]
    nbytes: int


class WeightStore:
    """Persistent storage for small weights plus metadata for streamed weights."""

    def __init__(
        self,
        pool: DeviceMemoryPool,
        backend: DeviceMemoryBackend,
    ) -> None:
        if pool.capacity_bytes != backend.capacity_bytes:
            raise ValueError(
                "Weight pool and backend capacities must match"
            )

        self.pool = pool
        self.backend = backend
        self.backend.attach_to_pool(self.pool)
        self._records: dict[str, WeightRecord] = {}
        self._external: dict[str, ExternalTensorReference] = {}
        self._copy_count = 0
        self._uploaded_bytes = 0

    @staticmethod
    def required_capacity(
        initializers: Mapping[str, Any],
        *,
        alignment_bytes: int = 256,
    ) -> int:
        total = 0
        for value in initializers.values():
            if isinstance(value, ExternalTensorReference):
                continue
            array = np.ascontiguousarray(np.asarray(value))
            size = int(array.nbytes)
            total += (
                (size + alignment_bytes - 1)
                // alignment_bytes
                * alignment_bytes
            )
        return max(alignment_bytes, total)

    @property
    def records(self) -> dict[str, WeightRecord]:
        return dict(self._records)

    @property
    def external_records(self) -> dict[str, ExternalTensorReference]:
        return dict(self._external)

    @property
    def copy_count(self) -> int:
        return self._copy_count

    @property
    def uploaded_bytes(self) -> int:
        return self._uploaded_bytes

    @property
    def resident_bytes(self) -> int:
        return sum(
            record.allocation.size_bytes
            for record in self._records.values()
        )

    def is_resident(self, name: str) -> bool:
        return name in self._records

    def preload(
        self,
        initializers: Mapping[str, Any],
    ) -> dict[str, WeightRecord]:
        for name, value in initializers.items():
            if isinstance(value, ExternalTensorReference):
                self._external[name] = value
            else:
                self.upload(name, value)
        return self.records

    def upload(
        self,
        name: str,
        value: Any,
    ) -> WeightRecord:
        if isinstance(value, ExternalTensorReference):
            raise TypeError(
                "External weights must be streamed by the runtime"
            )
        if self.is_resident(name):
            return self._records[name]

        array = np.ascontiguousarray(np.asarray(value))
        allocation = self.pool.allocate(name, int(array.nbytes))
        self.backend.copy_from_host(array, allocation)

        record = WeightRecord(
            name=name,
            allocation=allocation,
            dtype=array.dtype.str,
            shape=tuple(int(value) for value in array.shape),
            nbytes=int(array.nbytes),
        )
        self._records[name] = record
        self._copy_count += 1
        self._uploaded_bytes += int(array.nbytes)
        return record

    def get(self, name: str) -> Any:
        if name in self._external:
            raise KeyError(
                f"External weight {name!r} is registered but not resident"
            )
        try:
            record = self._records[name]
        except KeyError as exc:
            raise KeyError(
                f"Weight {name!r} is not resident"
            ) from exc

        return self.backend.get_view(
            record.allocation,
            dtype=np.dtype(record.dtype),
            shape=record.shape,
        )

    def get_record(self, name: str) -> WeightRecord:
        try:
            return self._records[name]
        except KeyError as exc:
            raise KeyError(
                f"Weight {name!r} is not resident"
            ) from exc

    def stats(self) -> dict[str, int | str | bool]:
        external_bytes = sum(
            value.nbytes or 0
            for value in self._external.values()
        )
        return {
            "backend": self.backend.info.name,
            "gpu": self.backend.info.is_gpu,
            "preloaded_weight_count": len(self._records),
            "external_weight_count": len(self._external),
            "external_weight_bytes": external_bytes,
            "copy_count": self.copy_count,
            "uploaded_bytes": self.uploaded_bytes,
            "resident_bytes": self.resident_bytes,
            "pool_capacity_bytes": self.pool.capacity_bytes,
            "pool_free_bytes": self.pool.free_bytes,
        }
