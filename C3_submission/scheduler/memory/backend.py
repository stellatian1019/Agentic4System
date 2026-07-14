from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .pool import DeviceMemoryPool, TensorAllocation


try:
    import cupy as cp
except Exception:
    cp = None


@dataclass(frozen=True)
class BackendInfo:
    name: str
    is_gpu: bool
    cupy_available: bool


class DeviceMemoryBackend:
    """
    One contiguous byte buffer used by DeviceMemoryPool.

    Preferred backend:
        CuPy uint8 device buffer.

    Fallback backend:
        NumPy uint8 host buffer, so tests can run without a GPU.
    """

    def __init__(
        self,
        capacity_bytes: int,
        *,
        prefer_gpu: bool = True,
    ) -> None:
        if capacity_bytes <= 0:
            raise ValueError("capacity_bytes must be positive")

        self.capacity_bytes = int(capacity_bytes)
        self.prefer_gpu = bool(prefer_gpu)

        self._xp = np
        self._buffer: Any
        self._info: BackendInfo

        if self.prefer_gpu and cp is not None:
            try:
                self._buffer = cp.empty(
                    self.capacity_bytes,
                    dtype=cp.uint8,
                )
                self._xp = cp
                self._info = BackendInfo(
                    name="cupy",
                    is_gpu=True,
                    cupy_available=True,
                )
            except Exception:
                self._buffer = np.empty(
                    self.capacity_bytes,
                    dtype=np.uint8,
                )
                self._xp = np
                self._info = BackendInfo(
                    name="numpy",
                    is_gpu=False,
                    cupy_available=True,
                )
        else:
            self._buffer = np.empty(
                self.capacity_bytes,
                dtype=np.uint8,
            )
            self._info = BackendInfo(
                name="numpy",
                is_gpu=False,
                cupy_available=cp is not None,
            )

    @property
    def info(self) -> BackendInfo:
        return self._info

    @property
    def buffer(self) -> Any:
        return self._buffer

    @property
    def xp(self) -> Any:
        return self._xp

    def attach_to_pool(
        self,
        pool: DeviceMemoryPool,
    ) -> None:
        if pool.capacity_bytes != self.capacity_bytes:
            raise ValueError(
                "Backend capacity and pool capacity must match"
            )
        pool.attach_device_buffer(self._buffer)

    @staticmethod
    def _dtype(dtype: Any) -> np.dtype:
        return np.dtype(dtype)

    def copy_from_host(
        self,
        host_array: Any,
        allocation: TensorAllocation,
    ) -> None:
        host = np.ascontiguousarray(np.asarray(host_array))
        required = int(host.nbytes)

        if required > allocation.size_bytes:
            raise ValueError(
                f"Allocation for {allocation.tensor_name!r} is too small: "
                f"required={required}, allocated={allocation.size_bytes}"
            )

        start = allocation.offset_bytes
        end = start + required

        host_bytes = host.view(np.uint8).reshape(-1)

        if self._info.is_gpu:
            self._buffer[start:end] = cp.asarray(host_bytes)
        else:
            self._buffer[start:end] = host_bytes

    def get_view(
        self,
        allocation: TensorAllocation,
        *,
        dtype: Any,
        shape: tuple[int, ...],
    ) -> Any:
        np_dtype = self._dtype(dtype)
        required = int(
            np.prod(shape, dtype=np.int64)
            * np_dtype.itemsize
        )

        if required > allocation.size_bytes:
            raise ValueError(
                f"Requested view requires {required} bytes, "
                f"allocation has {allocation.size_bytes}"
            )

        start = allocation.offset_bytes
        end = start + required
        byte_view = self._buffer[start:end]

        if self._info.is_gpu:
            return byte_view.view(
                cp.dtype(np_dtype)
            ).reshape(shape)

        return byte_view.view(np_dtype).reshape(shape)

    def to_host(self, array: Any) -> np.ndarray:
        if self._info.is_gpu:
            return cp.asnumpy(array)
        return np.asarray(array).copy()

    def synchronize(self) -> None:
        if self._info.is_gpu:
            cp.cuda.get_current_stream().synchronize()
