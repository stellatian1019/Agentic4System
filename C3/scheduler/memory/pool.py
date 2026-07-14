from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class MemoryBlock:
    """
    A contiguous region in the logical device memory pool.

    offset_bytes:
        Byte offset inside the pool.
    size_bytes:
        Block capacity in bytes.
    is_free:
        Whether the block can be reused.
    tensor_name:
        Optional owner when allocated.
    """
    offset_bytes: int
    size_bytes: int
    is_free: bool = True
    tensor_name: str | None = None

    @property
    def end_offset_bytes(self) -> int:
        return self.offset_bytes + self.size_bytes


@dataclass(frozen=True)
class TensorAllocation:
    tensor_name: str
    offset_bytes: int
    size_bytes: int
    pool_name: str = "device"

    @property
    def end_offset_bytes(self) -> int:
        return self.offset_bytes + self.size_bytes


class DeviceMemoryPool:
    """
    Logical device memory pool with first-fit allocation and block coalescing.

    This foundation intentionally separates allocation planning from the actual
    CuPy buffer. The pool can later attach one real device buffer while keeping
    the same offsets and allocation metadata.
    """

    def __init__(
        self,
        capacity_bytes: int,
        *,
        alignment_bytes: int = 256,
        pool_name: str = "device",
    ) -> None:
        if capacity_bytes <= 0:
            raise ValueError("capacity_bytes must be positive")
        if alignment_bytes <= 0:
            raise ValueError("alignment_bytes must be positive")

        self.capacity_bytes = int(capacity_bytes)
        self.alignment_bytes = int(alignment_bytes)
        self.pool_name = pool_name

        self._blocks: list[MemoryBlock] = [
            MemoryBlock(
                offset_bytes=0,
                size_bytes=self.capacity_bytes,
                is_free=True,
            )
        ]
        self._allocations: dict[str, TensorAllocation] = {}
        self._device_buffer: Any | None = None

    def _align(self, size_bytes: int) -> int:
        if size_bytes <= 0:
            raise ValueError("size_bytes must be positive")
        alignment = self.alignment_bytes
        return ((int(size_bytes) + alignment - 1) // alignment) * alignment

    @property
    def blocks(self) -> tuple[MemoryBlock, ...]:
        return tuple(self._blocks)

    @property
    def allocations(self) -> dict[str, TensorAllocation]:
        return dict(self._allocations)

    @property
    def used_bytes(self) -> int:
        return sum(
            block.size_bytes
            for block in self._blocks
            if not block.is_free
        )

    @property
    def free_bytes(self) -> int:
        return self.capacity_bytes - self.used_bytes

    @property
    def largest_free_block_bytes(self) -> int:
        return max(
            (
                block.size_bytes
                for block in self._blocks
                if block.is_free
            ),
            default=0,
        )

    @property
    def external_fragmentation(self) -> float:
        if self.free_bytes == 0:
            return 0.0
        return 1.0 - (
            self.largest_free_block_bytes / self.free_bytes
        )

    def attach_device_buffer(self, buffer: Any) -> None:
        """
        Attach a real backend buffer later, for example a CuPy uint8 array.
        """
        self._device_buffer = buffer

    def allocate(
        self,
        tensor_name: str,
        size_bytes: int,
    ) -> TensorAllocation:
        if not tensor_name:
            raise ValueError("tensor_name must be non-empty")
        if tensor_name in self._allocations:
            raise ValueError(
                f"Tensor {tensor_name!r} already has an allocation"
            )

        requested = self._align(size_bytes)

        for index, block in enumerate(self._blocks):
            if not block.is_free or block.size_bytes < requested:
                continue

            allocation = TensorAllocation(
                tensor_name=tensor_name,
                offset_bytes=block.offset_bytes,
                size_bytes=requested,
                pool_name=self.pool_name,
            )

            remainder = block.size_bytes - requested
            allocated_block = MemoryBlock(
                offset_bytes=block.offset_bytes,
                size_bytes=requested,
                is_free=False,
                tensor_name=tensor_name,
            )

            replacement = [allocated_block]
            if remainder:
                replacement.append(
                    MemoryBlock(
                        offset_bytes=block.offset_bytes + requested,
                        size_bytes=remainder,
                        is_free=True,
                    )
                )

            self._blocks[index:index + 1] = replacement
            self._allocations[tensor_name] = allocation
            return allocation

        raise MemoryError(
            f"Cannot allocate {requested} bytes for {tensor_name!r}; "
            f"free={self.free_bytes}, "
            f"largest_free={self.largest_free_block_bytes}"
        )

    def free(self, tensor_name: str) -> None:
        allocation = self._allocations.pop(tensor_name, None)
        if allocation is None:
            raise KeyError(
                f"Tensor {tensor_name!r} is not allocated"
            )

        for block in self._blocks:
            if (
                not block.is_free
                and block.tensor_name == tensor_name
            ):
                block.is_free = True
                block.tensor_name = None
                self._coalesce()
                return

        raise RuntimeError(
            f"Allocation metadata exists but block is missing: "
            f"{tensor_name!r}"
        )

    def get_allocation(
        self,
        tensor_name: str,
    ) -> TensorAllocation:
        try:
            return self._allocations[tensor_name]
        except KeyError as exc:
            raise KeyError(
                f"Tensor {tensor_name!r} is not allocated"
            ) from exc

    def get_buffer_view(
        self,
        tensor_name: str,
    ) -> Any:
        if self._device_buffer is None:
            raise RuntimeError(
                "No device buffer is attached to this pool"
            )

        allocation = self.get_allocation(tensor_name)
        start = allocation.offset_bytes
        end = allocation.end_offset_bytes
        return self._device_buffer[start:end]

    def _coalesce(self) -> None:
        if not self._blocks:
            return

        merged: list[MemoryBlock] = [self._blocks[0]]
        for block in self._blocks[1:]:
            previous = merged[-1]
            if (
                previous.is_free
                and block.is_free
                and previous.end_offset_bytes
                == block.offset_bytes
            ):
                previous.size_bytes += block.size_bytes
            else:
                merged.append(block)

        self._blocks = merged

    def validate(self) -> bool:
        if not self._blocks:
            raise ValueError("Memory pool has no blocks")

        expected_offset = 0
        total = 0
        seen_allocated: set[str] = set()

        for block in self._blocks:
            if block.offset_bytes != expected_offset:
                raise ValueError(
                    "Memory blocks contain a gap or overlap"
                )
            if block.size_bytes <= 0:
                raise ValueError(
                    "Memory block size must be positive"
                )

            expected_offset = block.end_offset_bytes
            total += block.size_bytes

            if block.is_free:
                if block.tensor_name is not None:
                    raise ValueError(
                        "Free block cannot have a tensor owner"
                    )
            else:
                if not block.tensor_name:
                    raise ValueError(
                        "Allocated block must have a tensor owner"
                    )
                seen_allocated.add(block.tensor_name)

        if total != self.capacity_bytes:
            raise ValueError(
                f"Pool block total {total} != capacity "
                f"{self.capacity_bytes}"
            )

        if seen_allocated != set(self._allocations):
            raise ValueError(
                "Allocation metadata does not match allocated blocks"
            )

        return True

    def stats(self) -> dict[str, int | float]:
        return {
            "capacity_bytes": self.capacity_bytes,
            "used_bytes": self.used_bytes,
            "free_bytes": self.free_bytes,
            "largest_free_block_bytes": (
                self.largest_free_block_bytes
            ),
            "external_fragmentation": (
                self.external_fragmentation
            ),
            "num_blocks": len(self._blocks),
            "num_allocations": len(self._allocations),
        }
