from __future__ import annotations

from dataclasses import dataclass, field
from math import prod
from typing import Any, Iterable, Mapping

VALID_PRECISIONS = ("fp32", "fp16", "fp8", "fp4")


@dataclass(frozen=True)
class TensorInfo:
    name: str
    dtype: str = "UNKNOWN"
    shape: tuple[int | str | None, ...] = ()

    @property
    def numel(self) -> int | None:
        dims = []
        for dim in self.shape:
            if not isinstance(dim, int) or dim < 0:
                return None
            dims.append(dim)
        return prod(dims) if dims else 1


@dataclass(frozen=True)
class ExternalTensorReference:
    """Metadata for an ONNX initializer stored in an external data file."""

    name: str
    dtype: str
    shape: tuple[int, ...]
    location: str
    offset: int = 0
    length: int | None = None

    @property
    def nbytes(self) -> int | None:
        return self.length


@dataclass(frozen=True)
class PrecisionProfile:
    precision: str
    accumulator_precision: str = "fp32"
    reason: str = ""

    def __post_init__(self) -> None:
        if self.precision not in VALID_PRECISIONS:
            raise ValueError(
                f"Unsupported precision {self.precision!r}; "
                f"expected one of {VALID_PRECISIONS}"
            )


@dataclass(frozen=True)
class KernelSpecRef:
    name: str
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    attributes: Mapping[str, Any] = field(default_factory=dict)

    @property
    def kernel_name(self) -> str:
        return self.name

    @property
    def kernel_id(self) -> str:
        return self.name


@dataclass(frozen=True)
class KernelTuningParams:
    block_x: int
    grid_x: int
    smem_bytes: int

    def validate(
        self,
        *,
        max_threads_per_block: int,
        max_smem_bytes: int,
    ) -> None:
        if not 0 < self.block_x <= max_threads_per_block:
            raise ValueError(
                f"block_x={self.block_x} is outside "
                f"(0, {max_threads_per_block}]"
            )
        if self.grid_x <= 0:
            raise ValueError(f"grid_x must be positive, got {self.grid_x}")
        if self.smem_bytes != -1 and self.smem_bytes > max_smem_bytes:
            raise ValueError(
                f"smem_bytes={self.smem_bytes} exceeds {max_smem_bytes}"
            )


@dataclass(frozen=True)
class ProblemSize:
    output_elements: int = 1
    m: int | None = None
    n: int | None = None
    k: int | None = None
    extra: Mapping[str, Any] = field(default_factory=dict)

    def normalized_output_elements(self) -> int:
        return max(1, int(self.output_elements))


@dataclass
class HardwareSpec:
    name: str = "generic_cuda_gpu"
    max_threads_per_block: int = 1024
    smem_bytes: int = 48 * 1024
    precisions: tuple[str, ...] = VALID_PRECISIONS

    def supported_precisions(self) -> set[str]:
        return set(self.precisions)

    def choose_supported(self, preferred: Iterable[str]) -> str:
        supported = self.supported_precisions()
        for precision in preferred:
            if precision in supported:
                return precision
        if "fp32" in supported:
            return "fp32"
        if not supported:
            raise RuntimeError("Hardware reports no supported precision")
        return sorted(supported)[0]
