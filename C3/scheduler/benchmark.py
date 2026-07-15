from __future__ import annotations

from dataclasses import dataclass
from math import ceil, sqrt
from typing import Any, Iterable, Mapping, Sequence

from .hardware import DetectedHardware
from .types import KernelSpecRef, KernelTuningParams, PrecisionProfile, ProblemSize


@dataclass(frozen=True)
class BenchmarkResult:
    params: KernelTuningParams
    elapsed_ms: float
    family: str
    benchmark_precision: str
    candidate_count: int


_ELEMENTWISE_SOURCE = r"""
extern "C" __global__
void c3_elementwise(const float* x, float* y, const int n) {
    const int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < n) y[idx] = x[idx] * 1.000001f + 0.000001f;
}
"""

_REDUCTION_SOURCE = r"""
extern "C" __global__
void c3_reduce(const float* x, float* partial, const int n) {
    extern __shared__ float smem[];
    const int tid = threadIdx.x;
    const int idx = blockIdx.x * blockDim.x + tid;
    smem[tid] = idx < n ? x[idx] : 0.0f;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) smem[tid] += smem[tid + stride];
        __syncthreads();
    }
    if (tid == 0) partial[blockIdx.x] = smem[0];
}
"""

_MATMUL_SOURCE = r"""
extern "C" __global__
void c3_matmul(const float* a, const float* b, float* c,
               const int m, const int n, const int k) {
    const int index = blockDim.x * blockIdx.x + threadIdx.x;
    const int total = m * n;
    if (index >= total) return;
    const int row = index / n;
    const int col = index - row * n;
    float acc = 0.0f;
    for (int inner = 0; inner < k; ++inner)
        acc += a[row * k + inner] * b[inner * n + col];
    c[index] = acc;
}
"""


class KernelBenchmarker:
    def __init__(
        self,
        detected_hardware: DetectedHardware,
        *,
        warmup: int = 3,
        repeat: int = 10,
        max_elements: int = 1_048_576,
        max_matmul_dimension: int = 256,
    ) -> None:
        if not detected_hardware.cupy_available:
            raise RuntimeError("CuPy/CUDA is unavailable: " + str(detected_hardware.diagnostics))
        import cupy as cp
        self.cp = cp
        self.hardware = detected_hardware
        self.warmup = max(0, int(warmup))
        self.repeat = max(1, int(repeat))
        self.max_elements = max(1, int(max_elements))
        self.max_matmul_dimension = max(8, int(max_matmul_dimension))
        self._elementwise_kernel = cp.RawKernel(_ELEMENTWISE_SOURCE, "c3_elementwise")
        self._reduction_kernel = cp.RawKernel(_REDUCTION_SOURCE, "c3_reduce")
        self._matmul_kernel = cp.RawKernel(_MATMUL_SOURCE, "c3_matmul")

    @staticmethod
    def classify_family(kernel_name: str) -> str:
        name = kernel_name.lower()
        if name.startswith(("reduce_", "global_average", "batch_norm_", "layernorm_")):
            return "reduction"
        if name.startswith(("matmul_", "winograd_forward_gemm_")):
            return "matmul"
        return "elementwise"

    @staticmethod
    def _precision_name(precision: PrecisionProfile | str) -> str:
        return precision.precision if isinstance(precision, PrecisionProfile) else str(precision)

    def _elements(self, problem_size: Any) -> int:
        if isinstance(problem_size, ProblemSize):
            return max(1, int(problem_size.output_elements))
        if isinstance(problem_size, int) and not isinstance(problem_size, bool):
            return max(1, problem_size)
        if isinstance(problem_size, Mapping):
            for key in ("output_elements", "numel", "elements", "size"):
                value = problem_size.get(key)
                if isinstance(value, int):
                    return max(1, value)
            m, n = problem_size.get("m"), problem_size.get("n")
            if isinstance(m, int) and isinstance(n, int):
                return max(1, m * n)
        if isinstance(problem_size, Sequence) and not isinstance(problem_size, (str, bytes)):
            result, found = 1, False
            for value in problem_size:
                if isinstance(value, int) and value > 0:
                    result *= value
                    found = True
            return max(1, result) if found else 1
        return 1

    def _matmul_shape(self, problem_size: Any) -> tuple[int, int, int]:
        if isinstance(problem_size, ProblemSize):
            m, n, k = problem_size.m, problem_size.n, problem_size.k
        elif isinstance(problem_size, Mapping):
            m, n, k = problem_size.get("m"), problem_size.get("n"), problem_size.get("k")
        else:
            m = n = k = None
        if all(isinstance(v, int) and v > 0 for v in (m, n, k)):
            cap = self.max_matmul_dimension
            return min(m, cap), min(n, cap), min(k, cap)
        elements = min(self._elements(problem_size), self.max_elements)
        side = max(8, min(int(sqrt(elements)), self.max_matmul_dimension))
        return side, side, side

    def _time_launch(self, launch: Any) -> float:
        cp = self.cp
        for _ in range(self.warmup):
            launch()
        cp.cuda.Stream.null.synchronize()
        start, end = cp.cuda.Event(), cp.cuda.Event()
        start.record()
        for _ in range(self.repeat):
            launch()
        end.record()
        end.synchronize()
        return float(cp.cuda.get_elapsed_time(start, end)) / self.repeat

    def _benchmark_elementwise(self, block_x: int, problem_size: Any):
        cp = self.cp
        elements = min(self._elements(problem_size), self.max_elements)
        x = cp.random.random(elements, dtype=cp.float32)
        y = cp.empty_like(x)
        grid_x = max(1, ceil(elements / block_x))
        elapsed = self._time_launch(
            lambda: self._elementwise_kernel((grid_x,), (block_x,), (x, y, elements))
        )
        return elapsed, KernelTuningParams(block_x, grid_x, 0)

    def _benchmark_reduction(self, block_x: int, problem_size: Any):
        cp = self.cp
        elements = min(self._elements(problem_size), self.max_elements)
        x = cp.random.random(elements, dtype=cp.float32)
        grid_x = max(1, ceil(elements / block_x))
        partial = cp.empty(grid_x, dtype=cp.float32)
        smem_bytes = block_x * 4
        elapsed = self._time_launch(
            lambda: self._reduction_kernel(
                (grid_x,), (block_x,), (x, partial, elements), shared_mem=smem_bytes
            )
        )
        return elapsed, KernelTuningParams(block_x, grid_x, smem_bytes)

    def _benchmark_matmul(self, block_x: int, problem_size: Any):
        cp = self.cp
        m, n, k = self._matmul_shape(problem_size)
        a = cp.random.random((m, k), dtype=cp.float32)
        b = cp.random.random((k, n), dtype=cp.float32)
        c = cp.empty((m, n), dtype=cp.float32)
        grid_x = max(1, ceil((m * n) / block_x))
        elapsed = self._time_launch(
            lambda: self._matmul_kernel((grid_x,), (block_x,), (a, b, c, m, n, k))
        )
        return elapsed, KernelTuningParams(block_x, grid_x, 0)

    def benchmark(
        self,
        ref: KernelSpecRef,
        precision: PrecisionProfile | str,
        problem_size: Any,
        *,
        candidates: Iterable[int] = (64, 128, 256, 512),
    ) -> BenchmarkResult:
        family = self.classify_family(ref.name)
        valid_candidates = sorted({
            int(block_x)
            for block_x in candidates
            if isinstance(block_x, int)
            and block_x > 0
            and block_x <= self.hardware.spec.max_threads_per_block
            and (family != "reduction" or block_x * 4 <= self.hardware.spec.smem_bytes)
        })
        if not valid_candidates:
            raise ValueError("No valid block-size candidates")

        best_elapsed = float("inf")
        best_params = None
        for block_x in valid_candidates:
            if family == "reduction":
                elapsed, params = self._benchmark_reduction(block_x, problem_size)
            elif family == "matmul":
                elapsed, params = self._benchmark_matmul(block_x, problem_size)
            else:
                elapsed, params = self._benchmark_elementwise(block_x, problem_size)
            params.validate(
                max_threads_per_block=self.hardware.spec.max_threads_per_block,
                max_smem_bytes=self.hardware.spec.smem_bytes,
            )
            if elapsed < best_elapsed:
                best_elapsed, best_params = elapsed, params

        assert best_params is not None
        return BenchmarkResult(
            params=best_params,
            elapsed_ms=best_elapsed,
            family=family,
            benchmark_precision=self._precision_name(precision),
            candidate_count=len(valid_candidates),
        )
