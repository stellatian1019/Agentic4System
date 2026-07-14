from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from .benchmark import KernelBenchmarker
from .hardware import DetectedHardware, detect_hardware
from .tuning_cache import TuningCache, make_cache_key
from .types import (
    KernelSpecRef,
    KernelTuningParams,
    PrecisionProfile,
)


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


class AutoTuner:
    """
    Lazy, cache-first autotuning facade used by SchedulingStrategy.

    Important behavior:
    1. cache hit never launches a benchmark;
    2. benchmark objects are created only on first cache miss;
    3. any CuPy/CUDA error returns the caller-provided fallback;
    4. autotuning can be disabled through C3_AUTOTUNE=0.
    """

    def __init__(
        self,
        *,
        enabled: bool | None = None,
        cache_path: str | Path | None = None,
        device_id: int = 0,
        warmup: int = 2,
        repeat: int = 5,
    ) -> None:
        self.enabled = (
            _env_flag("C3_AUTOTUNE", False)
            if enabled is None
            else bool(enabled)
        )
        self.device_id = int(device_id)
        self.warmup = int(warmup)
        self.repeat = int(repeat)

        if cache_path is None:
            cache_path = os.getenv(
                "C3_TUNING_CACHE",
                "tuning_cache.json",
            )
        self.cache = TuningCache(cache_path)

        self._detected: DetectedHardware | None = None
        self._benchmarker: KernelBenchmarker | None = None
        self._disabled_reason: str | None = None

    @property
    def disabled_reason(self) -> str | None:
        return self._disabled_reason

    def _get_detected(self) -> DetectedHardware:
        if self._detected is None:
            self._detected = detect_hardware(self.device_id)
        return self._detected

    def _get_benchmarker(self) -> KernelBenchmarker:
        if self._benchmarker is None:
            detected = self._get_detected()
            self._benchmarker = KernelBenchmarker(
                detected,
                warmup=self.warmup,
                repeat=self.repeat,
            )
        return self._benchmarker

    def tune(
        self,
        *,
        ref: KernelSpecRef,
        precision: PrecisionProfile | str,
        problem_size: Any,
        fallback: KernelTuningParams,
    ) -> KernelTuningParams:
        if not self.enabled:
            return fallback

        try:
            detected = self._get_detected()
            if not detected.cupy_available:
                self._disabled_reason = str(detected.diagnostics)
                return fallback

            key = make_cache_key(
                kernel_name=ref.name,
                precision=precision,
                problem_size=problem_size,
                hardware_fingerprint=detected.fingerprint,
            )

            cached = self.cache.get(key)
            if cached is not None:
                cached.validate(
                    max_threads_per_block=(
                        detected.spec.max_threads_per_block
                    ),
                    max_smem_bytes=detected.spec.smem_bytes,
                )
                return cached

            benchmarker = self._get_benchmarker()
            result = benchmarker.benchmark(
                ref,
                precision,
                problem_size,
            )
            self.cache.put(
                key,
                result.params,
                elapsed_ms=result.elapsed_ms,
                metadata={
                    "family": result.family,
                    "candidate_count": result.candidate_count,
                    "benchmark_precision": (
                        result.benchmark_precision
                    ),
                },
            )
            return result.params

        except Exception as exc:
            # A scheduler must remain usable even when benchmark compilation,
            # CUDA context creation, or cache I/O fails.
            self._disabled_reason = (
                f"{type(exc).__name__}: {exc}"
            )
            return fallback
