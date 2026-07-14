from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .types import HardwareSpec


@dataclass(frozen=True)
class DetectedHardware:
    spec: HardwareSpec
    device_id: int
    compute_capability: tuple[int, int]
    warp_size: int
    multiprocessor_count: int
    executable_precisions: tuple[str, ...]
    cupy_available: bool
    diagnostics: dict[str, Any]

    @property
    def fingerprint(self) -> str:
        major, minor = self.compute_capability
        return (
            f"{self.spec.name}|cc{major}{minor}|"
            f"sm{self.multiprocessor_count}|"
            f"threads{self.spec.max_threads_per_block}|"
            f"smem{self.spec.smem_bytes}"
        )


def _read_property(properties: dict[Any, Any], key: str, default: Any) -> Any:
    if key in properties:
        return properties[key]
    encoded = key.encode("utf-8")
    if encoded in properties:
        return properties[encoded]
    return default


def _decode_name(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace").rstrip("\x00")
    return str(value)


def _executable_precisions(major: int, minor: int) -> tuple[str, ...]:
    result = ["fp32"]
    if (major, minor) >= (5, 3):
        result.append("fp16")
    if (major, minor) >= (8, 9):
        result.append("fp8")
    if major >= 10:
        result.append("fp4")
    return tuple(result)


def detect_hardware(
    device_id: int = 0,
    *,
    advertise_spec_precisions: bool = True,
) -> DetectedHardware:
    try:
        import cupy as cp
    except Exception as exc:
        fallback = HardwareSpec()
        return DetectedHardware(
            spec=fallback,
            device_id=device_id,
            compute_capability=(0, 0),
            warp_size=32,
            multiprocessor_count=1,
            executable_precisions=("fp32",),
            cupy_available=False,
            diagnostics={"error": f"CuPy import failed: {exc}"},
        )

    try:
        properties = cp.cuda.runtime.getDeviceProperties(device_id)
        name = _decode_name(_read_property(properties, "name", f"cuda_device_{device_id}"))
        major = int(_read_property(properties, "major", 0))
        minor = int(_read_property(properties, "minor", 0))
        max_threads = int(_read_property(properties, "maxThreadsPerBlock", 1024))
        shared_mem = int(_read_property(properties, "sharedMemPerBlock", 48 * 1024))
        warp_size = int(_read_property(properties, "warpSize", 32))
        multiprocessors = int(_read_property(properties, "multiProcessorCount", 1))

        executable = _executable_precisions(major, minor)
        advertised = (
            ("fp32", "fp16", "fp8", "fp4")
            if advertise_spec_precisions
            else executable
        )

        spec = HardwareSpec(
            name=name,
            max_threads_per_block=max_threads,
            smem_bytes=shared_mem,
            precisions=advertised,
        )
        return DetectedHardware(
            spec=spec,
            device_id=device_id,
            compute_capability=(major, minor),
            warp_size=warp_size,
            multiprocessor_count=multiprocessors,
            executable_precisions=executable,
            cupy_available=True,
            diagnostics={
                "runtime_version": int(cp.cuda.runtime.runtimeGetVersion()),
                "driver_version": int(cp.cuda.runtime.driverGetVersion()),
            },
        )
    except Exception as exc:
        fallback = HardwareSpec()
        return DetectedHardware(
            spec=fallback,
            device_id=device_id,
            compute_capability=(0, 0),
            warp_size=32,
            multiprocessor_count=1,
            executable_precisions=("fp32",),
            cupy_available=False,
            diagnostics={"error": f"CUDA detection failed: {exc}"},
        )


def hardware_summary(detected: DetectedHardware) -> dict[str, Any]:
    return {
        "name": detected.spec.name,
        "device_id": detected.device_id,
        "compute_capability": list(detected.compute_capability),
        "max_threads_per_block": detected.spec.max_threads_per_block,
        "shared_mem_per_block": detected.spec.smem_bytes,
        "warp_size": detected.warp_size,
        "multiprocessor_count": detected.multiprocessor_count,
        "advertised_precisions": list(detected.spec.precisions),
        "executable_precisions": list(detected.executable_precisions),
        "cupy_available": detected.cupy_available,
        "fingerprint": detected.fingerprint,
        "diagnostics": detected.diagnostics,
    }
