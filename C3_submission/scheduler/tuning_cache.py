from __future__ import annotations

import json
import os
import tempfile
import threading
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from .types import KernelTuningParams, PrecisionProfile, ProblemSize

CACHE_VERSION = 1


def _precision_name(precision: PrecisionProfile | str) -> str:
    return precision.precision if isinstance(precision, PrecisionProfile) else str(precision)


def _problem_signature(problem_size: Any) -> str:
    if isinstance(problem_size, ProblemSize):
        return ",".join(
            (
                f"elements={problem_size.output_elements}",
                f"m={problem_size.m}",
                f"n={problem_size.n}",
                f"k={problem_size.k}",
            )
        )
    if isinstance(problem_size, Mapping):
        ordered = sorted((str(k), problem_size[k]) for k in problem_size)
        return ",".join(f"{key}={value}" for key, value in ordered)
    if isinstance(problem_size, Sequence) and not isinstance(problem_size, (str, bytes)):
        return "shape=" + "x".join(str(value) for value in problem_size)
    return f"size={problem_size}"


def make_cache_key(
    *,
    kernel_name: str,
    precision: PrecisionProfile | str,
    problem_size: Any,
    hardware_fingerprint: str,
) -> str:
    return "|".join(
        (
            hardware_fingerprint,
            kernel_name,
            _precision_name(precision),
            _problem_signature(problem_size),
        )
    )


class TuningCache:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._lock = threading.RLock()
        self._data: dict[str, Any] = {"version": CACHE_VERSION, "entries": {}}
        self.load()

    def load(self) -> None:
        with self._lock:
            if not self.path.exists():
                return
            try:
                payload = json.loads(self.path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                return
            if payload.get("version") != CACHE_VERSION:
                return
            if isinstance(payload.get("entries"), dict):
                self._data = payload

    def get(self, key: str) -> KernelTuningParams | None:
        with self._lock:
            record = self._data["entries"].get(key)
            if not isinstance(record, dict):
                return None
            params = record.get("params", record)
            try:
                return KernelTuningParams(
                    block_x=int(params["block_x"]),
                    grid_x=int(params["grid_x"]),
                    smem_bytes=int(params["smem_bytes"]),
                )
            except (KeyError, TypeError, ValueError):
                return None

    def get_record(self, key: str) -> dict[str, Any] | None:
        with self._lock:
            record = self._data["entries"].get(key)
            return dict(record) if isinstance(record, dict) else None

    def put(
        self,
        key: str,
        params: KernelTuningParams,
        *,
        elapsed_ms: float | None = None,
        metadata: Mapping[str, Any] | None = None,
        flush: bool = True,
    ) -> None:
        with self._lock:
            self._data["entries"][key] = {
                "params": asdict(params),
                "elapsed_ms": elapsed_ms,
                "metadata": dict(metadata or {}),
            }
            if flush:
                self.save()

    def save(self) -> None:
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            payload = json.dumps(self._data, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
            fd, temporary_name = tempfile.mkstemp(
                prefix=self.path.name + ".",
                suffix=".tmp",
                dir=str(self.path.parent),
            )
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as handle:
                    handle.write(payload)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temporary_name, self.path)
            finally:
                if os.path.exists(temporary_name):
                    os.unlink(temporary_name)

    def clear(self) -> None:
        with self._lock:
            self._data = {"version": CACHE_VERSION, "entries": {}}
            self.save()

    def __len__(self) -> int:
        with self._lock:
            return len(self._data["entries"])
