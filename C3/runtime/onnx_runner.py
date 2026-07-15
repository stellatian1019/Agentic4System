from __future__ import annotations

import ctypes
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np


def _nvidia_library_dirs() -> list[Path]:
    """Find pip-installed NVIDIA libraries for the active Python version."""
    directories: list[Path] = []
    seen: set[Path] = set()
    for entry in sys.path:
        if not entry:
            continue
        nvidia_root = Path(entry) / "nvidia"
        if not nvidia_root.is_dir():
            continue
        for candidate in nvidia_root.glob("*/lib"):
            resolved = candidate.resolve()
            if resolved not in seen:
                seen.add(resolved)
                directories.append(resolved)
    return directories


_NVIDIA_LIBRARY_DIRS = _nvidia_library_dirs()
if _NVIDIA_LIBRARY_DIRS:
    discovered_path = ":".join(str(path) for path in _NVIDIA_LIBRARY_DIRS)
    existing_path = os.environ.get("LD_LIBRARY_PATH", "")
    os.environ["LD_LIBRARY_PATH"] = (
        discovered_path
        if not existing_path
        else discovered_path + ":" + existing_path
    )

def _preload_nvidia_library(pattern: str) -> None:
    for directory in _NVIDIA_LIBRARY_DIRS:
        matches = sorted(directory.glob(pattern), reverse=True)
        if matches:
            ctypes.CDLL(str(matches[0]), mode=ctypes.RTLD_GLOBAL)
            return


# ONNX Runtime's CUDA provider resolves these libraries by SONAME. Updating
# LD_LIBRARY_PATH after process startup is insufficient on glibc, so load the
# pip-provided copies globally in dependency order before importing ORT.
for library_pattern in (
    "libcudart.so.*",
    "libnvJitLink.so.*",
    "libcublasLt.so.*",
    "libcublas.so.*",
    "libnvrtc-builtins.so.*",
    "libnvrtc.so.*",
    "libcurand.so.*",
    "libcufft.so.*",
    "libcudnn.so.*",
):
    _preload_nvidia_library(library_pattern)

import onnx
import onnxruntime as ort


class _CuPySessionView:
    def get_providers(self) -> list[str]:
        return ["CuPyExecutionProvider"]


class ONNXRunner:
    """Use ORT for ordinary models and bounded CuPy streaming for external data."""

    def __init__(
        self,
        model_path: str | Path,
        batch_size: int = 256,
    ) -> None:
        self.model_path = Path(model_path)
        self.batch_size = int(batch_size)
        self.delegate: Any | None = None

        if self._has_external_weights(self.model_path):
            from .cupy_graph_runner import CuPyGraphRunner

            self.delegate = CuPyGraphRunner(
                self.model_path,
                batch_size=self.batch_size,
            )
            self.session = _CuPySessionView()
            self.inputs = list(self.delegate.input_names)
            self.outputs = list(self.delegate.output_names)
            return

        available = ort.get_available_providers()
        providers: list[Any] = []
        if "CUDAExecutionProvider" in available:
            use_tf32 = os.getenv(
                "C3_ORT_USE_TF32",
                "0",
            ).strip().lower() not in {"0", "false", "no", "off"}
            max_workspace = os.getenv(
                "C3_CUDNN_MAX_WORKSPACE",
                "1",
            ).strip().lower() not in {"0", "false", "no", "off"}
            arena_strategy = os.getenv(
                "C3_CUDA_ARENA_STRATEGY",
                "kSameAsRequested",
            )
            providers.append(
                (
                    "CUDAExecutionProvider",
                    {
                        "use_tf32": int(use_tf32),
                        "cudnn_conv_use_max_workspace": int(max_workspace),
                        "arena_extend_strategy": arena_strategy,
                    },
                )
            )
        providers.append("CPUExecutionProvider")

        options = ort.SessionOptions()
        options.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        )
        options.enable_mem_pattern = True
        options.enable_cpu_mem_arena = True

        self.session = ort.InferenceSession(
            str(self.model_path),
            sess_options=options,
            providers=providers,
        )
        self.inputs = [item.name for item in self.session.get_inputs()]
        self.outputs = [item.name for item in self.session.get_outputs()]

    @staticmethod
    def _has_external_weights(model_path: Path) -> bool:
        model = onnx.load(
            str(model_path),
            load_external_data=False,
        )
        return any(
            initializer.external_data
            for initializer in model.graph.initializer
        )

    def run(
        self,
        inputs: dict[str, np.ndarray],
    ) -> dict[str, np.ndarray]:
        if self.delegate is not None:
            return self.delegate.run(inputs)

        sample_count = int(next(iter(inputs.values())).shape[0])
        result: dict[str, list[np.ndarray]] = {
            name: [] for name in self.outputs
        }

        for start in range(0, sample_count, self.batch_size):
            end = min(start + self.batch_size, sample_count)
            feed = {
                name: array[start:end]
                for name, array in inputs.items()
            }
            outputs = self.session.run(self.outputs, feed)
            for name, value in zip(self.outputs, outputs):
                result[name].append(value)

        return {
            name: np.concatenate(chunks, axis=0).astype(
                np.float32,
                copy=False,
            )
            for name, chunks in result.items()
        }
