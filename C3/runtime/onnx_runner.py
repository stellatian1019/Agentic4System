from __future__ import annotations

import ctypes
import os
from pathlib import Path
from typing import Any

import numpy as np


for path in (
    "/usr/local/lib/python3.12/dist-packages/nvidia/cu13/lib",
    "/usr/local/lib/python3.12/dist-packages/nvidia/cudnn/lib",
):
    if os.path.exists(path):
        os.environ["LD_LIBRARY_PATH"] = (
            path + ":" + os.environ.get("LD_LIBRARY_PATH", "")
        )


_CUDNN = "/usr/local/lib/python3.12/dist-packages/nvidia/cudnn/lib/libcudnn.so.9"
if os.path.exists(_CUDNN):
    ctypes.CDLL(_CUDNN, mode=ctypes.RTLD_GLOBAL)

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

            stream_batch_size = min(
                self.batch_size,
                int(os.environ.get("C3_STREAM_BATCH_SIZE", "64")),
            )
            self.delegate = CuPyGraphRunner(
                self.model_path,
                batch_size=stream_batch_size,
            )
            self.session = _CuPySessionView()
            self.inputs = list(self.delegate.input_names)
            self.outputs = list(self.delegate.output_names)
            return

        available = ort.get_available_providers()
        providers: list[Any] = []
        if "CUDAExecutionProvider" in available:
            providers.append(("CUDAExecutionProvider", {"use_tf32": 0}))
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

        with open("provider_debug.txt", "a", encoding="utf-8") as handle:
            handle.write(
                f"{self.model_path}: {self.session.get_providers()}\n"
            )

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
