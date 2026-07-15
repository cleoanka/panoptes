"""TensorRT detector backend (TensorRT >= 10, named-tensor API).

Runs ultralytics-exported ``.engine`` files with the same output
contract as the ONNX backend: e2e ``(B, 300, 6)`` or classic
``(B, 4+nc, N)``. Requires ``tensorrt`` plus ``cuda-python`` on a CUDA
machine; both are imported lazily so this module imports everywhere.

Engines carry no label metadata — class names default to the COCO-80
table; ``config.extra["names"] = {id: label}`` overrides for
custom-class engines.
"""

from __future__ import annotations

import contextlib
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np

from panoptes.core.config import DetectorConfig
from panoptes.core.errors import BackendUnavailableError, ConfigError, PanoptesError
from panoptes.core.types import Detection
from panoptes.detect._postprocess import (
    COCO80_NAMES,
    postprocess_output,
    preprocess_batch,
)
from panoptes.detect.base import Detector

__all__ = ["TensorRTDetector"]

_HINT = "install tensorrt-cu12 + cuda-python on the GPU server"


def _import_cudart() -> ModuleType:
    try:
        from cuda.bindings import runtime as cudart  # cuda-python >= 12.8 layout
    except ImportError:
        try:
            from cuda import cudart  # legacy cuda-python layout
        except ImportError as exc:
            raise BackendUnavailableError("tensorrt", _HINT) from exc
    return cudart


class TensorRTDetector(Detector):
    """Deserialized ``.engine`` executed via ``execute_async_v3``."""

    def __init__(self, config: DetectorConfig) -> None:
        super().__init__(config)
        try:
            import tensorrt as trt
        except ImportError as exc:
            raise BackendUnavailableError("tensorrt", _HINT) from exc
        self._cudart = _import_cudart()
        # Bind this backend to a specific GPU before any stream/buffer
        # allocation so they land on the requested device. "auto"/"cpu"
        # leave the CUDA default (device 0); "mps" is meaningless here.
        # config.half (FP16) is not honoured at load time: engine precision
        # is baked in at export (panoptes export --format engine --half).
        if config.device.startswith("cuda"):
            device_id = int(config.device.split(":", 1)[1]) if ":" in config.device else 0
            self._check(self._cudart.cudaSetDevice(device_id))
        engine_path = Path(config.model)
        if not engine_path.exists():
            raise ConfigError(f"tensorrt engine file not found: {engine_path}")
        self._trt = trt
        runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
        engine = runtime.deserialize_cuda_engine(engine_path.read_bytes())
        if engine is None:
            raise ConfigError(
                f"failed to deserialize engine {engine_path} (TensorRT version mismatch?)"
            )
        self._engine = engine
        self._context = engine.create_execution_context()
        self._input_name: str | None = None
        self._output_name: str | None = None
        for i in range(engine.num_io_tensors):
            tensor = engine.get_tensor_name(i)
            if engine.get_tensor_mode(tensor) == trt.TensorIOMode.INPUT:
                if self._input_name is None:
                    self._input_name = tensor
            elif self._output_name is None:
                self._output_name = tensor  # first output = detection head
        if self._input_name is None or self._output_name is None:
            raise ConfigError("engine must expose at least one input and one output tensor")
        input_shape = tuple(engine.get_tensor_shape(self._input_name))
        # dim 0 == -1 means a dynamic-batch profile
        self._fixed_batch = int(input_shape[0]) if input_shape and int(input_shape[0]) > 0 else None
        self._input_dtype = self._np_dtype(engine.get_tensor_dtype(self._input_name))
        self._output_dtype = self._np_dtype(engine.get_tensor_dtype(self._output_name))
        extra_names = config.extra.get("names")
        if isinstance(extra_names, dict):
            self._names: dict[int, str] = {int(k): str(v) for k, v in extra_names.items()}
        else:
            self._names = dict(COCO80_NAMES)
        self._buffers: dict[str, tuple[int, int]] = {}  # key -> (device ptr, capacity)
        self._stream: int | None = self._check(self._cudart.cudaStreamCreate())

    def _np_dtype(self, trt_dtype: Any) -> np.dtype:
        trt = self._trt
        if trt_dtype == trt.DataType.FLOAT:
            return np.dtype(np.float32)
        if trt_dtype == trt.DataType.HALF:
            return np.dtype(np.float16)
        raise ConfigError(f"unsupported engine tensor dtype: {trt_dtype}")

    @staticmethod
    def _check(result: Any) -> Any:
        """cuda-python calls return ``(err, *values)``; raise on failure."""
        if isinstance(result, tuple):
            err, *values = result
        else:
            err, values = result, []
        if int(err) != 0:
            raise PanoptesError(f"CUDA runtime call failed: {err!r}")
        if not values:
            return None
        return values[0] if len(values) == 1 else tuple(values)

    def _alloc(self, key: str, nbytes: int) -> int:
        """Cached, grow-only device allocation per IO tensor."""
        cached = self._buffers.get(key)
        if cached is None or cached[1] < nbytes:
            if cached is not None:
                self._check(self._cudart.cudaFree(cached[0]))
            ptr = self._check(self._cudart.cudaMalloc(nbytes))
            self._buffers[key] = (ptr, nbytes)
        return self._buffers[key][0]

    # ------------------------------------------------------------------
    def infer(self, frames: list[np.ndarray]) -> list[list[Detection]]:
        if not frames:
            return []
        if self._fixed_batch is not None and len(frames) > self._fixed_batch:
            results: list[list[Detection]] = []
            for start in range(0, len(frames), self._fixed_batch):
                results.extend(self.infer(frames[start : start + self._fixed_batch]))
            return results
        batch, metas = preprocess_batch(frames, self.config.imgsz)
        output = self._execute(batch)
        return postprocess_output(output, metas, frames, self.config, self._names)

    def _execute(self, batch: np.ndarray) -> np.ndarray:
        cudart = self._cudart
        n = batch.shape[0]
        run_n = self._fixed_batch if self._fixed_batch is not None else n
        if run_n != n:
            # static-batch engine, short final chunk: pad with zero frames
            padded = np.zeros((run_n, *batch.shape[1:]), dtype=batch.dtype)
            padded[:n] = batch
            batch = padded
        if self._fixed_batch is None:
            self._context.set_input_shape(self._input_name, (run_n, *batch.shape[1:]))
        batch = np.ascontiguousarray(batch.astype(self._input_dtype, copy=False))
        out_shape = tuple(self._context.get_tensor_shape(self._output_name))
        output = np.empty(out_shape, dtype=self._output_dtype)
        d_input = self._alloc("input", batch.nbytes)
        d_output = self._alloc("output", output.nbytes)
        self._context.set_tensor_address(self._input_name, d_input)
        self._context.set_tensor_address(self._output_name, d_output)
        self._check(
            cudart.cudaMemcpyAsync(
                d_input,
                batch.ctypes.data,
                batch.nbytes,
                cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
                self._stream,
            )
        )
        if not self._context.execute_async_v3(self._stream):
            raise PanoptesError("tensorrt inference execution failed")
        self._check(
            cudart.cudaMemcpyAsync(
                output.ctypes.data,
                d_output,
                output.nbytes,
                cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
                self._stream,
            )
        )
        self._check(cudart.cudaStreamSynchronize(self._stream))
        result = np.asarray(output, dtype=np.float32)
        return result[:n] if run_n != n else result

    def warmup(self) -> None:
        size = self.config.imgsz
        self.infer([np.zeros((size, size, 3), dtype=np.uint8)])

    def close(self) -> None:
        for ptr, _ in self._buffers.values():
            with contextlib.suppress(Exception):
                self._check(self._cudart.cudaFree(ptr))
        self._buffers.clear()
        if self._stream is not None:
            with contextlib.suppress(Exception):
                self._check(self._cudart.cudaStreamDestroy(self._stream))
            self._stream = None
        self._context = None
        self._engine = None

    @property
    def name(self) -> str:
        return f"tensorrt:{self.config.model}"
