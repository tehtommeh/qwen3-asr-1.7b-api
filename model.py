"""Model service for Qwen3-ASR-1.7B."""

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
from qwen_asr import Qwen3ASRModel


@dataclass
class TranscriptionResult:
    """Result from transcription."""

    text: str
    language: Optional[str]
    timestamps: Optional[list]
    processing_time: float


class ModelService:
    """Singleton service for Qwen3-ASR model."""

    _instance: Optional["ModelService"] = None
    _model: Optional[Qwen3ASRModel] = None

    def __new__(cls) -> "ModelService":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @property
    def model_name(self) -> str:
        return os.getenv("MODEL_NAME", "Qwen/Qwen3-ASR-1.7B")

    @property
    def device(self) -> str:
        return os.getenv("DEVICE", "cuda:0" if torch.cuda.is_available() else "cpu")

    @property
    def dtype(self) -> torch.dtype:
        dtype_str = os.getenv("DTYPE", "bfloat16")
        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        return dtype_map.get(dtype_str, torch.bfloat16)

    @property
    def max_inference_batch_size(self) -> int:
        return int(os.getenv("MAX_INFERENCE_BATCH_SIZE", "32"))

    @property
    def max_new_tokens(self) -> int:
        return int(os.getenv("MAX_NEW_TOKENS", "256"))

    def load_model(self) -> None:
        """Load the Qwen3-ASR model."""
        if self._model is not None:
            return

        print(f"Loading model {self.model_name}...")
        print(f"  Device: {self.device}")
        print(f"  Dtype: {self.dtype}")
        print(f"  Max batch size: {self.max_inference_batch_size}")
        print(f"  Max new tokens: {self.max_new_tokens}")

        self._model = Qwen3ASRModel.from_pretrained(
            self.model_name,
            dtype=self.dtype,
            device_map=self.device,
            max_inference_batch_size=self.max_inference_batch_size,
            max_new_tokens=self.max_new_tokens,
        )

        print("Model loaded successfully!")

    def transcribe(
        self,
        audio_path: str | Path,
        language: Optional[str] = None,
        return_time_stamps: bool = False,
    ) -> TranscriptionResult:
        """Transcribe an audio file.

        Args:
            audio_path: Path to the audio file.
            language: Optional language hint (None = auto-detect).
            return_time_stamps: Whether to return word-level timestamps.

        Returns:
            TranscriptionResult with text, language, and optional timestamps.
        """
        if self._model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        start_time = time.perf_counter()

        results = self._model.transcribe(
            audio=str(audio_path),
            language=language,
            return_time_stamps=return_time_stamps,
        )

        processing_time = time.perf_counter() - start_time

        result = results[0]

        timestamps = None
        if return_time_stamps and hasattr(result, "time_stamps") and result.time_stamps:
            timestamps = [
                {"text": ts.text, "start": ts.start_time, "end": ts.end_time}
                for ts in result.time_stamps
            ]

        return TranscriptionResult(
            text=result.text,
            language=getattr(result, "language", None),
            timestamps=timestamps,
            processing_time=processing_time,
        )

    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._model is not None


model_service = ModelService()
