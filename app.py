"""FastAPI application for Qwen3-ASR-1.7B."""

import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from model import model_service


class TranscriptionResponse(BaseModel):
    """Response model for transcription endpoint."""

    text: str
    language: Optional[str] = None
    timestamps: Optional[list] = None
    metadata: dict


class HealthResponse(BaseModel):
    """Response model for health endpoint."""

    status: str
    model_loaded: bool
    model_name: str
    device: str


class APIInfoResponse(BaseModel):
    """Response model for API info endpoint."""

    name: str
    version: str
    model: str
    description: str
    endpoints: dict


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup."""
    model_service.load_model()
    yield


app = FastAPI(
    title="Qwen3-ASR-1.7B API",
    description="FastAPI-based ASR API for Qwen3-ASR-1.7B model",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/", response_model=APIInfoResponse)
async def root():
    """Return API information."""
    return APIInfoResponse(
        name="Qwen3-ASR-1.7B API",
        version="1.0.0",
        model=model_service.model_name,
        description="Automatic Speech Recognition API powered by Qwen3-ASR-1.7B. "
        "Supports 30 languages and 22 Chinese dialects.",
        endpoints={
            "/": "API information",
            "/health": "Health check",
            "/transcribe": "Transcribe audio files (POST)",
        },
    )


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy" if model_service.is_loaded() else "loading",
        model_loaded=model_service.is_loaded(),
        model_name=model_service.model_name,
        device=model_service.device,
    )


@app.post("/transcribe", response_model=TranscriptionResponse)
async def transcribe(
    file: UploadFile = File(..., description="Audio file to transcribe"),
    language: Optional[str] = Form(
        None, description="Language hint (None = auto-detect)"
    ),
    return_time_stamps: bool = Form(
        False, description="Return word-level timestamps (requires forced aligner)"
    ),
):
    """Transcribe an audio file.

    Accepts audio files in various formats (wav, mp3, flac, etc.)
    and returns the transcribed text with optional timestamps.
    """
    if not model_service.is_loaded():
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    suffix = Path(file.filename).suffix if file.filename else ".wav"

    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name

        result = model_service.transcribe(
            audio_path=tmp_path,
            language=language,
            return_time_stamps=return_time_stamps,
        )

        Path(tmp_path).unlink(missing_ok=True)

        return TranscriptionResponse(
            text=result.text,
            language=result.language,
            timestamps=result.timestamps,
            metadata={
                "model": model_service.model_name,
                "processing_time": f"{result.processing_time:.2f}s",
                "device": model_service.device,
            },
        )

    except Exception as e:
        Path(tmp_path).unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Handle uncaught exceptions."""
    return JSONResponse(
        status_code=500,
        content={"detail": f"Internal server error: {str(exc)}"},
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
