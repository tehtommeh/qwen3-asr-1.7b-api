# Qwen3-ASR-1.7B FastAPI Application

A production-ready FastAPI application for audio transcription using Qwen's Qwen3-ASR-1.7B model. This service provides high-quality transcription with automatic language detection for audio files.

## Table of Contents

- [Quick Start](#quick-start)
- [Overview](#overview)
- [Features](#features)
- [System Requirements](#system-requirements)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [API Documentation](#api-documentation)
- [Model Cache](#model-cache)
- [Troubleshooting](#troubleshooting)
- [Performance](#performance)
- [License](#license)

## Quick Start

**New to this repo? Start here!**

```bash
# 1. Clone the repository
git clone git@github.com:tehtommeh/qwen3-asr-1.7b-api.git
cd qwen3-asr-1.7b-api

# 2. Create model_cache directory
mkdir -p model_cache

# 3. Download the model (choose ONE option below)
```

### Option A: Let Docker Download the Model (Easiest)
Simply build and start the service. On first run, it will download the model (~4GB) automatically:

```bash
docker compose build
docker compose up -d
```

The model will download to `model_cache/` (takes 5-10 minutes). Check progress:
```bash
docker compose logs -f
```

### Option B: Pre-download the Model (Faster Startup)
If you want to download the model first and avoid waiting on first startup:

```bash
# Install Python dependencies in a virtual environment
python3 -m venv venv
source venv/bin/activate
pip install huggingface_hub

# Download the model to model_cache/
python3 << 'EOF'
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="Qwen/Qwen3-ASR-1.7B",
    local_dir="model_cache/hub/models--Qwen--Qwen3-ASR-1.7B/snapshots/main",
    local_dir_use_symlinks=False
)
print("Model downloaded successfully!")
EOF

deactivate

# Now build and start
docker compose build
docker compose up -d
```

### Verify It's Working

```bash
# Check health
curl http://localhost:8000/health

# Should return:
# {"status":"healthy","model_loaded":true,"model_name":"Qwen/Qwen3-ASR-1.7B","device":"cuda:0"}
```

**That's it!** See [Usage](#usage) section for how to transcribe audio.

> **Note**: This requires an NVIDIA GPU. See [Installation](#installation) for complete setup including NVIDIA Container Toolkit.

## Overview

Qwen3-ASR-1.7B is a state-of-the-art speech recognition model developed by Qwen that can:
- Process audio in multiple languages (30 languages, 22 Chinese dialects)
- Perform automatic language detection
- Generate precise timestamps (optional)
- Deliver high-quality transcription with a relatively small model size

This FastAPI wrapper provides a simple REST API for transcribing audio files with all the capabilities of the Qwen3-ASR model.

## Features

### Core Capabilities
- **Multi-language support**: 30 languages and 22 Chinese dialects
- **Automatic language detection**: No need to specify the language
- **Optional timestamps**: Word-level timing information (requires forced aligner)
- **Multi-format support**: WAV, MP3, M4A, FLAC, OGG, Opus, WebM

### Technical Features
- **GPU acceleration**: Optimized for NVIDIA GPUs with CUDA
- **REST API**: Simple HTTP endpoint for easy integration
- **Docker containerized**: Reproducible environment with all dependencies
- **Model caching**: Model stored locally for fast startup
- **Health monitoring**: Health check endpoints for deployment
- **Lightweight**: 1.7B parameters - much smaller than many ASR models

## System Requirements

### Hardware
- **GPU**: NVIDIA GPU with CUDA support
  - Minimum: 6GB VRAM
  - Recommended: 8GB+ VRAM for comfortable operation
- **RAM**: 8GB+ system RAM
- **Disk Space**: ~20GB free
  - Model weights: ~4GB
  - Docker images: ~15GB
  - Working space: ~1GB

### Software
- **Operating System**: Linux (Ubuntu 20.04+ recommended)
- **Docker**: Version 20.10+
- **NVIDIA Driver**: 525+ (CUDA 12.0+)
- **NVIDIA Container Toolkit**: For GPU access in Docker

## Project Structure

```
qwen3-asr-1.7b-api/
├── app.py                  # FastAPI application with /transcribe endpoint
├── model.py                # ModelService class for model management
├── Dockerfile              # Container definition using NVIDIA PyTorch 25.11
├── docker-compose.yml      # Container orchestration with GPU support
├── requirements.txt        # Python dependencies
├── .dockerignore          # Files to exclude from Docker build
├── .gitignore             # Git ignore patterns
├── model_cache/           # ⚠️ NOT IN GIT - Download separately (~4GB)
│   ├── hub/               # Created on first run or manual download
│   │   └── models--Qwen--Qwen3-ASR-1.7B/
│   │       ├── snapshots/
│   │       ├── refs/
│   │       └── blobs/
│   └── xet/
└── README.md              # This file
```

> **Important**: The `model_cache/` directory is excluded from git. You must download the model separately - see [Quick Start](#quick-start).

**`model_cache/`** ⚠️ **NOT IN GIT**
- Local storage for HuggingFace model weights (~4GB)
- **You need to download this** - see [Quick Start](#quick-start)
- Eliminates need to re-download model on container restart
- Contains model snapshots, configuration, and tokenizer files
- Mounted as volume in Docker container
- Excluded from git via `.gitignore` due to size

### Key Files

**`app.py`**
- FastAPI application entry point
- Defines three endpoints: `/`, `/health`, `/transcribe`
- Handles file upload and validation
- Returns structured JSON with transcription and metadata

**`model.py`**
- `ModelService` singleton class
- Loads Qwen3-ASR model from HuggingFace
- Manages model lifecycle (loading, inference)
- Wraps transcription results

**`Dockerfile`**
- Based on `nvcr.io/nvidia/pytorch:25.11-py3` (NVIDIA official PyTorch container)
- Installs system dependencies (ffmpeg)
- Installs qwen-asr via pip
- Copies application code and installs Python dependencies

**`docker-compose.yml`**
- Configures GPU access using NVIDIA runtime
- Mounts local `model_cache/` directory for persistent model storage
- Sets environment variables
- Configures shared memory (8GB) for model operations

### Environment Variables

Configure these in `docker-compose.yml`:

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_NAME` | `Qwen/Qwen3-ASR-1.7B` | HuggingFace model ID |
| `DEVICE` | `cuda:0` | Device to run model on |
| `DTYPE` | `bfloat16` | Model precision (bfloat16, float16, float32) |
| `MAX_INFERENCE_BATCH_SIZE` | `32` | Max batch size (-1 = unlimited) |
| `MAX_NEW_TOKENS` | `256` | Max tokens to generate (increase for long audio) |

## Installation

> **TL;DR**: If you just cloned this repo, see [Quick Start](#quick-start) for the fastest way to get running.

This section provides detailed installation steps for all prerequisites.

### Step 1: Install Docker

Follow the [official Docker installation guide](https://docs.docker.com/get-docker/) for your Linux distribution.

For Ubuntu:
```bash
# Remove old versions
sudo apt-get remove docker docker-engine docker.io containerd runc

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Add your user to docker group (optional, requires logout/login)
sudo usermod -aG docker $USER
```

### Step 2: Install NVIDIA Container Toolkit

The NVIDIA Container Toolkit allows Docker containers to access your GPU.

**For Ubuntu/Debian:**

1. Configure the repository:
```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
```

2. Install the toolkit:
```bash
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
```

3. Configure Docker to use the NVIDIA runtime:
```bash
sudo nvidia-ctk runtime configure --runtime=docker
```

4. Restart Docker:
```bash
sudo systemctl restart docker
```

5. Verify GPU access in Docker:
```bash
docker run --rm --gpus all nvidia/cuda:12.0.0-base-ubuntu20.04 nvidia-smi
```

You should see your GPU information displayed.

### Step 3: Clone or Download This Repository

```bash
git clone <your-repo-url> qwen3-asr-1.7b-api
cd qwen3-asr-1.7b-api
```

### Step 4: Build the Docker Image

```bash
docker compose build
```

This process will:
1. Download NVIDIA PyTorch container (~10GB)
2. Install system dependencies (ffmpeg)
3. Install qwen-asr Python package and dependencies
4. Install FastAPI and uvicorn

### Step 5: Start the Service

```bash
docker compose up -d
```

On **first startup**, the model will download from HuggingFace (~4GB) which takes 5-10 minutes. The model is cached in `model_cache/` so subsequent startups are fast (~30 seconds).

### Step 6: Verify Service is Running

```bash
# Check health
curl http://localhost:8000/health

# Expected output:
# {"status":"healthy","model_loaded":true,"model_name":"Qwen/Qwen3-ASR-1.7B","device":"cuda:0"}
```

## Usage

### Converting Audio Files

Qwen3-ASR supports WAV, MP3, M4A, FLAC, OGG, Opus, and WebM formats. If you have an MP4 video or unsupported format, convert it first:

```bash
# Install ffmpeg (if not already installed)
sudo apt-get install ffmpeg

# Convert MP4 to WAV
ffmpeg -i input.mp4 -vn -acodec pcm_s16le -ar 16000 -ac 1 output.wav

# Convert any audio format to WAV
ffmpeg -i input.m4a -vn -acodec pcm_s16le -ar 16000 -ac 1 output.wav
```

### Basic Transcription

```bash
curl -X POST http://localhost:8000/transcribe \
  -F "file=@your_audio.wav"
```

### With Language Hint

Provide a language hint for better accuracy if you know the language:

```bash
curl -X POST http://localhost:8000/transcribe \
  -F "file=@your_audio.wav" \
  -F "language=en"
```

### With Timestamps

Request word-level timestamps (requires forced aligner to be configured):

```bash
curl -X POST http://localhost:8000/transcribe \
  -F "file=@your_audio.wav" \
  -F "return_time_stamps=true"
```

### Response Format

```json
{
  "text": "Full transcription text...",
  "language": "en",
  "timestamps": null,
  "metadata": {
    "model": "Qwen/Qwen3-ASR-1.7B",
    "processing_time": "2.45s",
    "device": "cuda:0"
  }
}
```

### Python Client Example

```python
import requests

# Basic transcription
with open("audio.wav", "rb") as f:
    response = requests.post(
        "http://localhost:8000/transcribe",
        files={"file": f}
    )

result = response.json()

# Print transcription
print("Transcription:")
print(result["text"])
print()
print(f"Language: {result['language']}")
print(f"Processing time: {result['metadata']['processing_time']}")

# With language hint
with open("audio.wav", "rb") as f:
    response = requests.post(
        "http://localhost:8000/transcribe",
        files={"file": f},
        data={"language": "en"}
    )
```

## API Documentation

Once running, visit these URLs for interactive API documentation:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### Endpoints

#### `GET /`
Returns API information and available endpoints.

**Response:**
```json
{
  "name": "Qwen3-ASR-1.7B API",
  "version": "1.0.0",
  "model": "Qwen/Qwen3-ASR-1.7B",
  "description": "Automatic Speech Recognition API powered by Qwen3-ASR-1.7B. Supports 30 languages and 22 Chinese dialects.",
  "endpoints": {
    "/": "API information",
    "/health": "Health check",
    "/transcribe": "Transcribe audio files (POST)"
  }
}
```

#### `GET /health`
Health check endpoint for monitoring and load balancers.

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_name": "Qwen/Qwen3-ASR-1.7B",
  "device": "cuda:0"
}
```

#### `POST /transcribe`
Transcribe audio file.

**Parameters:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `file` | File | Yes | - | Audio file (wav, mp3, m4a, flac, ogg, opus, webm) |
| `language` | String | No | None | Language hint (None = auto-detect) |
| `return_time_stamps` | Boolean | No | false | Return word-level timestamps (requires forced aligner) |

**Response:** See "Response Format" section above

**Error Responses:**
- `400 Bad Request`: Invalid file format or missing file
- `500 Internal Server Error`: Model inference failure
- `503 Service Unavailable`: Model not loaded yet

## Model Cache

The Qwen3-ASR model weights (~4GB) are stored in `model_cache/`:

```
model_cache/
├── hub/
│   └── models--Qwen--Qwen3-ASR-1.7B/
│       ├── refs/
│       │   └── main              # Points to current snapshot
│       ├── snapshots/
│       │   └── <hash>/           # Model files
│       │       ├── config.json
│       │       ├── model-00001-of-00002.safetensors
│       │       ├── model-00002-of-00002.safetensors
│       │       ├── model.safetensors.index.json
│       │       ├── preprocessor_config.json
│       │       ├── tokenizer_config.json
│       │       └── tokenizer files...
│       └── blobs/                # Content-addressed storage
└── xet/                          # HuggingFace XET metadata
```

**Why Local Storage?**
- **Fast startup**: No need to download 4GB on each restart
- **Offline capability**: Run without internet after initial setup
- **Reproducibility**: Version-locked model weights
- **Portability**: Can copy entire project with model

**Managing the Cache**:

```bash
# Check cache size
du -sh model_cache/

# Clean cache (will re-download on next startup)
rm -rf model_cache/*

# Backup cache
tar -czf qwen3-asr-model-cache.tar.gz model_cache/

# Restore cache
tar -xzf qwen3-asr-model-cache.tar.gz
```

## Troubleshooting

### Container Won't Start - GPU Not Found

**Error**: `could not select device driver "nvidia" with capabilities: [[gpu]]`

**Solution**: NVIDIA Container Toolkit not installed or configured
```bash
# Install toolkit
sudo apt-get install -y nvidia-container-toolkit

# Configure Docker runtime
sudo nvidia-ctk runtime configure --runtime=docker

# Restart Docker
sudo systemctl restart docker

# Verify
docker run --rm --gpus all nvidia/cuda:12.0.0-base-ubuntu20.04 nvidia-smi
```

### Out of Memory Error

**Error**: `CUDA out of memory` or container crashes

**Solutions**:
1. Check GPU memory: `nvidia-smi`
2. Close other GPU applications
3. Reduce `MAX_NEW_TOKENS` environment variable in docker-compose.yml
4. Ensure you have at least 6GB VRAM

### Model Download Fails

**Error**: Connection errors during first startup

**Solutions**:
1. Check internet connection
2. Check HuggingFace is accessible: `curl https://huggingface.co`
3. Use VPN if HuggingFace is blocked in your region
4. Download model manually and place in `model_cache/`

### Slow Performance

**Symptoms**: Transcription takes longer than expected

**Solutions**:
1. Verify GPU is being used: Check logs for "device: cuda"
2. Check GPU utilization: `nvidia-smi` should show high GPU usage
3. Ensure flash-attention is installed (check startup logs)
4. Try shorter audio clips first to isolate issue

### Audio Format Not Supported

**Error**: `Unsupported file format`

**Solution**: Convert audio using ffmpeg
```bash
ffmpeg -i input.mp4 -vn -acodec pcm_s16le -ar 16000 -ac 1 output.wav
```

## Performance

### Benchmarks (RTX 3090, 24GB VRAM)

| Audio Duration | Processing Time | Real-time Factor |
|----------------|-----------------|------------------|
| 1.3 minutes    | 11.5 seconds    | **6.8x faster**  |
| 16.5 minutes   | 18.7 seconds    | **52.7x faster** |

**Real-time Factor**: How much faster than real-time (e.g., 52x = processes 52 minutes of audio per minute)

**Note**: Longer audio files show significantly better throughput due to fixed overhead (model warmup, file loading) being amortized over more audio, plus better batching efficiency.

### Resource Usage

- **GPU Memory**: ~4-6GB during inference
- **System RAM**: ~2-4GB
- **CPU**: Minimal (mostly I/O and preprocessing)
- **Disk I/O**: Moderate during audio loading

## Managing the Service

```bash
# Start service
docker compose up -d

# Stop service
docker compose down

# Restart service
docker compose restart

# View logs
docker compose logs -f

# View logs for last 100 lines
docker compose logs --tail 100

# Check service status
docker compose ps

# Rebuild after code changes
docker compose build && docker compose up -d

# Execute command in container
docker exec qwen3-asr <command>

# Open shell in container
docker exec -it qwen3-asr bash
```

## License

This project uses Qwen's Qwen3-ASR-1.7B model. Please refer to the [model's HuggingFace page](https://huggingface.co/Qwen/Qwen3-ASR-1.7B) for licensing information.

The FastAPI wrapper code in this repository is provided as-is for educational and research purposes.

## Resources

- [Qwen3-ASR on HuggingFace](https://huggingface.co/Qwen/Qwen3-ASR-1.7B)
- [qwen-asr Python Package](https://pypi.org/project/qwen-asr/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [NVIDIA Container Toolkit Documentation](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/index.html)

---

**Built with Qwen3-ASR-1.7B, FastAPI, and Docker**
