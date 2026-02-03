FROM nvcr.io/nvidia/pytorch:25.11-py3

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Optional: Install flash-attn for better performance
RUN pip install --no-cache-dir flash-attn --no-build-isolation || true

# Copy application code
COPY model.py app.py ./

# Expose port
EXPOSE 8000

# Create model cache directory
RUN mkdir -p /app/model_cache

# Environment variables
ENV MODEL_NAME="Qwen/Qwen3-ASR-1.7B"
ENV DEVICE="cuda:0"
ENV DTYPE="bfloat16"
ENV MAX_INFERENCE_BATCH_SIZE="32"
ENV MAX_NEW_TOKENS="256"
ENV HF_HOME="/app/model_cache"

# Run the application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
