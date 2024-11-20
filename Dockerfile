# Start with the NVIDIA CUDA image for GPU support
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu20.04

# Set environment variables for LlamaCpp
ENV CMAKE_ARGS="-DGGML_CUDA=on" FORCE_CMAKE=1

# Install system dependencies and Python 3.11
RUN apt-get update && apt-get install -y \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa -y \
    && apt-get update && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3.11-venv \
    python3-pip \
    ffmpeg \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set python3 and pip3 if needed
RUN ln -sf /usr/bin/python3.11 /usr/bin/python3 && \
    ln -sf /usr/bin/pip3 /usr/bin/pip

# Set working directory
WORKDIR /app

# Create a virtual environment in /app/.venv
RUN python3.11 -m venv .venv

# Activate the virtual environment and upgrade pip
RUN ./.venv/bin/pip install --upgrade pip

# Copy and install Python dependencies into the virtual environment
COPY requirements.txt .
RUN ./.venv/bin/pip install --no-cache-dir -r requirements.txt

# NLTK Downloads
RUN ./.venv/bin/python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('stopwords')"

# Use BuildKit secrets to securely pass the Hugging Face token, create .env, and download models
RUN --mount=type=secret,id=hf_token \
    echo "API_KEY=$(cat /run/secrets/hf_token)" > .env && \
    ./.venv/bin/huggingface-cli login --token "$(cat /run/secrets/hf_token)" && \
    ./.venv/bin/huggingface-cli download openai/whisper-large-v3-turbo && \
    ./.venv/bin/huggingface-cli download openai/whisper-large-v3 && \
    ./.venv/bin/huggingface-cli download openai/whisper-large-v2 && \
    ./.venv/bin/huggingface-cli download openai/whisper-medium && \
    ./.venv/bin/huggingface-cli download openai/whisper-small && \
    ./.venv/bin/huggingface-cli download openai/whisper-base && \
    ./.venv/bin/huggingface-cli download openai/whisper-tiny && \
    ./.venv/bin/huggingface-cli download sentence-transformers/all-MiniLM-L6-v2 && \
    ./.venv/bin/huggingface-cli download sentence-transformers/paraphrase-multilingual-mpnet-base-v2 && \
    ./.venv/bin/huggingface-cli download pyannote/speaker-diarization-3.1 && \
    mkdir -p models/gguf && \
    curl -L -o models/gguf/gemma-2-9b-it-Q8_0.gguf  \
        https://huggingface.co/bartowski/gemma-2-9b-it-GGUF/resolve/main/gemma-2-9b-it-Q8_0.gguf && \
    ./.venv/bin/huggingface-cli logout && \
    chmod 600 .env  # Secure the .env file

# Set environment variables for Hugging Face offline mode
ENV HF_HUB_OFFLINE=1

# Copy application code
COPY . .

# Make the run.sh script executable
RUN chmod +x run.sh

# Set the entrypoint to run.sh
ENTRYPOINT ["./run.sh"]