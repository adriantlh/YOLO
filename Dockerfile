FROM python:3.10-slim

# Basic envs for reliable logs and fewer .pyc files
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HOST=0.0.0.0 \
    PORT=8000 \
    PIP_DEFAULT_TIMEOUT=300 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# System deps for building some wheels (e.g., pycocotools) and OpenCV runtime
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        python3-dev \
        libglib2.0-0 \
        libgl1 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first (leverages layer caching)
COPY requirements.txt ./
# Upgrade pip, install with increased timeouts/retries and use PyTorch CPU index for reliability
RUN python -m pip install --upgrade pip && \
    pip install --retries 5 --timeout 300 \
      --extra-index-url https://download.pytorch.org/whl/cpu \
      -r requirements.txt && \
    rm -rf /root/.cache/pip

# Copy the rest of the repo
COPY . .

# Expose FastAPI port
EXPOSE 8000

# Start the server using the provided script
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD ["python", "serve/healthcheck.py"]
CMD ["bash", "serve/start.sh"]
