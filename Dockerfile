# -------------------------------------------------
#  Minimal image for AgriVision AI
# -------------------------------------------------
FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    APP_VERSION=2.0.0

WORKDIR /app

# Install Python dependencies (cached)
COPY requirements-docker.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir -r requirements-docker.txt

# Copy source code and model weight preserving directory structure
COPY app.py predict.py ./
COPY runs/detect/train/weights/best.pt ./runs/detect/train/weights/best.pt

EXPOSE 5000
CMD ["python","app.py"]
