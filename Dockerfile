FROM python:3.11-slim

WORKDIR /app

# libgl1/libglib2.0-0: OpenCV • libegl1/libgles2: MediaPipe Tasks (chạy CPU nhưng vẫn cần thư viện EGL)
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libegl1 \
    libgles2 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV MODELS_DIR=/app/models
ENV PORT=8000
EXPOSE 8000

# Render inject $PORT — phải bind đúng cổng này
CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}
