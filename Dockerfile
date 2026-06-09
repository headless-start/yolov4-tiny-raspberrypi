# CPU-only image for the edge benchmark. Run it with a Pi-4B-like envelope so the
# container is throttled to the same core/memory budget as the target device:
#
#   docker build -t yolov4-tiny-edge .
#   docker run --rm --cpus=4 --memory=4g -v "$PWD/results:/app/results" yolov4-tiny-edge
#
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN python models/download_weights.py

ENTRYPOINT ["python", "src/benchmark.py"]
CMD ["--models", "tiny", "--profiles", "pi4b"]
