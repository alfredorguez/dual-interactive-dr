FROM python:3.10-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# libgomp1: OpenMP runtime used by openTSNE/numpy.
# build-essential: fallback in case openTSNE must be compiled from source.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /usr/src/app

# Install dependencies first to take advantage of the layer cache.
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY src/ ./src/
COPY data/dataicann.mat ./data/dataicann.mat

EXPOSE 5006

# Paths are relative to the working directory (/usr/src/app).
CMD ["bokeh", "serve", "src/dualIDR.py", "--allow-websocket-origin=*", "--port=5006"]
