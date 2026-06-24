# Speech-to-text + speaker diarization (WhisperX) — GPU and CPU capable.
#
# CUDA 12.8 base: required for RTX 50-series (Blackwell, sm_120). The cu128 torch
# wheels below match it. This image also runs CPU-only (the CPU profile just never
# touches the GPU), so one image serves both profiles.
FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/cache/hf

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip python3-venv ffmpeg git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Isolated venv (Ubuntu 24.04 is PEP-668 "externally managed").
RUN python3 -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# PyTorch built for CUDA 12.8 (Blackwell needs cu128+). Same wheels run on CPU too.
RUN pip install --no-cache-dir torch torchaudio --index-url https://download.pytorch.org/whl/cu128

# WhisperX pulls faster-whisper, ctranslate2, pyannote.audio, etc.
RUN pip install --no-cache-dir whisperx

WORKDIR /app
COPY transcribe.py /app/transcribe.py

ENTRYPOINT ["python3", "/app/transcribe.py"]
