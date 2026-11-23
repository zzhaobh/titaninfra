FROM nvidia/cuda:13.2.0-cudnn-devel-ubuntu24.04

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       python3 python3-venv python3-pip git build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH

WORKDIR /opt
RUN git init /opt/torchtitan \
    && git -C /opt/torchtitan remote add origin https://github.com/pytorch/torchtitan.git \
    && git -C /opt/torchtitan -c http.version=HTTP/1.1 fetch --depth 1 origin 692c9bb5b26f606e5fb739acd55effb6f8850bc9 \
    && git -C /opt/torchtitan checkout FETCH_HEAD

WORKDIR /workspace
COPY requirements.txt ./
RUN python -m pip install --upgrade pip \
    && python -m pip install -r requirements.txt \
    && python -m pip install --no-deps -e /opt/torchtitan

RUN git -c http.version=HTTP/1.1 clone --depth 1 https://github.com/mosaicml/streaming.git /opt/streaming \
    && cd /opt/streaming \
    && git fetch --depth 1 origin 1381ca1213fc95b6573efb44e5ec9f85c765ce1c \
    && git checkout --detach FETCH_HEAD \
    && python -m pip install --no-build-isolation --no-deps .

COPY . .
RUN python -m pip install --no-deps -e ".[data]"

ENV PYTHONPATH=/opt/torchtitan:/opt/streaming
CMD ["python"]
