# syntax=docker/dockerfile:1.7
#
# paperless-ngx + paperless-paddleocr (classic-cpu engine) from the local
# checkout, for the smoke run and for local verification. It follows
# examples/Dockerfile.classic-cpu, which fetches the plugin from GitHub.
#
# Build from the repository root:
#   docker build -f docker/local.Dockerfile -t paperless-paddleocr:local .
#
# PAPERLESS_TAG - paperless-ngx base image tag.

ARG PAPERLESS_TAG=latest

# --- Stage 1: build the plugin wheel from the local sources ----------------
FROM python:3.13-slim AS plugin
WORKDIR /src
COPY pyproject.toml README.md LICENSE ./
COPY paperless_paddleocr ./paperless_paddleocr
RUN pip install --no-cache-dir build \
 && python -m build --wheel --outdir /dist

# --- Stage 2: layer the worker venv + the plugin onto paperless-ngx -------
FROM ghcr.io/paperless-ngx/paperless-ngx:${PAPERLESS_TAG}

# Native libraries required by PaddleOCR's OpenCV backend.
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      libgl1 \
      libglib2.0-0 \
      libsm6 \
      libxext6 \
      libxrender1 \
      libgomp1 \
 && rm -rf /var/lib/apt/lists/*

# The base image ships uv too, at a version that varies with the tag.
COPY --from=ghcr.io/astral-sh/uv:0.13.5 /uv /uvx /usr/local/bin/

# PaddlePaddle supports no Python newer than 3.13, and the paperless-ngx 3.1
# and later images run 3.14, so PaddlePaddle gets a venv of its own at the
# path the plugin looks for. uv would install the interpreter under /root,
# which the paperless user that runs the Celery workers cannot traverse.
ENV UV_PYTHON_INSTALL_DIR=/opt/paperless-paddleocr/python
COPY paddle-requirements.txt /tmp/plugin/
RUN uv python install 3.13 \
 && uv venv --python 3.13 /opt/paperless-paddleocr/venv \
 && uv pip install --no-cache --python /opt/paperless-paddleocr/venv/bin/python \
      -r /tmp/plugin/paddle-requirements.txt \
 && chmod -R a+rX /opt/paperless-paddleocr

# The plugin goes into the interpreter that runs paperless. The paperless
# user, which runs the Celery workers and with them the paddle worker,
# downloads the models into its ~/.paddlex, and a new named volume mounted
# there takes the ownership of the image's directory.
COPY --from=plugin /dist/*.whl /tmp/plugin/
RUN PY=$([ -x /usr/src/paperless/.venv/bin/python ] && echo /usr/src/paperless/.venv/bin/python || command -v python3) \
 && "$PY" -m pip install --no-cache-dir /tmp/plugin/*.whl \
 && rm -rf /tmp/plugin \
 && install -d -o paperless -g paperless /usr/src/paperless/.paddlex
