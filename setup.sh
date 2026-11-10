#!/bin/bash
# Bootstrap paperless-paddleocr inside a paperless-ngx Docker container.
#
# Layout under /custom-cont-init.d/, which the paperless-ngx base image
# runs before starting paperless itself:
#
#   setup.sh
#   paperless-paddleocr/
#     paddle-requirements.txt
#     paperless-paddleocr.tar.gz or paperless_paddleocr-*.whl
#
# The image runs every file at the top of that directory with bash and skips
# directories, so everything but this script goes into the subdirectory.
#
# This script installs the CPU build of PaddlePaddle and PaddleOCR into the
# worker venv /opt/paperless-paddleocr/venv, a Python 3.13 venv at the path
# the plugin looks for, and the plugin into the interpreter that runs
# paperless. For local GPU acceleration use setup-gpu.sh. For remote VL
# recognition use the `vl-remote` engine pointed at a
# paddleocr-genai-vllm-server sidecar (see examples/docker-compose.vl-remote.yml).
#
# A pre-built artifact is REQUIRED in the subdirectory - paperless-paddleocr
# is not published to PyPI. The script supports both shapes:
#   * paperless-paddleocr.tar.gz   (sdist)
#   * paperless_paddleocr-*.whl    (wheel - any version, first match wins)
#
# Obtain one with either of:
#   pip wheel --no-deps "git+https://github.com/flobernd/paperless-paddleocr.git@v0.1.1"
#   # or extract from the docker/builder.Dockerfile artifact -
#   # see examples/extract-wheel/README.md
#
# Each step is skipped when its result exists, so repeated container
# restarts skip work that is already done.

set -euo pipefail

# The init of the paperless-ngx image runs this script as root with HOME
# set to the home of the paperless user. uv would leave root-owned .cache
# and .local directories there, and paddle, which runs as paperless,
# fails on import when it cannot create ~/.cache/paddle.
HOME="$(getent passwd "$(id -u)" | cut -d: -f6)"
export HOME

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ARTIFACT_DIR="${SCRIPT_DIR}/paperless-paddleocr"
REQUIREMENTS="${ARTIFACT_DIR}/paddle-requirements.txt"
# The sdist before the first wheel, and the subdirectory before the
# script's own directory, where earlier releases put the artifact. nullglob
# drops an unmatched wheel pattern instead of keeping it as a literal.
ARTIFACTS=()
shopt -s nullglob
for dir in "${ARTIFACT_DIR}" "${SCRIPT_DIR}"; do
    if [ -f "${dir}/paperless-paddleocr.tar.gz" ]; then
        ARTIFACTS+=("${dir}/paperless-paddleocr.tar.gz")
    fi
    ARTIFACTS+=("${dir}"/paperless_paddleocr-*.whl)
done
shopt -u nullglob

# shellcheck disable=SC2015 # echo does not fail, so command -v runs only without the venv
PY=$([ -x /usr/src/paperless/.venv/bin/python ] && echo /usr/src/paperless/.venv/bin/python || command -v python3)

WORKER_DIR=/opt/paperless-paddleocr
WORKER_PY="${WORKER_DIR}/venv/bin/python"
# uv would install the interpreter under /root, which the paperless user
# that runs the Celery workers cannot traverse.
export UV_PYTHON_INSTALL_DIR="${WORKER_DIR}/python"

# ---------------------------------------------------------------------------
# Native libraries (required by PaddleOCR's OpenCV backend)
# ---------------------------------------------------------------------------
NATIVE_PKGS=(
    libgl1
    libglib2.0-0
    libsm6
    libxext6
    libxrender1
    libgomp1
)

# A dry run of apt resolves a name that a renamed package provides, such as
# libglib2.0-0 on Debian 13 (libglib2.0-0t64), which dpkg -l reports as not
# installed. Without package lists it fails for a package it has to fetch.
if simulation=$(apt-get install --simulate --no-install-recommends "${NATIVE_PKGS[@]}" 2>/dev/null) \
    && ! grep -q "^Inst " <<<"${simulation}"; then
    echo "Native dependencies already present - skipping apt"
else
    echo "=== Installing native dependencies ==="
    apt-get update
    apt-get install -y --no-install-recommends "${NATIVE_PKGS[@]}"
    rm -rf /var/lib/apt/lists/*
fi

# ---------------------------------------------------------------------------
# Worker venv. PaddlePaddle supports no Python newer than 3.13, and the
# paperless-ngx 3.1 and later images run 3.14. A venv that already imports
# paddleocr is kept as it is, so this CPU script run on top of an image
# baked with paddlepaddle-gpu does not clobber the GPU wheel.
# ---------------------------------------------------------------------------
if [ -x "${WORKER_PY}" ] && "${WORKER_PY}" -c "import paddleocr" >/dev/null 2>&1; then
    echo "Worker venv already imports paddleocr - skipping"
else
    if [ ! -f "${REQUIREMENTS}" ]; then
        echo "ERROR: ${REQUIREMENTS} not found." >&2
        echo "       Put paddle-requirements.txt into ${ARTIFACT_DIR}/." >&2
        exit 1
    fi

    if command -v uv >/dev/null 2>&1; then
        UV=uv
    else
        echo "=== Installing uv into ${PY} ==="
        "${PY}" -m pip install --no-cache-dir uv
        UV="$("${PY}" -c 'import uv; print(uv.find_uv_bin())')"
    fi

    if [ -x "${WORKER_PY}" ]; then
        echo "Worker venv already exists - skipping its creation"
    else
        echo "=== Creating the Python 3.13 worker venv ==="
        "${UV}" python install 3.13
        "${UV}" venv --python 3.13 "${WORKER_DIR}/venv"
    fi

    echo "=== Installing paddlepaddle (CPU) and paddleocr into the worker venv ==="
    "${UV}" pip install --no-cache --python "${WORKER_PY}" -r "${REQUIREMENTS}"
    chmod -R a+rX "${WORKER_DIR}"
fi

# ---------------------------------------------------------------------------
# Model cache. The paperless user, which runs the Celery workers and with
# them the paddle worker, downloads the models into its ~/.paddlex. A new
# named volume mounted there belongs to root.
# ---------------------------------------------------------------------------
MODEL_CACHE=/usr/src/paperless/.paddlex

if [ -d "${MODEL_CACHE}" ] && id paperless >/dev/null 2>&1; then
    if [ "$(stat -c %U "${MODEL_CACHE}")" = paperless ]; then
        echo "Model cache ${MODEL_CACHE} already owned by paperless - skipping"
    else
        echo "=== Handing ${MODEL_CACHE} to the paperless user ==="
        chown -R paperless:paperless "${MODEL_CACHE}"
    fi
fi

# ---------------------------------------------------------------------------
# Plugin, in the interpreter that runs paperless
# ---------------------------------------------------------------------------
PACKAGE_NAME="paperless-paddleocr"

if "${PY}" -m pip show "${PACKAGE_NAME}" >/dev/null 2>&1; then
    echo "${PACKAGE_NAME} already installed - skipping pip install"
elif [ ${#ARTIFACTS[@]} -gt 0 ]; then
    echo "=== Installing ${PACKAGE_NAME} from ${ARTIFACTS[0]} ==="
    "${PY}" -m pip install --no-cache-dir "${ARTIFACTS[0]}"
else
    echo "ERROR: no paperless-paddleocr artifact found." >&2
    echo "       Put paperless-paddleocr.tar.gz or paperless_paddleocr-*.whl into ${ARTIFACT_DIR}/." >&2
    echo "       See examples/extract-wheel/README.md for how to build one." >&2
    exit 1
fi

echo "=== paperless-paddleocr bootstrap complete ==="
