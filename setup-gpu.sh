#!/bin/bash
# Bootstrap paperless-paddleocr inside a paperless-ngx Docker container
# with the GPU paddlepaddle wheel.
#
# Layout under /custom-cont-init.d/, which the paperless-ngx base image
# runs before starting paperless itself:
#
#   setup-gpu.sh
#   paperless-paddleocr/
#     paddle-requirements-gpu.txt
#     paperless-paddleocr.tar.gz or paperless_paddleocr-*.whl
#
# The image runs every file at the top of that directory with bash and skips
# directories, so everything but this script goes into the subdirectory.
#
# The script installs paddlepaddle-gpu and PaddleOCR into the worker venv
# /opt/paperless-paddleocr/venv, a Python 3.13 venv at the path the plugin
# looks for, and the plugin into the interpreter that runs paperless.
#
# WARNING: this script installs paddlepaddle-gpu but does NOT install the
# CUDA / cuDNN runtime libraries the wheel depends on. The paperless-ngx
# base image does not carry them, so this script alone is not enough for a
# working classic-gpu setup unless the host image was pre-built with the
# right libs. For production GPU use, bake a custom image from
# examples/Dockerfile.classic-gpu instead - it ships the CUDA runtime libs
# and the GPU wheel together. This script is best-effort, intended for
# users on managed paperless deployments where rebuilding the image is
# inconvenient.
#
# Configuration via env (set in the compose file or container env):
#   PADDLE_CUDA_WHEEL   wheel index suffix - cu126 (default), cu129, cu118.
#                       See https://www.paddlepaddle.org.cn/packages/stable/
#
# A pre-built artifact is REQUIRED in the subdirectory (same shapes as
# setup.sh):
#   * paperless-paddleocr.tar.gz   (sdist)
#   * paperless_paddleocr-*.whl    (wheel - any version, first match wins)
# Obtain one via examples/extract-wheel/README.md.
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
REQUIREMENTS="${ARTIFACT_DIR}/paddle-requirements-gpu.txt"
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

CUDA_WHEEL="${PADDLE_CUDA_WHEEL:-cu126}"
PADDLE_INDEX="https://www.paddlepaddle.org.cn/packages/stable/${CUDA_WHEEL}/"

# shellcheck disable=SC2015 # echo does not fail, so command -v runs only without the venv
PY=$([ -x /usr/src/paperless/.venv/bin/python ] && echo /usr/src/paperless/.venv/bin/python || command -v python3)

WORKER_DIR=/opt/paperless-paddleocr
WORKER_PY="${WORKER_DIR}/venv/bin/python"
# uv would install the interpreter under /root, which the paperless user
# that runs the Celery workers cannot traverse.
export UV_PYTHON_INSTALL_DIR="${WORKER_DIR}/python"

# Whether the worker venv holds the distribution $1. Both paddle wheels
# install the module paddle, so only the distribution metadata tells them
# apart.
worker_has() {
    [ -x "${WORKER_PY}" ] && "${WORKER_PY}" -c \
        'import importlib.metadata, sys; importlib.metadata.distribution(sys.argv[1])' \
        "$1" >/dev/null 2>&1
}

UV=""
ensure_uv() {
    if [ -n "${UV}" ]; then
        return
    fi
    if command -v uv >/dev/null 2>&1; then
        UV=uv
    else
        echo "=== Installing uv into ${PY} ==="
        "${PY}" -m pip install --no-cache-dir uv
        UV="$("${PY}" -c 'import uv; print(uv.find_uv_bin())')"
    fi
}

# ---------------------------------------------------------------------------
# Native libraries
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
# paperless-ngx 3.1 and later images run 3.14.
# ---------------------------------------------------------------------------
if [ -x "${WORKER_PY}" ] && "${WORKER_PY}" -c "import paddleocr" >/dev/null 2>&1; then
    need_paddleocr=false
else
    need_paddleocr=true
    if [ ! -f "${REQUIREMENTS}" ]; then
        echo "ERROR: ${REQUIREMENTS} not found." >&2
        echo "       Put paddle-requirements-gpu.txt into ${ARTIFACT_DIR}/." >&2
        exit 1
    fi
fi

if [ -x "${WORKER_PY}" ]; then
    echo "Worker venv already exists - skipping its creation"
else
    ensure_uv
    echo "=== Creating the Python 3.13 worker venv ==="
    "${UV}" python install 3.13
    "${UV}" venv --python 3.13 "${WORKER_DIR}/venv"
fi

# Refuse to run if paddlepaddle (CPU) is in the worker venv - the two are
# mutually exclusive Python packages and the installer would happily
# install both, leading to silent import-order surprises.
if worker_has paddlepaddle && ! worker_has paddlepaddle-gpu; then
    echo "ERROR: paddlepaddle (CPU) is installed in the worker venv. Refusing to layer paddlepaddle-gpu on top." >&2
    echo "       Uninstall it first:  uv pip uninstall --python ${WORKER_PY} paddlepaddle" >&2
    echo "       Or use setup.sh (CPU) instead of setup-gpu.sh." >&2
    exit 1
fi

installed=false

if worker_has paddlepaddle-gpu; then
    echo "paddlepaddle-gpu already installed in the worker venv - skipping"
else
    ensure_uv
    echo "=== Installing paddlepaddle-gpu from ${PADDLE_INDEX} ==="
    "${UV}" pip install --no-cache --python "${WORKER_PY}" \
        --index-url "${PADDLE_INDEX}" "paddlepaddle-gpu>=3.0,<4"
    installed=true
fi

# Baidu's index carries no paddleocr release, so paddleocr comes from PyPI
# in an install of its own.
if [ "${need_paddleocr}" = false ]; then
    echo "Worker venv already imports paddleocr - skipping"
else
    ensure_uv
    echo "=== Installing paddleocr into the worker venv ==="
    "${UV}" pip install --no-cache --python "${WORKER_PY}" -r "${REQUIREMENTS}"
    installed=true
fi

if [ "${installed}" = true ]; then
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

echo "=== paperless-paddleocr bootstrap complete (GPU profile, ${CUDA_WHEEL}) ==="
echo "Reminder: set PAPERLESS_PADDLEOCR_ENGINE=classic-gpu in the container env."
