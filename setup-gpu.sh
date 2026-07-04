#!/bin/bash
# Bootstrap paperless-paddleocr inside a paperless-ngx Docker container
# with the GPU paddlepaddle wheel.
#
# Mount this script (plus the matching paperless-paddleocr wheel/sdist)
# under /custom-cont-init.d/ - the paperless-ngx base image runs every
# executable in that directory before starting paperless itself.
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
# A pre-built artifact is REQUIRED next to this script (same shapes as
# setup.sh):
#   * paperless-paddleocr.tar.gz   (sdist)
#   * paperless_paddleocr-*.whl    (wheel - any version, first match wins)
# Obtain one via examples/extract-wheel/README.md.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARBALL="${SCRIPT_DIR}/paperless-paddleocr.tar.gz"
shopt -s nullglob
WHEELS=("${SCRIPT_DIR}"/paperless_paddleocr-*.whl)
shopt -u nullglob

CUDA_WHEEL="${PADDLE_CUDA_WHEEL:-cu126}"
PADDLE_INDEX="https://www.paddlepaddle.org.cn/packages/stable/${CUDA_WHEEL}/"

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

need_apt=false
for pkg in "${NATIVE_PKGS[@]}"; do
    if ! dpkg -l "${pkg}" 2>/dev/null | grep -q "^ii"; then
        need_apt=true
        break
    fi
done

if [ "${need_apt}" = true ]; then
    echo "=== Installing native dependencies ==="
    apt-get update
    apt-get install -y --no-install-recommends "${NATIVE_PKGS[@]}"
    rm -rf /var/lib/apt/lists/*
else
    echo "Native dependencies already present - skipping apt"
fi

# ---------------------------------------------------------------------------
# GPU paddlepaddle. Skipped if paddlepaddle-gpu is already present.
# Refuse to run if paddlepaddle (CPU) is installed - the two are mutually
# exclusive Python packages and pip would happily install both, leading to
# silent import-order surprises.
# ---------------------------------------------------------------------------
if pip show paddlepaddle >/dev/null 2>&1 && ! pip show paddlepaddle-gpu >/dev/null 2>&1; then
    echo "ERROR: paddlepaddle (CPU) is installed; refusing to layer paddlepaddle-gpu on top." >&2
    echo "       Uninstall it first:  pip uninstall -y paddlepaddle" >&2
    echo "       Or use setup.sh (CPU) instead of setup-gpu.sh." >&2
    exit 1
fi

if pip show paddlepaddle-gpu >/dev/null 2>&1; then
    echo "paddlepaddle-gpu already installed - skipping"
else
    echo "=== Installing paddlepaddle-gpu from ${PADDLE_INDEX} ==="
    pip install --no-cache-dir --index-url "${PADDLE_INDEX}" paddlepaddle-gpu
fi

# ---------------------------------------------------------------------------
# Plugin
# ---------------------------------------------------------------------------
PACKAGE_NAME="paperless-paddleocr"

if pip show "${PACKAGE_NAME}" >/dev/null 2>&1; then
    echo "${PACKAGE_NAME} already installed - skipping pip install"
else
    if [ -f "${TARBALL}" ]; then
        echo "=== Installing ${PACKAGE_NAME} from ${TARBALL} ==="
        pip install --no-cache-dir "${TARBALL}"
    elif [ ${#WHEELS[@]} -gt 0 ]; then
        echo "=== Installing ${PACKAGE_NAME} from ${WHEELS[0]} ==="
        pip install --no-cache-dir "${WHEELS[0]}"
    else
        echo "ERROR: no paperless-paddleocr artifact found next to setup-gpu.sh." >&2
        echo "       Expected one of:" >&2
        echo "         ${TARBALL}" >&2
        echo "         ${SCRIPT_DIR}/paperless_paddleocr-*.whl" >&2
        echo "       See examples/extract-wheel/README.md for how to build one." >&2
        exit 1
    fi
fi

echo "=== paperless-paddleocr bootstrap complete (GPU profile, ${CUDA_WHEEL}) ==="
echo "Reminder: set PAPERLESS_PADDLEOCR_ENGINE=classic-gpu in the container env."
