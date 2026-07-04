#!/bin/bash
# Bootstrap paperless-paddleocr inside a paperless-ngx Docker container.
#
# Mount this script (plus the matching paperless-paddleocr wheel/sdist)
# under /custom-cont-init.d/ - the paperless-ngx base image runs every
# executable in that directory before starting paperless itself.
#
# This script installs the CPU build of PaddlePaddle. For local GPU
# acceleration use setup-gpu.sh; for remote VL recognition use the
# `vl-remote` engine pointed at a paddleocr-genai-vllm-server sidecar
# (see examples/docker-compose.vl-remote.yml).
#
# A pre-built artifact is REQUIRED next to this script - paperless-paddleocr
# is not published to PyPI. The script supports both shapes:
#   * paperless-paddleocr.tar.gz   (sdist)
#   * paperless_paddleocr-*.whl    (wheel - any version, first match wins)
#
# Obtain one with either of:
#   pip wheel --no-deps "git+https://github.com/flobernd/paperless-paddleocr.git@v0.1.0"
#   # or extract from the docker/builder.Dockerfile artifact -
#   # see examples/extract-wheel/README.md
#
# Both install paths are idempotent: repeated container restarts skip
# work that is already done.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARBALL="${SCRIPT_DIR}/paperless-paddleocr.tar.gz"
# First whl matching the pattern wins. shopt avoids a literal glob string
# in the unmatched case.
shopt -s nullglob
WHEELS=("${SCRIPT_DIR}"/paperless_paddleocr-*.whl)
shopt -u nullglob

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
# Python package
# ---------------------------------------------------------------------------
PACKAGE_NAME="paperless-paddleocr"

is_installed() {
    pip show "${PACKAGE_NAME}" >/dev/null 2>&1
}

# CPU paddlepaddle. Skipped if paddlepaddle (or paddlepaddle-gpu) is already
# present - running this CPU script on top of a baked GPU image must not
# clobber the GPU wheel.
if pip show paddlepaddle >/dev/null 2>&1 || pip show paddlepaddle-gpu >/dev/null 2>&1; then
    echo "paddlepaddle already installed - skipping"
else
    echo "=== Installing paddlepaddle (CPU) ==="
    pip install --no-cache-dir "paddlepaddle>=3.0"
fi

if is_installed; then
    echo "${PACKAGE_NAME} already installed - skipping pip install"
else
    if [ -f "${TARBALL}" ]; then
        echo "=== Installing ${PACKAGE_NAME} from ${TARBALL} ==="
        pip install --no-cache-dir "${TARBALL}"
    elif [ ${#WHEELS[@]} -gt 0 ]; then
        echo "=== Installing ${PACKAGE_NAME} from ${WHEELS[0]} ==="
        pip install --no-cache-dir "${WHEELS[0]}"
    else
        echo "ERROR: no paperless-paddleocr artifact found next to setup.sh." >&2
        echo "       Expected one of:" >&2
        echo "         ${TARBALL}" >&2
        echo "         ${SCRIPT_DIR}/paperless_paddleocr-*.whl" >&2
        echo "       See examples/extract-wheel/README.md for how to build one." >&2
        exit 1
    fi
fi

echo "=== paperless-paddleocr bootstrap complete ==="
