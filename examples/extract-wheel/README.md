# Installing paperless-paddleocr without Docker

The recommended deployment paths (see the top-level [README](../../README.md))
build the plugin into a custom paperless-ngx image via the recipes in
`examples/Dockerfile.classic-cpu`, `Dockerfile.classic-gpu`, and
`Dockerfile.vl-remote`. This page covers two non-Docker flows:

* **Direct pip install** - paperless-ngx running on bare metal / a VM /
  inside an unrelated container image.
* **Wheel-file extraction** - air-gapped hosts, private mirrors, or any
  pipeline that wants a `.whl` file rather than a `pip install git+…` call.

The plugin wheel is **CUDA-agnostic** - install one paddle wheel alongside
it: `paddlepaddle` for CPU or `paddlepaddle-gpu` for GPU.

## Native libraries

PaddleOCR's OpenCV backend needs a handful of native libraries. Install
them first so neither pip flow fails at runtime with `libGL.so.1`-style
errors.

| Distro            | Command                                                                                                                              |
|-------------------|--------------------------------------------------------------------------------------------------------------------------------------|
| Debian / Ubuntu   | `sudo apt-get update && sudo apt-get install -y --no-install-recommends libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 libgomp1`    |
| Alpine            | `sudo apk add mesa-gl glib libsm libxext libxrender libgomp`                                                                         |
| Fedora / RHEL     | `sudo dnf install -y mesa-libGL glib2 libSM libXext libXrender libgomp`                                                              |
| Arch              | `sudo pacman -S mesa glib2 libsm libxext libxrender gcc-libs`                                                                        |

For GPU use, additionally install the CUDA runtime / cuDNN libraries that
match the Paddle wheel you choose (see "GPU notes" below).

## Flavour 1 - Direct pip install from Git (simplest)

`pyproject.toml` is a valid PEP 517 package, so pip can build and install
the plugin directly from a Git ref.

```bash
# 1. Install the paddle wheel you want.
pip install "paddlepaddle>=3.0"
# …or, for GPU:
pip install --index-url https://www.paddlepaddle.org.cn/packages/stable/cu126/ paddlepaddle-gpu

# 2. Install the plugin.
pip install "git+https://github.com/flobernd/paperless-paddleocr.git@v0.1.0"
```

Replace `v0.1.0` with any tag, branch, or commit SHA. Use `master` to
track the latest development snapshot.

To upgrade, just re-run the second `pip install` with a newer ref.

## Flavour 2 - Wheel-file extraction

Useful when:

* The target host has no internet access to GitHub at install time.
* You want to stage a specific wheel through a private PyPI mirror.
* You're producing a reproducible build artifact for CI/CD.

The repo ships a minimal builder Dockerfile that produces just the wheel
under `/dist`. Build it once and `docker cp` the wheel out:

```bash
# 1. Build the builder image directly from a Git ref. No local clone needed.
docker build \
    -f docker/builder.Dockerfile \
    -t paddleocr-plugin-builder \
    "https://github.com/flobernd/paperless-paddleocr.git#v0.1.0"

# 2. Create a throwaway container and copy /dist out.
docker create --name paddleocr-extract paddleocr-plugin-builder
docker cp paddleocr-extract:/dist/. ./dist/
docker rm paddleocr-extract

# 3. Install on the target host.
pip install "paddlepaddle>=3.0"
pip install ./dist/paperless_paddleocr-*.whl
```

For GPU, swap step 3's `paddlepaddle` install for the matching
`paddlepaddle-gpu` invocation.

## After install

Paperless-ngx discovers the plugin through its `paperless_ngx.parsers`
entry point. Restart the paperless process(es) and confirm in the logs:

```text
Loaded third-party parser 'Paperless-ngx PaddleOCR Parser' v0.1.0
    by Florian Bernd (entrypoint: 'paddleocr').
```

Then configure the engine via env var:

```bash
export PAPERLESS_PADDLEOCR_ENGINE=classic-cpu    # default
# or classic-gpu, or vl-remote (with PAPERLESS_PADDLEOCR_VL_SERVER_URL)
```

See the top-level [README](../../README.md#environment) for the full env
var reference.

## GPU notes

`paddlepaddle-gpu` ships from Baidu's custom index. Pick the one that
matches your host's CUDA driver:

| Index                                                                | CUDA | Hardware                                |
|----------------------------------------------------------------------|------|-----------------------------------------|
| <https://www.paddlepaddle.org.cn/packages/stable/cu126/>               | 12.6 | Ampere / Ada / Hopper (most users)      |
| <https://www.paddlepaddle.org.cn/packages/stable/cu129/>               | 12.9 | Blackwell (RTX 50-series, sm_120)       |
| <https://www.paddlepaddle.org.cn/packages/stable/cu118/>               | 11.8 | Older CUDA-11-only drivers              |

Install the matching CUDA runtime / cuDNN libraries on the host
(`libcudnn9-cuda-12`, `libnccl2`, …, or the official CUDA toolkit
installer). The major CUDA version must match the wheel.

When in doubt, prefer building [`Dockerfile.classic-gpu`](../Dockerfile.classic-gpu) -
it bundles a known-good set of CUDA libraries with the wheel.
