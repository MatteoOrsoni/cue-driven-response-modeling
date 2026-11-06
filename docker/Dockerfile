# GPU environment used for the NumPyro models of the article (Models 1-3, hierarchical truncated DDM).
# JAX with CUDA 12.9 from pip wheels; the NVIDIA driver comes from the host (>= 575 for RTX 50xx / Blackwell).
# Versions are pinned in requirements-gpu.txt (pip freeze of the image used for the article; it also
# contains HSSM, which is installed but not used by this repository).
#
#   docker build -t eyeriders-gpu docker
#   docker run --rm --gpus all -v "$(pwd)":/work -w /work eyeriders-gpu python code/06_fit_model1.py main
FROM python:3.12-slim

ENV PIP_NO_CACHE_DIR=1 \
    XLA_PYTHON_CLIENT_PREALLOCATE=false \
    PYTHONUNBUFFERED=1 \
    MIX_X64=0

COPY requirements-gpu.txt /tmp/requirements-gpu.txt
RUN pip install --upgrade pip && \
    pip install -r /tmp/requirements-gpu.txt
