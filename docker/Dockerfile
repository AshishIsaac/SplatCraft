# SplatCraft (Linux). Build from the repository root:
#   docker build -f docker/Dockerfile -t splatcraft .                      # CUDA build of PyTorch (default)
#   docker build -f docker/Dockerfile --build-arg TORCH=cpu -t splatcraft .
# Run (the workspace volume keeps datasets, jobs and published scenes between containers):
#   docker run --gpus all -p 8770:8770 -v splatcraft-ws:/workspace splatcraft
# The PyTorch CUDA wheels bundle CUDA + cuDNN (and Triton on Linux): the host only needs the NVIDIA
# driver and the NVIDIA Container Toolkit. The demo scenes in web/scenes are served without any setup.
FROM python:3.12-slim

ARG TORCH=gpu
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 SPLATCRAFT_HOME=/workspace
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements ./requirements
RUN pip install -r requirements/torch-${TORCH}.txt
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
COPY configs ./configs
COPY web ./web
COPY benchmarks ./benchmarks
COPY docker/entrypoint.sh /entrypoint.sh
RUN pip install -e . && chmod +x /entrypoint.sh \
    && useradd -m -u 1000 sc && mkdir -p /workspace && chown sc /workspace
USER sc
VOLUME /workspace
EXPOSE 8770
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8770/api/health')"
ENTRYPOINT ["/entrypoint.sh"]
CMD ["serve", "--host", "0.0.0.0", "--no-browser"]
