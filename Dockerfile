FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel

WORKDIR /workspace/dyco-rl
ENV PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y --no-install-recommends build-essential git \
    && rm -rf /var/lib/apt/lists/*
COPY . /workspace/dyco-rl
ARG INSTALL_FLASH_ATTN=1
RUN INSTALL_FLASH_ATTN=${INSTALL_FLASH_ATTN} bash setup.sh
CMD ["/bin/bash"]
