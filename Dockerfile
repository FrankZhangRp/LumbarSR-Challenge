# LumbarSR Challenge - Docker Environment
# Base: NVIDIA GPU + PyTorch + Medical Imaging Libraries

FROM nvcr.io/nvidia/pytorch:24.01-py3

LABEL maintainer="zhangrp@sjtu.edu.cn"
LABEL description="LumbarSR Challenge - Super-Resolution for Lumbar CT"

# Set working directory
WORKDIR /workspace

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    vim \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --upgrade pip

# Install Python packages
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Install ANTsPy (for registration)
RUN pip install --no-cache-dir antspyx

# Create workspace directories
RUN mkdir -p /workspace/data \
    /workspace/checkpoints \
    /workspace/results \
    /workspace/outputs

# Set environment variables
ENV PYTHONPATH=/workspace:$PYTHONPATH
ENV CUDA_VISIBLE_DEVICES=0

# Default command
CMD ["/bin/bash"]
