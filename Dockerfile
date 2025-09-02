# Build Args
#-----------------------------------------------------------------------------------------
ARG CUDA_VERSION=12.9.1
ARG UBUNTU_VERSION=24.04
ARG PYTHON_VERSION=3.12
# Specify CUDA architectures -> 7.5: Quadro RTX 6000 & T4, 8.0: A100, 8.6: A40, 8.9: L40S, 9.0: H100
ARG TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;8.9;9.0+PTX"
# No GPUs visible during build
ARG CUDA_VISIBLE_DEVICES=none

# Build Stage Image
#-----------------------------------------------------------------------------------------
FROM python:${PYTHON_VERSION}-slim-trixie AS builder

# Install uv by copying binaries into bin
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Set UV env vars:
# UV will compile python source files to improve container startup time
ENV UV_COMPILE_BYTECODE=1 
# Changing default link mode silences warnings when using uv cache
ENV UV_LINK_MODE=copy
# Ensure installed tools can be executed out of the box
ENV UV_TOOL_BIN_DIR=/usr/local/bin
# Change python install directory
# When using singularity to run image, we can't use root user, therefore avoid having binaries in /root/.local
ENV UV_PYTHON_INSTALL_DIR=/usr/local/share/uv/python

# Tells vllm to use precompiled cuda kernels to reduce install time
ENV VLLM_USE_PRECOMPILED=1

# Set workdir
WORKDIR /app

# Get cuda tag. If provided 12.9.1 or 12.9 converts to 129
RUN CUDA_TAG=$(echo $CUDA_VERSION | cut -d. -f1,2 | tr -d .)

# Install uv-managed python
RUN uv python install ${PYTHON_VERSION}

# Use uv to install environment
# Set pytorch index to cuda 12.9
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --locked --no-install-project --no-dev --managed-python \
    --python-platform linux \
    --index https://download.pytorch.org/whl/cu${CUDA_TAG}

# Final Image
#-----------------------------------------------------------------------------------------
FROM nvidia/cuda:${CUDA_VERSION}-cudnn-runtime-ubuntu${UBUNTU_VERSION} AS runtime

# Create workdir
WORKDIR /app

# Set UV env vars again for final image
ENV UV_TOOL_BIN_DIR=/usr/local/bin
ENV UV_PYTHON_INSTALL_DIR=/usr/local/share/uv/python

# Install uv by copying binaries into bin
COPY --from=ghcr.io/astral-sh/uv:latest --chmod=777 /uv /uvx /bin/

# Copy uv-managed python (faster than installing again)
COPY --from=builder --chmod=777 /usr/local/share/uv /usr/local/share/uv

# Copy virtual environment
COPY --from=builder --chmod=777 /app/.venv /app/.venv

# Make virtual env the default by placing at the front of path
ENV PATH="/app/.venv/bin:$PATH"

# Copy source code
COPY --chmod=777 src /app/src

# Install nano and vim
RUN apt-get update && apt-get install -y \
    nano vim \
    && rm -rf /var/lib/apt/lists/*

# Reset entrypoint. uv image invokes `uv` by default
ENTRYPOINT []

# Run entrypoint script
CMD ["bash"]


