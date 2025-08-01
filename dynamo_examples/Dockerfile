FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

# Install system packages
RUN apt-get update && apt-get install -y \
    git wget curl build-essential \
    python3 python3-pip python3-dev python-is-python3 \
    libucx0 \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --upgrade pip

# Install Python dependencies
RUN pip install packaging

# Install PyTorch with CUDA 12.1
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install flash-attn and triton
RUN pip install flash-attn triton

# Clone and install Dynamo
WORKDIR /opt
RUN git clone https://github.com/ai-dynamo/dynamo.git
WORKDIR /opt/dynamo
RUN pip install -e .
RUN pip install nixl

# Set working directory for user code
WORKDIR /app

# Add entrypoint wrapper script
RUN echo '#!/bin/bash\n' \
         'echo "==> Listing /app contents:"\n' \
         'ls -alh /app\n' \
         'echo ""\n' \
         'if [[ ! -x "/app/start.sh" ]]; then\n' \
         '  echo "❌ ERROR: start.sh not found or not executable in /app!" >&2\n' \
         '  exit 1\n' \
         'fi\n' \
         'echo "🚀 Launching /app/start.sh..."\n' \
         '/app/start.sh' > /entrypoint.sh && chmod +x /entrypoint.sh

#ENTRYPOINT ["/entrypoint.sh"]
CMD ["/bin/bash"]
