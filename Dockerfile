FROM nvidia/cuda:12.8.2-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    libgl1 \
    libglib2.0-0 \
    libgl1-mesa-glx \
    python3.10 \
    python3.10-dev \
    python3-pip \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.10 /usr/bin/python && \
    ln -sf /usr/bin/python3.10 /usr/bin/python3 && \
    ln -sf /usr/bin/pip3 /usr/bin/pip

RUN pip install \
 torch==2.8.0 \
 torchvision==0.23.0 \
 trimesh \
 numba \
 scikit-image \ 
 scikit-learn \
 tqdm \
 plyfile \
 zstandard \
 easydict \
 ninja \
 opencv-python \
 --extra-index-url https://download.pytorch.org/whl/cu128

RUN pip install https://github.com/eugene-kamenev/ply2glb/releases/download/1.0.0/cumesh-0.0.0-cp310-cp310-linux_x86_64.whl \
 https://github.com/eugene-kamenev/ply2glb/releases/download/1.0.0/flex_gemm-0.0.0-cp310-cp310-linux_x86_64.whl \
 https://github.com/eugene-kamenev/ply2glb/releases/download/1.0.0/nvdiffrast-0.4.0-cp310-cp310-linux_x86_64.whl \
 https://github.com/eugene-kamenev/ply2glb/releases/download/1.0.0/o_voxel-0.0.0-cp310-cp310-linux_x86_64.whl

WORKDIR /app
COPY ./ply2glb.py /app/ply2glb.py
