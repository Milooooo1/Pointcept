# Use the official CUDA base image with PyTorch support
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu20.04

# Set an environment variable to prevent the use of an interactive prompt
ENV DEBIAN_FRONTEND=noninteractive



# Install required dependencies 
RUN apt-get update && apt-get install -y \
    git \
    curl \
    software-properties-common \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# RUN add-apt-repository 'ppa:deadsnakes/ppa'
# RUN apt-get update
RUN apt-get install python3

# Install CA certificates
RUN apt-get update && apt-get install -y ca-certificates

# Clone your GitHub repository using a build argument for the token
ARG GITHUB_TOKEN
RUN git clone https://github.com/Milooooo1/Pointcept /workspace

# ====================================
#    PIP REQUIREMENTS INSTALLATION
# ====================================

RUN python3 -m pip install --upgrade pip

# Install PyTorch
RUN python3 -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

RUN python3 -m pip install ninja

# In order to build the cpp_wrappers for KPConvX
RUN python3 -m pip install numpy==1.23.5 

# Other dependencies
RUN python3 -m pip install easydict h5py pyyaml sharedarray tensorboard tensorboardx yapf addict einops scipy plyfile termcolor timm

RUN python3 -m pip install torch-geometric torch-scatter torch-cluster torch-sparse

# spconv (SparseUNet)
# refer https://github.com/traveller59/spconv
RUN python3 -m pip install spconv-cu118

# flash-attn (FlashAttention)
# refer https://github.com/Dao-AILab/flash-attention
RUN python3 -m pip install flash-attn

# PPT (clip)
RUN python3 -m pip install ftfy regex tqdm
RUN python3 -m pip install git+https://github.com/openai/CLIP.git

# PTv1 & PTv2 or precise eval
WORKDIR /workspace/Pointcept/libs/pointops
RUN python3 setup.py install

# Build KPConvX dependencies
WORKDIR /workspace/Pointcept/pointcept/models/kpconvx/cpp_wrappers/cpp_subsampling
RUN python3 setup.py build_ext --inplace
WORKDIR /workspace/Pointcept/pointcept/models/kpconvx/cpp_wrappers/cpp_neighbors
RUN python3 setup.py build_ext --inplace

# ====================================
# END OF PIP REQUIREMENTS INSTALLATION
# ====================================

# reset working directory
WORKDIR /workspace/Pointcept

# Expose a volume for your dataset
VOLUME ["/dataset"]

# Preprocess dataset
RUN python pointcept/datasets/preprocessing/rail3d/preprocess_rail3d.py --dataset_root /dataset/Rail3D --output_root data/rail3d --num_workers 2

# Set the entrypoint to your main script

# Train PTV3
ENTRYPOINT ["sh", "scripts/train.sh", "-p", "python3", "-g", "1", "-d", "rail3d", "-c", "semseg-pt-v3", "-n", "rail3d-ptv3-base"]

# # Train KPConvX
# ENTRYPOINT ["sh", "scripts/train.sh", "-p", "python3", "-g", "1", "-d", "rail3d", "-c", "semseg-kpconvx-base",-n, "rail3d-kpconvx-base"]

# EXAMPLE USAGE:
# docker build --build-arg GITHUB_TOKEN=<YOUR_TOKEN>-t pointcept-image .
# docker run -v /path/to/your/local/dataset:/dataset pointcept-image