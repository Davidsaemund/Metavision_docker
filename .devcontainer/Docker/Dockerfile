FROM nvidia/cuda:11.7.1-cudnn8-devel-ubuntu22.04
ARG DEBIAN_FRONTEND=noninteractive

#Installing dependencies
RUN apt update
RUN apt -y install \ 
    libcanberra-gtk-module \
    mesa-utils \
    ffmpeg
RUN apt -y install \
    python3-pip
RUN apt -y install \
    python3.10-dev 
RUN python3 -m pip install pip --upgrade
RUN python3 -m pip install "opencv-python==4.5.5.64" "sk-video==1.1.10" "fire==0.4.0" "numpy==1.23.4" "h5py==3.7.0" pandas scipy
RUN python3 -m pip install matplotlib "ipywidgets==7.6.5"
RUN apt -y install \
    cmake \
    libboost-program-options-dev \
    libeigen3-dev \
    libprotobuf-dev \
    protobuf-compiler

# Installing metavision studio 
COPY metavision.list /etc/apt/sources.list.d
RUN apt update
RUN apt -y install metavision-sdk

# To use HDF5 ECF codec plugin
RUN apt install \
    hdf5-plugin-ecf \
    hdf5-plugin-ecf-dev
RUN export HDF5_PLUGIN_PATH=$HDF5_PLUGIN_PATH:/usr/lib/x86_64-linux-gnu/hdf5/serial/plugins

# Additional Dependencies for Machine Learning Module

# Python packages
RUN python3 -m pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
RUN python3 -m pip install "numba==0.56.3" "profilehooks==1.12.0" "pytorch_lightning==1.8.6" "tqdm==4.63.0" "kornia==0.6.8"
RUN python3 -m pip install "llvmlite==0.39.1" "pycocotools==2.0.7" "seaborn==0.11.2" "torchmetrics==0.7.2" "pillow==9.3.0"
#Libtorch fo c++
#COPY /home/davidsae14/Downloads/libtorch-cxx11-abi-shared-with-deps-1.13.1+cu117.zip /root
RUN apt-get install -y \
    git \
    x11vnc \
    xauth \
    net-tools \
    xvfb \
    language-pack-en-base \
    libcanberra-gtk3-module