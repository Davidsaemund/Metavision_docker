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

# Additional tools not specified in the prophesee guide
RUN apt update
RUN apt-get install -y \
    git \
    x11vnc \
    xauth \
    net-tools \
    xvfb \
    language-pack-en-base \
    libcanberra-gtk-module \
    libcanberra-gtk3-module \
    libxcb-xinerama0

RUN python3 -m pip install mlflow tensorboard pyqt5==5.14.0


# Adding Ros_2 Humble installation

RUN apt install -y software-properties-common 
RUN add-apt-repository universe 
RUN apt update && apt install curl -y
RUN curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg
RUN echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | tee /etc/apt/sources.list.d/ros2.list > /dev/null
RUN apt update
RUN apt upgrade -y
RUN apt install ros-humble-desktop -y
RUN echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc

# Extra ROS features

#Install rqt
RUN apt update
RUN apt install ~nros-humble-rqt* -y

#Install colcon
RUN apt install -y python3-colcon-common-extensions


RUN apt update && apt install -y \
    python3-flake8-docstrings \
    python3-pip \
    python3-pytest-cov \
    ros-dev-tools
  
RUN apt install -y \
    python3-flake8-blind-except \
    python3-flake8-builtins \
    python3-flake8-class-newline \
    python3-flake8-comprehensions \
    python3-flake8-deprecated \
    python3-flake8-import-order \
    python3-flake8-quotes \
    python3-pytest-repeat \
    python3-pytest-rerunfailures

