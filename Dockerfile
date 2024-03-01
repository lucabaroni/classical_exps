# Use the base image with specified version details
ARG BASE_IMAGE=sinzlab/pytorch:v3.9-torch1.9.0-cuda11.1-dj0.12.7
FROM ${BASE_IMAGE}


# Set arguments for GitHub credentials
ARG GITHUB_USER
ARG GITHUB_TOKEN

WORKDIR /src

# Configure git to use credentials for pulling repositories and clone them
RUN git config --global credential.helper store && \
    echo https://${GITHUB_USER}:${GITHUB_TOKEN}@github.com >> ~/.git-credentials && \
    git clone -b model_builder https://github.com/sinzlab/nnvision.git && \
    git clone https://github.com/CSNG-MFF/imagen.git

RUN git clone https://${GITHUB_USER}:${GITHUB_TOKEN}@github.com/lucabaroni/featurevis.git

RUN python3.9 -m pip --no-cache-dir install torch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1
RUN python3.9 -m pip --no-cache-dir install transformers==4.35.2

# Upgrade pip and install the required Python packages
RUN python3.9 -m pip install --upgrade pip && \
    python3.9 -m pip install -e /src/nnvision && \
    python3.9 -m pip install -e /src/featurevis && \
    python3.9 -m pip install -e /src/imagen && \
    python3.9 -m pip install "deeplake[enterprise]" && \
    python3.9 -m pip --no-cache-dir install \
        wandb \
        moviepy \
        imageio \
        tqdm \
        statsmodels  && \
    python3.9 -m pip install param==1.5.1  
    # python3.9 -m pip install git+https://github.com/sinzlab/insilico-stimuli.git

# lines below are necessasry to fix an issue explained here: https://github.com/NVIDIA/nvidia-docker/issues/1631
# RUN rm /etc/apt/sources.list.d/cuda.list
# RUN rm /etc/apt/sources.list.d/nvidia-ml.list
# RUN apt-key del 7fa2af80
# RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub
# RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub

# Install additional utilities
RUN apt-get -y update 
RUN apt-get install -y screen

# Add the project directory and install its dependencies
ADD . /project
RUN python3.9 -m pip install -e /project

# Set the working directory
WORKDIR /notebooks
