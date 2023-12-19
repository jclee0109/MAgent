FROM ubuntu:18.04

# Install Python3.7
RUN apt-get update
RUN apt-get install -y software-properties-common
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt-get update
RUN apt-get install -y python3.7
RUN apt-get install -y python3-pip

# Install Anaconda3
RUN apt-get install -y wget
RUN wget -c https://repo.continuum.io/archive/Anaconda3-2023.09-0-Linux-x86_64.sh
RUN bash Anaconda3-2023.09-0-Linux-x86_64.sh
RUN rm Anaconda3-2023.09-0-Linux-x86_64.sh
RUN /bin/bash -c "source ~/anaconda3/bin/activate"

# Install conda env
RUN conda env create -f env.yaml

# Activate conda env (name: game_theory)
RUN conda activate game_theory

# Clone the repo MAgent
RUN git clone https://github.com/geek-ai/MAgent.git
WORKDIR /MAgent

# Install MAgent
RUN apt-get update
RUN apt-get install cmake libboost-system-dev libjsoncpp-dev libwebsocketpp-dev
RUN apt-get update && apt-get install build-essential
RUN rm -rf build
RUN bash build.sh
RUN /bin/bash -c "export PYTHONPATH=$(pwd)/python:$PYTHONPATH"
RUN pip3 install protobuf==3.20.1

# Install PyTorch
RUN pip3 install pytorch

