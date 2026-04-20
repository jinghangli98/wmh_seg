FROM ubuntu:24.04

# Set non-interactive mode for apt-get
ENV DEBIAN_FRONTEND=noninteractive

# Install dependencies
RUN apt update && \
    apt install -y --no-install-recommends \
    git \
    python3 \
    python3-pip \
    wget

# Clone repository and download model file
RUN git clone https://github.com/jinghangli98/wmh_seg.git /wmh_seg && \
    cd /wmh_seg && \
    wget https://huggingface.co/jil202/wmh_seg/resolve/main/ChallengeMatched_Unet_mit_b5.pth

# Install Miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /miniconda.sh && \
    bash /miniconda.sh -b -p /opt/conda && \
    rm /miniconda.sh

# Set up Conda environment variables
ENV PATH="/opt/conda/bin:$PATH"

# Initialize Conda (this modifies .bashrc for future interactive sessions)
RUN conda init bash

# Change shell to ensure Conda is available in future commands
SHELL ["/bin/bash", "-c"]

COPY wmh.yml /wmh_seg/

# Create Conda environment
RUN cd /wmh_seg && \
    conda env create -f wmh.yml -n wmh

# Set environment variables for convenience
ENV wmh_seg_home=/wmh_seg
ENV PATH="$wmh_seg_home:$PATH"

# Set Conda environment to be activated in every new shell
RUN echo "conda activate wmh" >> ~/.bashrc

# Use conda run to ensure correct environment
ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "wmh"]

# Default command
CMD ["python", "/wmh_seg/wmh_seg"]