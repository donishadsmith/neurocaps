# Use Ubuntu version 22.04 as base image
FROM ubuntu:22.04

# No installation prompts
ENV DEBIAN_FRONTEND=noninteractive

# Set default permissions for new files in bashrc
RUN echo "umask 000" >> /root/.bashrc

# Install Python, Pip, Git & additional system libraries for headless VTK and file uncompressing; Modify folder permissions recursively
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    git \
    libxrender1 \
    libglib2.0-0 \
    libosmesa6-dev \
    libglu1-mesa-dev \
    libxt-dev \
    xvfb \
    tar \
    gzip \
    bzip2 \
    curl \
    wget && \
    chmod -R 777 /root /mnt

# Working directory
WORKDIR /mnt

# Copy files
COPY neurocaps/ neurocaps/
COPY demos/* demos/
COPY docker/scripts/entry.sh entrypoint/
COPY pyproject.toml .

# Install package in editable mode and install notebook
RUN pip install --upgrade pip setuptools wheel
RUN pip install -e .[demo] notebook

# Make entry script executable
RUN chmod +x /mnt/entrypoint/entry.sh

# Document port jupyter notebook will use
EXPOSE 9999

# Shell script as entry point
ENTRYPOINT ["/mnt/entrypoint/entry.sh"]

# Will use exec bash by default if "docker run -it -p 9999:9999 neurocaps jupyter notebook" not used
