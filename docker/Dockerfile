# Use Ubuntu version 22.04 as base image
FROM ubuntu:22.04

# No installation prompts
ENV DEBIAN_FRONTEND=noninteractive

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
    nano \
    vim \
    curl \
    wget && \
    rm -rf /var/lib/apt/lists/*

# Create user, set bash as default shell
RUN useradd -ms /bin/bash user

# Working directory
WORKDIR /home/user
ENV HOME="/home/user"

# Copy files
COPY neurocaps/ src/neurocaps/
COPY demos/* demos/
COPY docker/scripts/entrypoint.sh /usr/local/bin
COPY pyproject.toml src/

# Install package in editable mode and install notebook
RUN pip install --upgrade pip setuptools wheel && \
    pip install -e src[demo] notebook && \
    rm -rf ~/.cache/pip && sync

# Change ownership and permissions
RUN chown -R user:user /home/user && chmod -R 775 /home/user

# Make entry script executable
RUN chmod +x /usr/local/bin/entrypoint.sh

# Document port jupyter notebook will use
EXPOSE 9999
# More metadata
LABEL maintainer="Donisha Smith <donishasmith@outlook.com>"

# Shell script as entry point
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]

# Will use exec bash by default if "docker run -it -p 9999:9999 neurocaps jupyter notebook" not used
