# Use Ubuntu version 22.04 as base image
FROM ubuntu:22.04

# No installation prompts
ENV DEBIAN_FRONTEND=noninteractive

# Get repos, install Python, Pip, Git, additional system libraries for headless VTK, etc; add symbolic links
RUN apt-get update && apt-get install -y \
    software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && apt-get install -y \
    sudo \
    python3.11 \
    python3.11-distutils \
    python3.11-venv \
    git \
    libxrender1 \
    xvfb \
    tar \
    gzip \
    nano \
    vim \
    curl \
    wget && \
    rm -rf /var/lib/apt/lists/* && \
    ln -sf /usr/bin/python3.11 /usr/bin/python && \
    ln -sf /usr/bin/python3.11 /usr/bin/python3 && \
    curl https://bootstrap.pypa.io/get-pip.py | python3.11

# Create user and home directory; recursively change ownership of home directory to user; add user to sudo group
# Append no password requirement for sudo users to sudoers
RUN useradd -md /home/user user && \
    chown -R user /home/user && \
    adduser user sudo && \
    echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers

WORKDIR /home/user
ENV HOME="/home/user"

COPY neurocaps/ src/neurocaps/
COPY demos/* demos/
COPY docker/scripts/entrypoint.sh /usr/local/bin
COPY pyproject.toml src/
COPY tests/data/nilearn_data /home/user/nilearn_data
COPY tests/data/neuromaps-data /home/user/neuromaps-data

# Change ownership and permissions for recently copied files which uses root
RUN chown -R user:user /home/user/ && \
    chmod -R 775 /home/user/ && \
    chown user /usr/local/bin/entrypoint.sh

# Install package in editable mode and install notebook
RUN pip install --upgrade pip setuptools wheel && \
    pip install -e src[demo] notebook && \
    rm -rf ~/.cache/pip && sync

# Metadata
EXPOSE 9999
LABEL maintainer="Donisha Smith <donishasmith@outlook.com>"

# Modify permissions for X11
RUN mkdir -p /tmp/.X11-unix && chmod 1777 /tmp/.X11-unix

# Run as user when running container
USER user

# Shell script as entry point
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]

# Will use exec bash by default if "docker run -it -p 9999:9999 neurocaps jupyter notebook" not used
CMD ["bash"]
