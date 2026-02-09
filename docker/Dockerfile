FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && apt-get install -y \
    sudo \
    python3.12 \
    python3.12-venv \
    git \
    libxrender1 \
    xvfb \
    tar \
    gzip \
    nano \
    vim \
    curl \
    wget \
    libnss3 \
    libatk-bridge2.0-0 \
    libcups2 \
    libxcomposite1 \
    libxdamage1 \
    libxfixes3 \
    libxrandr2 \
    libgbm1 \
    libxkbcommon0 \
    libpango-1.0-0 \
    libcairo2 \
    libasound2 && \
    rm -rf /var/lib/apt/lists/* && \
    ln -sf /usr/bin/python3.12 /usr/bin/python && \
    ln -sf /usr/bin/python3.12 /usr/bin/python3 && \
    curl https://bootstrap.pypa.io/get-pip.py | python3.12

# Create user and home directory;
# recursively change ownership of home directory to user;
# add user to sudo group
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

RUN chown -R user:user /home/user/ && \
    chmod -R 775 /home/user/ && \
    chown user /usr/local/bin/entrypoint.sh && \
    sed -i "s/\r$//" /usr/local/bin/entrypoint.sh && \
    chmod +x /usr/local/bin/entrypoint.sh

RUN pip install --upgrade pip setuptools && \
    pip install -e src[demo] notebook && \
    rm -rf ~/.cache/pip && sync

RUN yes | plotly_get_chrome

EXPOSE 9999
LABEL maintainer="Donisha Smith <donishasmith@outlook.com>"

RUN mkdir -p /tmp/.X11-unix && chmod 1777 /tmp/.X11-unix

USER user

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]

# Will use exec bash by default if
# "docker run -it -p 9999:9999 neurocaps jupyter notebook" not used
CMD ["bash"]
