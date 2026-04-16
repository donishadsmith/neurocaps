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

ENV NILEARN_SHARED_DATA=/opt/neurocaps/data/nilearn_data \
    NEUROMAPS_DATA=/opt/neurocaps/data/neuromaps-data \
    NEUROCAPS_DATA=/opt/neurocaps/data/neurocaps_data

RUN mkdir -p /opt/neurocaps/src \
    /opt/neurocaps/demos \
    $NILEARN_SHARED_DATA \
    $NEUROMAPS_DATA \
    $NEUROCAPS_DATA

WORKDIR /opt/neurocaps

COPY neurocaps/ src/neurocaps/
COPY pyproject.toml src/
COPY demos/* demos/
COPY tests/data/nilearn_data $NILEARN_SHARED_DATA
COPY tests/data/neuromaps-data $NEUROMAPS_DATA
COPY docker/scripts/entrypoint.sh /usr/local/bin/

RUN sed -i "s/\r$//" /usr/local/bin/entrypoint.sh && \
    chmod +x /usr/local/bin/entrypoint.sh

RUN pip install --upgrade pip setuptools && \
    pip install -e src[demo] notebook && \
    rm -rf ~/.cache/pip && sync

RUN yes | plotly_get_chrome

RUN chmod -R a+rwX /opt/neurocaps

RUN mkdir -p /tmp/.X11-unix && chmod 1777 /tmp/.X11-unix

EXPOSE 9999
LABEL maintainer="Donisha Smith <donishasmith@outlook.com>"

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]

# Will use exec bash by default if
# "docker run -it -p 9999:9999 neurocaps jupyter notebook" not used
CMD ["bash"]
