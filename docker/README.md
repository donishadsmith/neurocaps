# Docker

To build the Docker image, run the following command in the root directory:

```bash
# Add submodules needed to build image
git submodule update --init

# Build image
docker build -t neurocaps -f docker/Dockerfile .
```
