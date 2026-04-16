#!/bin/bash

Xvfb :0 -screen 0 1600x1200x24 & export DISPLAY=:0

if [[ "$1" == "notebook" || "$1" == "jupyter" || "$1" == "jupyter-notebook" ]]; then
    exec jupyter notebook --allow-root --no-browser --ip=0.0.0.0 --port=9999
else
    exec "$@"
fi
