#!/bin/bash

# Off-screen rendering
Xvfb :0 -screen 0 1600x1200x24 & export DISPLAY=:0

# Sleep
sleep 1

if [ "$1" = "notebook" ]; then
    jupyter notebook --allow-root --no-browser --ip=0.0.0.0 --port=9999
else
    exec bash
fi
