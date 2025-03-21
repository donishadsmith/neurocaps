#!/bin/bash

# Off-screen rendering and export for all processes to access the headless display
Xvfb :0 -screen 0 1600x1200x24 & export DISPLAY=:0

# Variable wait time to ensure Xvfb start; send standard output and error to the void
while !xset q &> /dev/null; do
    sleep 0.1
done

# If first arg after script is "notebook" then run jupyter else bash shell
if [ "$1" = "notebook" ]; then
    jupyter notebook --allow-root --no-browser --ip=0.0.0.0 --port=9999
else
    exec "$@"
fi
