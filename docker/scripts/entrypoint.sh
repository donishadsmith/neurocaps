#!/bin/bash

Xvfb :0 -screen 0 1600x1200x24 & export DISPLAY=:0

while !xset q &> /dev/null; do
    sleep 0.1
done

if [ "$1" = "notebook" ]; then
    exec jupyter notebook --allow-root --no-browser --ip=0.0.0.0 --port=9999
else
    exec "$@"
fi
