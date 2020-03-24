#!/bin/bash

export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLEW.so
source ~/mujoco_env/bin/activate
cd /home/alireza/pycharm-community-2019.3.2/bin
./pycharm.sh
