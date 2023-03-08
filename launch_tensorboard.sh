#!/bin/bash

# Script for launching the tensorboard server, listening on the port 8080
echo Launching Tensorboard with command \"python venv/lib/python3.6/site-packages/tensorboard/main.py --logdir=models/foraging/ --port=8060 --bind_all\"

source venv/bin/activate

python venv/lib/python3.6/site-packages/tensorboard/main.py --logdir=models/foraging/ --port=8060 --bind_all

