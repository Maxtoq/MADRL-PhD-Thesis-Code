#!/bin/bash

# Script for launching the tensorboard server, listening on the port 8080
echo Launching Tensorboard with command \"python venv/lib/python3.8/site-packages/tensorboard/main.py --logdir=models/click_n_push/ --port=8080 --bind_all\"

source venv/bin/activate

python venv/lib/python3.8/site-packages/tensorboard/main.py --logdir=models/click_n_push/ --port=8050 --bind_all

