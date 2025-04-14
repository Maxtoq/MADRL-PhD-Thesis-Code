#!/bin/bash

offline_runs="outputs/wandb/offline-run*"
for ofrun in $offline_runs
do
    wandb sync $ofrun --mark-synced --no-include-synced;
done