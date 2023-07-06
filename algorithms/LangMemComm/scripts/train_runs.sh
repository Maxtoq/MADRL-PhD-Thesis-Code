#!/bin/sh
n_run=11
experiment_name="mappo_30"
n_rollout_threads=32
n_steps=5000000
env_name="rel_overgen"
ro_optim_diff_coeff=30
cuda_device="cuda:1"

for n in $(seq 1 $n_run)
do
    printf "Run ${n}/${n_run}\n"
    seed=$RANDOM
    comm="python algorithms/LangMemComm/train.py --seed ${seed} \
    --experiment_name ${experiment_name} \
    --n_rollout_threads ${n_rollout_threads} \
    --n_steps ${n_steps} \
    --env_name ${env_name} \
    --ro_optim_diff_coeff ${ro_optim_diff_coeff} \
    --cuda_device ${cuda_device}"
    printf "Starting training with command:\n${comm}\n\nSEED IS ${seed}\n"
    eval $comm
    printf "DONE\n\n"
done