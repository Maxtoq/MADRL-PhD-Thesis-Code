#!/bin/sh
n_run=15
experiment_name="mappo32_JIM_50"
n_rollout_threads=32
n_steps=5000000
algorithm_name="mappo"
ppo_epoch=32
env_name="rel_overgen"
ro_optim_diff_coeff=50
ir_algo="e2s_noveld"
ir_mode="central"
cuda_device="cuda:3"

for n in $(seq 1 $n_run)
do
    printf "Run ${n}/${n_run}\n"
    seed=$RANDOM
    comm="python algorithms/LangMemComm/train.py --seed ${seed}\
    --experiment_name ${experiment_name}\
    --n_rollout_threads ${n_rollout_threads}\
    --n_steps ${n_steps}\
    --algorithm_name ${algorithm_name}\
    --ppo_epoch ${ppo_epoch}\
    --env_name ${env_name}\
    --ro_optim_diff_coeff ${ro_optim_diff_coeff}\
    --ir_algo ${ir_algo}\
    --ir_mode ${ir_mode}\
    --share_policy\
    --cuda_device ${cuda_device}"
    printf "Starting training with command:\n${comm}\n\nSEED IS ${seed}\n"
    eval $comm
    printf "DONE\n\n"
done