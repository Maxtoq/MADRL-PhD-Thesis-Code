#!/bin/sh
n_run=5
experiment_name="pretrain_sharedmem_8x8"
n_parallel_envs=64
n_steps=5000000
policy_algo="mappo"
ppo_epoch=15 # default 15
entropy_coef=0.01 #default 0.01
env_name="magym_PredPrey"
episode_length=100
comm_policy_algo="perfect_comm"
lang_lr=0.0009 # default 0.0007
lang_n_epochs=1 # default 2
lang_batch_size=128 # default 128
shared_mem_lr=0.0001 # default 0.0005
magym_env_size=8
cuda_device="cuda:1"

source venv3.8/bin/activate

for n in $(seq 1 $n_run)
do
    printf "Run ${n}/${n_run}\n"
    seed=$RANDOM
    comm="python algorithms/LAMARL/pretrain_lmc_sharedmem.py --seed ${seed}\
    --magym_global_state\
    --experiment_name ${experiment_name}\
    --n_parallel_envs ${n_parallel_envs}\
    --n_steps ${n_steps}\
    --policy_algo ${policy_algo}\
    --ppo_epoch ${ppo_epoch}\
    --entropy_coef ${entropy_coef}\
    --env_name ${env_name}\
    --episode_length ${episode_length}\
    --cuda_device ${cuda_device}\
    --comm_policy_algo ${comm_policy_algo}\
    --lang_lr ${lang_lr}\
    --lang_n_epochs ${lang_n_epochs}\
    --lang_batch_size ${lang_batch_size}\
    --shared_mem_lr ${shared_mem_lr}\
    --magym_env_size ${magym_env_size}"
    printf "Starting training with command:\n${comm}\n\nSEED IS ${seed}\n"
    eval $comm
    printf "DONE\n\n"
done