#!/bin/bash
#SBATCH --partition=hard
#SBATCH --nodelist=lizzy
#SBATCH --job-name=perf_comm
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=3000
#SBATCH --mail-type=ALL
#SBATCH --mail-user=maxime.toquebiau@sorbonne.universite.fr
#SBATCH --output=outputs/%x-%j.out

source venv/bin/activate

n_run=4
experiment_name="TAT_ACC_9x9_pt_perfect_comm"
n_parallel_envs=200
n_steps=10000000
ppo_epoch=15 # default 15
n_mini_batch=1 # default 2
entropy_coef=0.01 #default 0.01
env_name="magym_PredPrey"
episode_length=100
comm_type="perfect_comm" # default language
comm_ec_strategy="mean" # default sum
context_dim=16 # default 16
lang_lr=0.0009 # default 0.0007
lang_clip_n_epochs=1 # default 2
lang_clip_batch_size=128 # default 128
magym_env_size=9
magym_obs_range=5 # default 5
cuda_device="cuda:0"

for n in $(seq 1 $n_run)
do
    printf "Run ${n}/${n_run}\n"
    seed=$RANDOM
    comm="python algorithms/LGMARL/pretrain_lgmarl.py --seed ${seed}\
    --experiment_name ${experiment_name}\
    --n_parallel_envs ${n_parallel_envs}\
    --n_steps ${n_steps}\
    --ppo_epoch ${ppo_epoch}\
    --n_mini_batch ${n_mini_batch}\
    --entropy_coef ${entropy_coef}\
    --env_name ${env_name}\
    --episode_length ${episode_length}\
    --cuda_device ${cuda_device}\
    --comm_type ${comm_type}\
    --comm_ec_strategy ${comm_ec_strategy}\
    --context_dim ${context_dim}\
    --lang_lr ${lang_lr}\
    --lang_clip_n_epochs ${lang_clip_n_epochs}\
    --lang_clip_batch_size ${lang_clip_batch_size}\
    --magym_env_size ${magym_env_size}\
    --magym_obs_range ${magym_obs_range}"
    # --enc_obs"
    #--comm_head_learns_rl"
    printf "Starting training with command:\n${comm}\n\nSEED IS ${seed}\n"
    eval $comm
    printf "DONE\n\n"
done