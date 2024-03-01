#!/bin/sh
n_run=7
experiment_name="ACC_9x9_no_CRITICOBSENC_mb128"
n_parallel_envs=250
n_steps=10000000
hidden_dim=64 # default 64
policy_recurrent_N=1 # default 1
ppo_epoch=15 # default 15
lr=0.0005 # default 0.0005
n_mini_batch=1 # default 2
entropy_coef=0.01 #default 0.01
env_name="magym_PredPrey"
episode_length=100
comm_type="no_comm" # default language
comm_ec_strategy="mean" # default sum
comm_token_penalty=0.001
context_dim=16 # default 16
lang_clip_lr=0.0009 # default 0.007
lang_clip_batch_size=128 # default 256
lang_capt_lr=0.001 # default 0.007
lang_capt_lr_anneal_to=0.0001 # default 0.0001
lang_capt_n_epochs=2 # default 2
lang_capt_batch_size=10 # default 100
magym_env_size=9
cuda_device="cuda:3"

source venv3.8/bin/activate

for n in $(seq 1 $n_run)
do
    printf "Run ${n}/${n_run}\n"
    seed=$RANDOM
    comm="python algorithms/LGMARL/pretrain_lgmarl.py --seed ${seed}\
    --experiment_name ${experiment_name}\
    --n_parallel_envs ${n_parallel_envs}\
    --n_steps ${n_steps}\
    --hidden_dim ${hidden_dim}\
    --policy_recurrent_N ${policy_recurrent_N}\
    --ppo_epoch ${ppo_epoch}\
    --lr ${lr}\
    --n_mini_batch ${n_mini_batch}\
    --entropy_coef ${entropy_coef}\
    --env_name ${env_name}\
    --episode_length ${episode_length}\
    --cuda_device ${cuda_device}\
    --comm_type ${comm_type}\
    --comm_ec_strategy ${comm_ec_strategy}\
    --comm_token_penalty ${comm_token_penalty}\
    --context_dim ${context_dim}\
    --lang_clip_lr ${lang_clip_lr}\
    --lang_clip_batch_size ${lang_clip_batch_size}\
    --lang_capt_lr ${lang_capt_lr}\
    --lang_capt_lr_anneal_to ${lang_capt_lr_anneal_to}\
    --lang_capt_n_epochs ${lang_capt_n_epochs}\
    --lang_capt_batch_size ${lang_capt_batch_size}\
    --magym_env_size ${magym_env_size}"
    # --share_params\
    # --no_comm_head_learns_rl"
    printf "Starting training with command:\n${comm}\n\nSEED IS ${seed}\n"
    eval $comm
    printf "DONE\n\n"
done