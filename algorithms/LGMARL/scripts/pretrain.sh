#!/bin/sh
n_run=3
experiment_name="9o5_ACC_no"
n_parallel_envs=250
n_steps=10000000
hidden_dim=64 # default 64
policy_recurrent_N=1 # default 1
ppo_epoch=15 # default 15
lr=0.0005 # default 0.0005
rollout_length=100 # default 100
n_mini_batch=1 # default 2
entropy_coef=0.01 #default 0.01
env_name="magym_PredPrey"
episode_length=100
comm_type="no_comm" # default language
comm_ec_strategy="mean" # default sum
comm_eps_smooth=2.0 # default 1.0
comm_token_penalty=0.001
context_dim=16 # default 16
lang_lr=0.0009 # default 0.007
lang_batch_size=256 # default 256
lang_capt_loss_weight=1 # default 0.0001
lang_capt_loss_weight_anneal=1 # default 0.0001
magym_env_size=9
magym_obs_range=5 # default 5
cuda_device="cuda:0"

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
    --rollout_length ${rollout_length}\
    --n_mini_batch ${n_mini_batch}\
    --entropy_coef ${entropy_coef}\
    --env_name ${env_name}\
    --episode_length ${episode_length}\
    --cuda_device ${cuda_device}\
    --comm_type ${comm_type}\
    --comm_ec_strategy ${comm_ec_strategy}\
    --comm_eps_smooth ${comm_eps_smooth}\
    --comm_token_penalty ${comm_token_penalty}\
    --context_dim ${context_dim}\
    --lang_lr ${lang_lr}\
    --lang_batch_size ${lang_batch_size}\
    --lang_capt_loss_weight ${lang_capt_loss_weight}\
    --lang_capt_loss_weight_anneal ${lang_capt_loss_weight_anneal}\
    --magym_env_size ${magym_env_size}\
    --magym_obs_range ${magym_obs_range}"
    # --share_params\
    # --no_comm_head_learns_rl"
    printf "Starting training with command:\n${comm}\n\nSEED IS ${seed}\n"
    eval $comm
    printf "DONE\n\n"
done