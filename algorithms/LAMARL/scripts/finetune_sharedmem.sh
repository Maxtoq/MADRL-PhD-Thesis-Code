#!/bin/sh
n_run=1
experiment_name="FT_SM_NREM"
n_parallel_envs=128
n_steps=10000000
policy_algo="mappo"
ppo_epoch=15 # default 15
entropy_coef=0.01 #default 0.01
env_name="magym_PredPrey"
episode_length=100
context_dim=16 # default 16
comm_policy_type="context_mappo"
comm_policy_algo="rmappo" # default mappo
comm_lr=0.00005 # default 0.0005
comm_entropy_coef=0.01 # default 0.01
# comm_gamma=1.0 # default 0.99
comm_ppo_epochs=16 # default 16
comm_n_warmup_steps=100000 # default 100000
comm_num_mini_batch=8 # default 2
# comm_train_topk=3 # default 1, TODO
# comm_klpretrain_coef=0.05 # default 0.01
comm_token_penalty=0.2 # default 0.1
comm_env_reward_coef=1.0 # default 1.0
comm_obs_dist_coef=0.0 # default 0.1
comm_shared_mem_coef=20.0 # default 1.0
comm_shared_mem_reward_type="shaping" # default "direct"
FT_pretrained_model_path="models/magym_PredPrey/PT_SM_EMBED/run2/model_ep.pt"
FT_n_steps_fix_policy=11000000
# lang_lr=0.0009 # default 0.0007
# lang_n_epochs=1 # default 2
# lang_batch_size=128 # default 128
magym_env_size=8
cuda_device="cuda:3"

source venv3.8/bin/activate

for n in $(seq 1 $n_run)
do
    printf "Run ${n}/${n_run}\n"
    seed=$RANDOM
    comm="python algorithms/LAMARL/finetune_sharedmem.py --seed ${seed}\
    --experiment_name ${experiment_name}\
    --n_parallel_envs ${n_parallel_envs}\
    --n_steps ${n_steps}\
    --policy_algo ${policy_algo}\
    --ppo_epoch ${ppo_epoch}\
    --entropy_coef ${entropy_coef}\
    --env_name ${env_name}\
    --episode_length ${episode_length}\
    --context_dim ${context_dim}\
    --cuda_device ${cuda_device}\
    --comm_policy_type ${comm_policy_type}\
    --comm_policy_algo ${comm_policy_algo}\
    --comm_lr ${comm_lr}\
    --comm_entropy_coef ${comm_entropy_coef}\
    --comm_ppo_epochs ${comm_ppo_epochs}\
    --comm_n_warmup_steps ${comm_n_warmup_steps}\
    --comm_num_mini_batch ${comm_num_mini_batch}\
    --comm_token_penalty ${comm_token_penalty}\
    --comm_env_reward_coef ${comm_env_reward_coef}\
    --comm_obs_dist_coef ${comm_obs_dist_coef}\
    --comm_shared_mem_coef ${comm_shared_mem_coef}\
    --comm_shared_mem_reward_type ${comm_shared_mem_reward_type}\
    --comm_noreward_empty_mess\
    --FT_pretrained_model_path ${FT_pretrained_model_path}\
    --FT_n_steps_fix_policy ${FT_n_steps_fix_policy}\
    --magym_env_size ${magym_env_size}\
    --magym_global_state\
    --log_communication"
    printf "Starting training with command:\n${comm}\n\nSEED IS ${seed}\n"
    eval $comm
    printf "DONE\n\n"
done