#!/bin/bash
#SBATCH --partition=gpu_p2
#SBATCH --job-name=forperf
#SBATCH --nodes=1
#SBATCH --gres=gpu:1 
#SBATCH --time=100:00:00
#SBATCH --qos=qos_gpu-t4
#SBATCH --output=outputs/%x-%j.out
#SBATCH -C v100
#SBATCH -A bqo@v100

source venv/bin/activate

n_parallel_envs=250
n_steps=10000000
hidden_dim=128 # default 64
policy_recurrent_N=1 # default 1
ppo_epoch=15 # default 15
rollout_length=100 # default 100
n_mini_batch=1 # default 2
env_name="magym_Foraging"
episode_length=100
comm_ec_strategy="cat" # default sum
comm_eps_smooth=2.0 # default 1.0
comm_token_penalty=0.001
lang_batch_size=1024 # default 256
lang_capt_loss_weight=1 # default 0.0001
lang_embed_dim=4 # default 4
lang_hidden_dim=128

n_run=8
experiment_name="18o5_perf"
lr=0.0009 # default 0.0005
entropy_coef=0.01 #default 0.01
comm_type="perfect" # default language
context_dim=16 # default 16
lang_lr=0.009 # default 0.007
cuda_device="cuda:0"

magym_env_size=18
magym_obs_range=5 # default 5
magym_n_agents=4

for n in $(seq 1 $n_run)
do
    printf "Run ${n}/${n_run}\n"
    seed=$RANDOM
    comm="python algorithms/LGMARL_new/train_lgmarl_diff.py --seed ${seed}
    --experiment_name ${experiment_name}
    --n_parallel_envs ${n_parallel_envs}
    --n_steps ${n_steps}
    --hidden_dim ${hidden_dim}
    --policy_recurrent_N ${policy_recurrent_N}
    --ppo_epoch ${ppo_epoch}
    --lr ${lr}
    --rollout_length ${rollout_length}
    --n_mini_batch ${n_mini_batch}
    --entropy_coef ${entropy_coef}
    --env_name ${env_name}
    --episode_length ${episode_length}
    --cuda_device ${cuda_device}
    --comm_type ${comm_type}
    --comm_ec_strategy ${comm_ec_strategy}
    --comm_eps_smooth ${comm_eps_smooth}
    --comm_token_penalty ${comm_token_penalty}
    --context_dim ${context_dim}
    --lang_lr ${lang_lr}
    --lang_batch_size ${lang_batch_size}
    --lang_capt_loss_weight ${lang_capt_loss_weight}
    --lang_hidden_dim ${lang_hidden_dim}
    --magym_env_size ${magym_env_size}
    --magym_obs_range ${magym_obs_range}
    --magym_n_agents ${magym_n_agents}
    --dyna_weight_loss
    --magym_see_agents"
    # --share_params"
    # --lang_imp_sample"
    # --log_comm"
    # --no_comm_head_learns_rl"
    printf "Starting training with command:\n${comm}\n\nSEED IS ${seed}\n"
    eval $comm
    printf "DONE\n\n"
done
