#!/bin/bash
#SBATCH --partition=hard
#SBATCH --job-name=hardcolor
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=1000
#SBATCH --mail-type=ALL
#SBATCH --mail-user=maxime.toquebiau@sorbonne.universite.fr
#SBATCH --output=outputs/%x-%j.out

source venv/bin/activate

n_parallel_envs=250
n_steps=10000000
ppo_epoch=15 # default 15
rollout_length=100 # default 100
n_mini_batch=1 # default 2
comm_eps_smooth=1000.0 # default 1.0
comm_token_penalty=0.001
lang_batch_size=1024 # default 256
lang_capt_loss_weight=1 # default 0.0001
lang_embed_dim=4 # default 4
lr=0.0005 # default 0.0005
hidden_dim=128 # default 64
policy_layer_N=2 # default 1
policy_recurrent_N=2 # default 1
entropy_coef=0.01 #default 0.01
lang_lr=0.007 # default 0.007
lang_hidden_dim=64
log_exp_device="scai"

n_run=1
experiment_name="18s50_lang_noclip"
episode_length=50
comm_type="lang+no_clip" # default language
context_dim=16 # default 16
cuda_device="cuda:0"
comm_langground_pt="results/data/lamarl_data/MPEHardSimpRef_l5s5_lg.pt"

env_name="magym_PredPrey_RGB"
magym_env_size=18
magym_obs_range=5 # default 5
n_agents=4
n_preys=2
magym_scaleenv_after_n=10000100


export WANDB_MODE=offline
# export WANDB_API_KEY=46dca67f37b349cf3cffb8d28591dbbb1b266fcc

for n in $(seq 1 $n_run)
do
    printf "Run ${n}/${n_run}\n"
    seed=$RANDOM
    comm="python algorithms/LAMARL/train_lgmarl_diff.py --seed ${seed}
    --experiment_name ${experiment_name}
    --n_parallel_envs ${n_parallel_envs}
    --n_steps ${n_steps}
    --hidden_dim ${hidden_dim}
    --policy_layer_N ${policy_layer_N}
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
    --comm_eps_smooth ${comm_eps_smooth}
    --comm_token_penalty ${comm_token_penalty}
    --comm_langground_pt ${comm_langground_pt}
    --context_dim ${context_dim}
    --lang_lr ${lang_lr}
    --lang_batch_size ${lang_batch_size}
    --lang_capt_loss_weight ${lang_capt_loss_weight}
    --lang_hidden_dim ${lang_hidden_dim}
    --magym_env_size ${magym_env_size}
    --magym_obs_range ${magym_obs_range}
    --n_agents ${n_agents}
    --magym_scaleenv_after_n ${magym_scaleenv_after_n}
    --n_preys ${n_preys}
    --log_exp_device ${log_exp_device}
    --dyna_weight_loss
    --magym_see_agents"
    # --save_increments"
    # --share_params"
    # --lang_imp_sample"
    # --log_comm"
    # --no_comm_head_learns_rl"
    printf "Starting training with command:\n${comm}\n\nSEED IS ${seed}\n"
    eval $comm
    printf "DONE\n\n"
done
