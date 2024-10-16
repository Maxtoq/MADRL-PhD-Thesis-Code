#!/bin/bash
#SBATCH --partition=gpu_p2
#SBATCH --job-name=100k
#SBATCH --nodes=1
#SBATCH --gres=gpu:1 
#SBATCH --time=20:00:00
#SBATCH --output=outputs/%x-%j.out
#SBATCH -A bqo@v100

source venv/bin/activate

n_run=2
experiment_name="Ad_9o5SA_15o5_langsup_frzlCE100k"
n_steps=10000000
lr=0.0005 # default 0.0005
FT_env_name="magym_PredPrey_new"
FT_magym_env_size=15
FT_magym_actual_obsrange=5
FT_freeze_lang_after_n=100000 # default None
FT_comm_eps_start=1.0 # default 1.0
model_dir="models/magym_PredPrey_new/9o5SA_Diff_langsup/run4/"
cuda_device="cuda:0"

for n in $(seq 1 $n_run)
do
    printf "Run ${n}/${n_run}\n"
    seed=$RANDOM
    comm="python algorithms/LGMARL_new/train_lgmarl_diff.py --seed ${seed}
    --experiment_name ${experiment_name}
    --FT_env_name ${FT_env_name}
    --model_dir ${model_dir}
    --n_steps ${n_steps}
    --lr ${lr}
    --FT_magym_env_size ${FT_magym_env_size}
    --FT_freeze_lang_after_n ${FT_freeze_lang_after_n}
    --FT_comm_eps_start ${FT_comm_eps_start}
    --cuda_device ${cuda_device}
    --adapt_run"
    #--FT_magym_not_see_agents"
    # --FT_freeze_lang"
    printf "Starting training with command:\n${comm}\n\nSEED IS ${seed}\n"
    eval $comm
    printf "DONE\n\n"
done
