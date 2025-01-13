#!/bin/bash
#SBATCH --partition=gpu_p2
#SBATCH --job-name=pp_f
#SBATCH --nodes=1
#SBATCH --gres=gpu:1 
#SBATCH --time=50:00:00
#SBATCH --qos=qos_gpu-t4
#SBATCH --output=outputs/%x-%j.out
#SBATCH -A bqo@v100

source venv/bin/activate

n_run=15
experiment_name="Ad_PP9_F18_noc"
n_steps=10000000
lr=0.0009 # default 0.0005
lang_lr=0.009
FT_env_name="magym_Foraging"
FT_magym_env_size=18
FT_magym_actual_obsrange=5
FT_freeze_lang_after_n=10000000 # default None
FT_comm_eps_start=1.0 # default 1.0
model_dir="models/magym_PredPrey_new/9o5SA_Diff_noc/run29/"
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
    --lang_lr ${lang_lr}
    --FT_magym_env_size ${FT_magym_env_size}
    --FT_freeze_lang_after_n ${FT_freeze_lang_after_n}
    --FT_comm_eps_start ${FT_comm_eps_start}
    --cuda_device ${cuda_device}
    --adapt_run"
    # --FT_magym_not_see_agents"
    # --FT_freeze_lang"
    printf "Starting training with command:\n${comm}\n\nSEED IS ${seed}\n"
    eval $comm
    printf "DONE\n\n"
done
