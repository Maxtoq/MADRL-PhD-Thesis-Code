#!/bin/sh
n_run=1
experiment_name="Ad_2a9o5SA_15o5_langsup"
n_steps=5000000
lr=0.0005 # default 0.0005
FT_env_name="magym_PredPrey_new"
FT_magym_env_size=15
FT_magym_actual_obsrange=5
FT_freeze_lang_after_n=5000000 # default None
FT_comm_eps_start=1.0 # default 1.0
model_dir="models/magym_PredPrey_new/2a6-9o5SA_langsup/run1/"
cuda_device="cuda:1"

source venv3.8/bin/activate

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
    # --FT_magym_not_see_agents"
    # --FT_freeze_lang"
    printf "Starting training with command:\n${comm}\n\nSEED IS ${seed}\n"
    eval $comm
    printf "DONE\n\n"
done