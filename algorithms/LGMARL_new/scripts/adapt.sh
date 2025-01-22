#!/bin/sh
n_run=7
experiment_name="Ad_12_18s50np_ec4_lg"
n_steps=10000000
lr=0.0005 # default 0.0005
lang_lr=0.007
env_name="magym_PredPrey_RGB"
FT_magym_env_size=18
FT_magym_actual_obsrange=5
FT_freeze_lang_after_n=10000000 # default None
FT_comm_eps_start=1.0 # default 1.0
model_dir="models/magym_PredPrey_RGB/12s50np_ec4_lg"
cuda_device="cuda:1"

comm_langground_pt="results/data/lamarl_data/PPrgb_18_langground.pt"

source venv3.8/bin/activate

for n in $(seq 1 $n_run)
do
    printf "Run ${n}/${n_run}\n"
    seed=$RANDOM
    comm="python algorithms/LGMARL_new/train_lgmarl_diff.py --seed ${seed}
    --experiment_name ${experiment_name}
    --env_name ${env_name}
    --model_dir ${model_dir}
    --n_steps ${n_steps}
    --lr ${lr}
    --lang_lr ${lang_lr}
    --FT_magym_env_size ${FT_magym_env_size}
    --FT_freeze_lang_after_n ${FT_freeze_lang_after_n}
    --FT_comm_eps_start ${FT_comm_eps_start}
    --cuda_device ${cuda_device}
    --adapt_run
    --comm_langground_pt ${comm_langground_pt}"
    # --FT_magym_not_see_agents"
    # --FT_freeze_lang"
    printf "Starting training with command:\n${comm}\n\nSEED IS ${seed}\n"
    eval $comm
    printf "DONE\n\n"
done