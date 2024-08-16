#!/bin/sh
n_run=7
experiment_name="Adapt_obs3_ACC_lang_FrzLng"
n_steps=5000000
lr=0.0005 # default 0.0005
FT_env_name="magym_PredPrey"
FT_magym_env_size=9
FT_magym_actual_obsrange=3
# model_dir="models/magym_PredPrey/ACC_9o5_perf_alldynawl/run15"
# model_dir="models/magym_PredPrey/9o5_ACC_no/run14/"
model_dir="models/magym_PredPrey/9o5_ACC_lang/run9/"
cuda_device="cuda:1"

source venv3.8/bin/activate

for n in $(seq 1 $n_run)
do
    printf "Run ${n}/${n_run}\n"
    seed=$RANDOM
    comm="python algorithms/LGMARL/train_lgmarl.py --seed ${seed}
    --experiment_name ${experiment_name}
    --FT_env_name ${FT_env_name}
    --model_dir ${model_dir}
    --n_steps ${n_steps}
    --lr ${lr}
    --FT_magym_env_size ${FT_magym_env_size}
    --cuda_device ${cuda_device}
    --adapt_run
    --FT_freeze_lang"
    printf "Starting training with command:\n${comm}\n\nSEED IS ${seed}\n"
    eval $comm
    printf "DONE\n\n"
done