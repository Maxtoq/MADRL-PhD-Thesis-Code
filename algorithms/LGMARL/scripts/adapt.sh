#!/bin/sh
n_run=3
experiment_name="Adapt12_ACC_perf"
n_steps=2000000
lr=0.0005 # default 0.0005
FT_env_name="magym_PredPrey"
FT_magym_env_size=12
model_dir="models/magym_PredPrey/ACC_9o5_perf_alldynawl/run15"
# model_dir="models/magym_PredPrey/9o5_ACC_no/run14/"
cuda_device="cuda:2"

source venv3.8/bin/activate

for n in $(seq 1 $n_run)
do
    printf "Run ${n}/${n_run}\n"
    seed=$RANDOM
    comm="python algorithms/LGMARL/pretrain_lgmarl.py --seed ${seed}
    --experiment_name ${experiment_name}
    --FT_env_name ${FT_env_name}
    --model_dir ${model_dir}
    --n_steps ${n_steps}
    --lr ${lr}
    --FT_magym_env_size ${FT_magym_env_size}
    --cuda_device ${cuda_device}"
    printf "Starting training with command:\n${comm}\n\nSEED IS ${seed}\n"
    eval $comm
    printf "DONE\n\n"
done