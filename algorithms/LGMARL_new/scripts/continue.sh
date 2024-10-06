#!/bin/sh
runs=( 6 )
model_dir="models/magym_PredPrey_new/9o5SA_obs/"
n_steps=10000000
cuda_device="cuda:2"

source venv3.8/bin/activate

n_runs=${#runs[@]}
for ((i=0; i < $n_runs; i++))
do
    printf "Run $((i+1))/$n_runs\n"
    printf "Continuing run ${model_dir}run${runs[$i]}\n"
    comm="python algorithms/LGMARL_new/train_lgmarl_diff.py
    --model_dir ${model_dir}run${runs[$i]}
    --n_steps ${n_steps}
    --cuda_device ${cuda_device}
    --continue_run"
    printf "Starting training with command:\n${comm}\n\nSEED IS ${seed}\n"
    eval $comm
    printf "DONE\n\n"
done