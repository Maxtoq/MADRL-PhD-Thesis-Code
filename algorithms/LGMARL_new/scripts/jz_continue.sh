#!/bin/bash
#SBATCH --partition=gpu_p2
#SBATCH --job-name=ad
#SBATCH --nodes=1
#SBATCH --gres=gpu:1 
#SBATCH --time=20:00:00
#SBATCH --output=outputs/%x-%j.out
#SBATCH -A bqo@v100

runs=( 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 )
model_dir="models/magym_PredPrey_new/Adapt_9o5SA-noSA_Diff_noc/"
n_steps=5000000
cuda_device="cuda:0"

source venv/bin/activate

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