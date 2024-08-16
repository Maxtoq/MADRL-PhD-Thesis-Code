#!/bin/bash
#SBATCH --partition=hard
#SBATCH --job-name=adapt
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=5000
#SBATCH --mail-type=ALL
#SBATCH --mail-user=maxime.toquebiau@sorbonne.universite.fr
#SBATCH --output=outputs/%x-%j.out

source venv/bin/activate

n_run=8
experiment_name="Adapt12_ACC_lang"
n_steps=5000000
lr=0.0005 # default 0.0005
FT_env_name="magym_PredPrey"
FT_magym_env_size=12
# model_dir="models/magym_PredPrey/ACC_9o5_perf_alldynawl/run15"
# model_dir="models/magym_PredPrey/9o5_ACC_no/run14/"
model_dir="models/magym_PredPrey/9o5_ACC_lang/run9/"
cuda_device="cuda:0"

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
    --adapt_run"
    printf "Starting training with command:\n${comm}\n\nSEED IS ${seed}\n"
    eval $comm
    printf "DONE\n\n"
done