#!/bin/sh
n_parallel_envs=24 # 24 is number of hard scenarios
eval_scenario="algorithms/LGMARL_new/src/envs/magym_PredPrey/hard_eval.json"
n_eval_runs=20

model_dir="models/magym_PredPrey_new/9o5SA_Diff_noc/run30,models/magym_PredPrey_new/9o5SA_Diff_noc/run29"

seed=$RANDOM
comm="python algorithms/LGMARL_new/eval_lgmarl.py 
    --seed ${seed}
    --model_dir ${model_dir} 
    --n_steps 100 
    --eval_scenario ${eval_scenario} 
    --n_eval_runs ${n_eval_runs}
    --n_parallel_envs ${n_parallel_envs}"
eval $comm