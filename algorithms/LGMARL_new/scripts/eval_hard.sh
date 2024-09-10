#!/bin/sh
n_parallel_envs=24 # 24 is number of hard scenarios
eval_scenar="algorithms/LGMARL_new/src/envs/magym_PredPrey/hard_eval.json"

model_dir="models/magym_PredPrey_new/9o5_Diff_perf/run10,models/magym_PredPrey_new/9o5_Diff_perf/run13"


python algorithms/LGMARL_new/eval_lgmarl.py --model_dir ${model_dir} --n_steps 100 --eval_scenario ${eval_scenar} --n_parallel_envs ${n_parallel_envs}

seed=$RANDOM
comm="python algorithms/LGMARL_new/eval_lgmarl.py 
    --seed ${seed}
    --model_dir ${model_dir} 
    --n_steps 100 
    --eval_scenario ${eval_scenar} 
    --n_parallel_envs ${n_parallel_envs}"
eval $comm