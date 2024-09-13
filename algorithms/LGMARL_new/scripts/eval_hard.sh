#!/bin/sh
n_parallel_envs=24 # 24 is number of hard scenarios
eval_scenario="algorithms/LGMARL_new/src/envs/magym_PredPrey/hard_eval.json"
n_eval_runs=100

model_dir="models/magym_PredPrey_new/9o5_noc/run3,models/magym_PredPrey_new/9o5_noc/run6"
# model_dir="models/magym_PredPrey_new/9o5_Diff_perf/run3,models/magym_PredPrey_new/9o5_Diff_perf/run9"
# model_dir="models/magym_PredPrey_new/9o5_Diff_langsup/run10,models/magym_PredPrey_new/9o5_Diff_langsup/run3"
# model_dir="models/magym_PredPrey_new/9o5_Diff_ec2/run9,models/magym_PredPrey_new/9o5_Diff_ec2/run5"

seed=$RANDOM
comm="python algorithms/LGMARL_new/eval_zst.py 
    --seed ${seed}
    --model_dir ${model_dir} 
    --n_steps 100 
    --eval_scenario ${eval_scenario} 
    --n_eval_runs ${n_eval_runs}
    --n_parallel_envs ${n_parallel_envs}"
echo $comm
eval $comm