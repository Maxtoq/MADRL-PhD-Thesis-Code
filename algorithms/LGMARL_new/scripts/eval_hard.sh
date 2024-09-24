#!/bin/sh
n_parallel_envs=240 # 24 is number of hard scenarios
eval_scenario="algorithms/LGMARL_new/src/envs/magym_PredPrey/hard_eval_12.json"
n_eval_runs=1000
cuda_device="cuda:1"

# model_dir="models/magym_PredPrey_new/9o5_noc/run3,models/magym_PredPrey_new/9o5_noc/run3,models/magym_PredPrey_new/9o5_noc/run6,models/magym_PredPrey_new/9o5_noc/run6"
# model_dir="models/magym_PredPrey_new/9o5_Diff_perf/run3,models/magym_PredPrey_new/9o5_Diff_perf/run3,models/magym_PredPrey_new/9o5_Diff_perf/run9,models/magym_PredPrey_new/9o5_Diff_perf/run9"
model_dir="models/magym_PredPrey_new/9o5_Diff_langsup/run10,models/magym_PredPrey_new/9o5_Diff_langsup/run10,models/magym_PredPrey_new/9o5_Diff_langsup/run3,models/magym_PredPrey_new/9o5_Diff_langsup/run3"
# model_dir="models/magym_PredPrey_new/9o5_Diff_ec2/run9,models/magym_PredPrey_new/9o5_Diff_ec2/run9,models/magym_PredPrey_new/9o5_Diff_ec2/run9,models/magym_PredPrey_new/9o5_Diff_ec2/run5"

seed=$RANDOM # 27425
comm="python algorithms/LGMARL_new/eval_zst.py 
    --seed ${seed}
    --cuda_device ${cuda_device}
    --model_dir ${model_dir} 
    --n_steps 100 
    --eval_scenario ${eval_scenario} 
    --n_eval_runs ${n_eval_runs}
    --n_parallel_envs ${n_parallel_envs}"
    # --use_render
    # --render_wait_input"
echo $comm
eval $comm