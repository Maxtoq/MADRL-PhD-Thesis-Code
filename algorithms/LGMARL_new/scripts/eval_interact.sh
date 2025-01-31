#!/bin/sh
source venv3.8/bin/activate

n_parallel_envs=250
# eval_scenario="algorithms/LGMARL_new/src/envs/magym_PredPrey/mid_eval_6.json"
env_name="magym_PredPrey_RGB"
n_eval_runs=100
episode_length=50
cuda_device="cuda:0"
comm_langground_pt="results/data/lamarl_data/PPrgb_18_langground.pt"

model_dir="models/magym_PredPrey_RGB/18s50np_lang_ce0/run13" 


seed=$RANDOM # 27425
comm="python algorithms/LGMARL_new/eval_interact.py 
    --seed ${seed}
    --cuda_device ${cuda_device}
    --model_dir ${model_dir} 
    --env_name ${env_name}
    --n_steps 100 
    --n_eval_runs ${n_eval_runs}
    --episode_length ${episode_length}
    --n_parallel_envs ${n_parallel_envs}
    --comm_langground_pt ${comm_langground_pt}"
    # --eval_scenario ${eval_scenario} 
    # --use_render
    # --render_wait_input"
echo $comm
eval $comm
exit 0