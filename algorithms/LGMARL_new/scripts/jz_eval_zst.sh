#!/bin/bash
#SBATCH --partition=gpu_p2
#SBATCH --job-name=zst_lg
#SBATCH --nodes=1
#SBATCH --gres=gpu:1 
#SBATCH --time=100:00:00
#SBATCH --qos=qos_gpu-t4
#SBATCH --output=outputs/%x-%j.out
#SBATCH -C v100
#SBATCH -A bqo@v100	

source venv/bin/activate

n_parallel_envs=250
# eval_scenario="algorithms/LGMARL_new/src/envs/magym_PredPrey/mid_eval_6.json"
env_name="magym_PredPrey_RGB"
n_eval_runs=24
cuda_device="cuda:0"

comm_langground_pt="results/data/lamarl_data/PPrgb_18_langground.pt"

# model_dir="models/magym_PredPrey_RGB/18s50np_noc/run11,models/magym_PredPrey_RGB/18s50np_noc/run15,models/magym_PredPrey_RGB/18s50np_noc/run10,models/magym_PredPrey_RGB/18s50np_noc/run4"
# model_dir="models/magym_PredPrey_RGB/18s50np_lang_ce0/run13,models/magym_PredPrey_RGB/18s50np_lang_ce0/run4,models/magym_PredPrey_RGB/18s50np_lang_ce0/run14,models/magym_PredPrey_RGB/18s50np_lang_ce0/run6" 
# model_dir="models/magym_PredPrey_RGB/18s50np_ec2_ae/run4,models/magym_PredPrey_RGB/18s50np_ec2_ae/run7,models/magym_PredPrey_RGB/18s50np_ec2_ae/run6,models/magym_PredPrey_RGB/18s50np_ec2_ae/run2" 
# model_dir="models/magym_PredPrey_RGB/18s50np_ec2/run12,models/magym_PredPrey_RGB/18s50np_ec2/run3,models/magym_PredPrey_RGB/18s50np_ec2/run2,models/magym_PredPrey_RGB/18s50np_ec2/run11" 
model_dir="models/magym_PredPrey_RGB/18s50np_ec4_lg/run7,models/magym_PredPrey_RGB/18s50np_ec4_lg/run5,models/magym_PredPrey_RGB/18s50np_ec4_lg/run1,models/magym_PredPrey_RGB/18s50np_ec4_lg/run2"


seed=$RANDOM # 27425
comm="python algorithms/LGMARL_new/eval_zst.py 
    --seed ${seed}
    --cuda_device ${cuda_device}
    --model_dir ${model_dir} 
    --env_name ${env_name}
    --n_steps 100 
    --n_eval_runs ${n_eval_runs}
    --n_parallel_envs ${n_parallel_envs}
    --comm_langground_pt ${comm_langground_pt}"
    # --eval_scenario ${eval_scenario} 
    # --use_render
    # --render_wait_input"
echo $comm
eval $comm
exit 0
