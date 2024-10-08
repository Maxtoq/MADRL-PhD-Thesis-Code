#!/bin/bash
#SBATCH --partition=hard
#SBATCH --job-name=zst
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=5000
#SBATCH --mail-type=ALL
#SBATCH --mail-user=maxime.toquebiau@sorbonne.universite.fr
#SBATCH --output=outputs/%x-%j.out

source venv/bin/activate

n_parallel_envs=240 # 24 is number of hard scenarios
# eval_scenario="algorithms/LGMARL_new/src/envs/magym_PredPrey/mid_eval_6.json"
n_eval_runs=1000
cuda_device="cuda:0"

# model_dir="models/magym_PredPrey_new/9o5_noc/run3,models/magym_PredPrey_new/9o5_noc/run6,models/magym_PredPrey_new/9o5_noc/run4,models/magym_PredPrey_new/9o5_noc/run15"
# model_dir="models/magym_PredPrey_new/9o5_Diff_perf/run16,models/magym_PredPrey_new/9o5_Diff_perf/run9,models/magym_PredPrey_new/9o5_Diff_perf/run3,models/magym_PredPrey_new/9o5_Diff_perf/run7" # 16 9 7 12
# model_dir="models/magym_PredPrey_new/9o5_Diff_langsup/run10,models/magym_PredPrey_new/9o5_Diff_langsup/run3,models/magym_PredPrey_new/9o5_Diff_langsup/run13,models/magym_PredPrey_new/9o5_Diff_langsup/run5" # 10 3 13 5 
# model_dir="models/magym_PredPrey_new/9o5_Diff_ec2/run9,models/magym_PredPrey_new/9o5_Diff_ec2/run5,models/magym_PredPrey_new/9o5_Diff_ec2/run8,models/magym_PredPrey_new/9o5_Diff_ec2/run4" # 9 5 8 4
# model_dir="models/magym_PredPrey_new/9o5_Diff_edl/run16,models/magym_PredPrey_new/9o5_Diff_edl/run5,models/magym_PredPrey_new/9o5_Diff_edl/run11,models/magym_PredPrey_new/9o5_Diff_edl/run12"

# model_dir="models/magym_PredPrey_new/9o5SA_Diff_noc/run29,models/magym_PredPrey_new/9o5SA_Diff_noc/run24,models/magym_PredPrey_new/9o5SA_Diff_noc/run30,models/magym_PredPrey_new/9o5SA_Diff_noc/run18"
# model_dir="models/magym_PredPrey_new/9o5SA_Diff_perf/run17,models/magym_PredPrey_new/9o5SA_Diff_perf/run19,models/magym_PredPrey_new/9o5SA_Diff_perf/run14,models/magym_PredPrey_new/9o5SA_Diff_perf/run21" 
# model_dir="models/magym_PredPrey_new/9o5SA_Diff_langsup/run4,models/magym_PredPrey_new/9o5SA_Diff_langsup/run15,models/magym_PredPrey_new/9o5SA_Diff_langsup/run5,models/magym_PredPrey_new/9o5SA_Diff_langsup/run14" 
model_dir="models/magym_PredPrey_new/9o5SA_Diff_ec2/run22,models/magym_PredPrey_new/9o5SA_Diff_ec2/run21,models/magym_PredPrey_new/9o5SA_Diff_ec2/run16,models/magym_PredPrey_new/9o5SA_Diff_ec2/run29" 
# model_dir="models/magym_PredPrey_new/9o5SA_Diff_edl/run14,models/magym_PredPrey_new/9o5SA_Diff_edl/run5,models/magym_PredPrey_new/9o5SA_Diff_edl/run8,models/magym_PredPrey_new/9o5SA_Diff_edl/run3"

seed=$RANDOM 
comm="python algorithms/LGMARL_new/eval_zst.py 
    --seed ${seed}
    --cuda_device ${cuda_device}
    --model_dir ${model_dir} 
    --n_steps 100 
    --n_eval_runs ${n_eval_runs}
    --n_parallel_envs ${n_parallel_envs}"
    # --eval_scenario ${eval_scenario} 
    # --use_render
    # --render_wait_input"
echo $comm
eval $comm
exit 0