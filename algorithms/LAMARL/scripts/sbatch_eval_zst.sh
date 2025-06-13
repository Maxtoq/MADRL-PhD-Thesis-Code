#!/bin/bash
#SBATCH --partition=hard
#SBATCH --job-name=zst
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=1000
#SBATCH --mail-type=ALL
#SBATCH --mail-user=maxime.toquebiau@sorbonne.universite.fr
#SBATCH --output=outputs/%x-%j.out

source venv/bin/activate

n_parallel_envs=250
env_name="magym_CoordPlace_RGB"
n_eval_runs=1
episode_length=50
n_agents=2
cuda_device="cuda:0"

ulimit -n 2048
ulimit -n

# model_dir="models/magym_PredPrey_new/9o5_noc/run3,models/magym_PredPrey_new/9o5_noc/run6,models/magym_PredPrey_new/9o5_noc/run4,models/magym_PredPrey_new/9o5_noc/run15"
# model_dir="models/magym_PredPrey_new/9o5_Diff_perf/run16,models/magym_PredPrey_new/9o5_Diff_perf/run9,models/magym_PredPrey_new/9o5_Diff_perf/run3,models/magym_PredPrey_new/9o5_Diff_perf/run7" # 16 9 7 12
# model_dir="models/magym_PredPrey_new/9o5_Diff_langsup/run10,models/magym_PredPrey_new/9o5_Diff_langsup/run3,models/magym_PredPrey_new/9o5_Diff_langsup/run13,models/magym_PredPrey_new/9o5_Diff_langsup/run5" # 10 3 13 5 
# model_dir="models/magym_PredPrey_new/9o5_Diff_ec2/run9,models/magym_PredPrey_new/9o5_Diff_ec2/run5,models/magym_PredPrey_new/9o5_Diff_ec2/run8,models/magym_PredPrey_new/9o5_Diff_ec2/run4" # 9 5 8 4
# model_dir="models/magym_PredPrey_new/9o5_Diff_edl/run16,models/magym_PredPrey_new/9o5_Diff_edl/run5,models/magym_PredPrey_new/9o5_Diff_edl/run11,models/magym_PredPrey_new/9o5_Diff_edl/run12"

# model_dir="models/magym_PredPrey_new/9o5SA_Diff_noc/run29,models/magym_PredPrey_new/9o5SA_Diff_noc/run24,models/magym_PredPrey_new/9o5SA_Diff_noc/run30,models/magym_PredPrey_new/9o5SA_Diff_noc/run18"
# model_dir="models/magym_PredPrey_new/9o5SA_Diff_perf/run17,models/magym_PredPrey_new/9o5SA_Diff_perf/run19,models/magym_PredPrey_new/9o5SA_Diff_perf/run14,models/magym_PredPrey_new/9o5SA_Diff_perf/run21" 
# model_dir="models/magym_PredPrey_new/9o5SA_Diff_langsup/run4,models/magym_PredPrey_new/9o5SA_Diff_langsup/run15,models/magym_PredPrey_new/9o5SA_Diff_langsup/run5,models/magym_PredPrey_new/9o5SA_Diff_langsup/run14" 
# model_dir="models/magym_PredPrey_new/9o5SA_Diff_ec2/run22,models/magym_PredPrey_new/9o5SA_Diff_ec2/run21,models/magym_PredPrey_new/9o5SA_Diff_ec2/run16,models/magym_PredPrey_new/9o5SA_Diff_ec2/run29" 
# model_dir="models/magym_PredPrey_new/9o5SA_Diff_edl/run14,models/magym_PredPrey_new/9o5SA_Diff_edl/run5,models/magym_PredPrey_new/9o5SA_Diff_edl/run8,models/magym_PredPrey_new/9o5SA_Diff_edl/run3"

# model_dir="models/magym_Foraging_RGB/18_ec2_ae/run7,models/magym_Foraging_RGB/18_ec2_ae/run6,models/magym_Foraging_RGB/18_ec2_ae/run1,models/magym_Foraging_RGB/18_ec2_ae/run4"

# model_dir="models/mpe_simple_color_reference/ec4_LG/run6,models/mpe_simple_color_reference/ec4_LG/run7,models/mpe_simple_color_reference/ec4_LG/run1,models/mpe_simple_color_reference/ec4_LG/run4"
# model_dir="models/mpe_simple_color_reference/ec2_AE/run2,models/mpe_simple_color_reference/ec2_AE/run7,models/mpe_simple_color_reference/ec2_AE/run1,models/mpe_simple_color_reference/ec2_AE/run5"
# model_dir="models/mpe_simple_color_reference/ec2/run4,models/mpe_simple_color_reference/ec2/run7,models/mpe_simple_color_reference/ec2/run1,models/mpe_simple_color_reference/ec2/run3"
# model_dir="models/mpe_simple_color_reference/noc/run3,models/mpe_simple_color_reference/noc/run7,models/mpe_simple_color_reference/noc/run4,models/mpe_simple_color_reference/noc/run1"

# model_dir="models/mpe_PredPrey_shape/lang_cont/run3,models/mpe_PredPrey_shape/lang_cont/run6,models/mpe_PredPrey_shape/lang_cont/run1,models/mpe_PredPrey_shape/lang_cont/run4"

# model_dir="models/magym_CoordPlace_RGB/2a_noc/run6,models/magym_CoordPlace_RGB/2a_noc/run7,models/magym_CoordPlace_RGB/2a_noc/run1,models/magym_CoordPlace_RGB/2a_noc/run4"
model_dir="models/magym_CoordPlace_RGB/2a_ec4_lg/run6,models/magym_CoordPlace_RGB/2a_ec4_lg/run5,models/magym_CoordPlace_RGB/2a_ec4_lg/run3,models/magym_CoordPlace_RGB/2a_ec4_lg/run2"

seed=$RANDOM 
comm="python algorithms/LAMARL/eval_zst.py 
    --seed ${seed}
    --cuda_device ${cuda_device}
    --model_dir ${model_dir} 
    --env_name ${env_name}
    --n_steps 100 
    --n_eval_runs ${n_eval_runs}
    --episode_length ${episode_length}
    --n_parallel_envs ${n_parallel_envs}
    --n_agents ${n_agents}"
    # --eval_scenario ${eval_scenario} 
    # --use_render
    # --render_wait_input"
echo $comm
eval $comm
exit 0