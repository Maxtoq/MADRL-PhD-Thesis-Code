#!/bin/sh
source venv3.8/bin/activate

n_parallel_envs=250
# eval_scenario="algorithms/LAMARL/src/envs/magym_PredPrey/mid_eval_6.json"
env_name="mpe_simple_color_reference"
n_eval_runs=1
episode_length=100
cuda_device="cuda:0"
comm_langground_pt="results/data/lamarl_data/Frgb_18_langground.pt"

# model_dir="models/magym_PredPrey_new/9o5_noc/run3,models/magym_PredPrey_new/9o5_noc/run6,models/magym_PredPrey_new/9o5_noc/run4,models/magym_PredPrey_new/9o5_noc/run15"
# model_dir="models/magym_PredPrey_new/9o5_Diff_perf/run16,models/magym_PredPrey_new/9o5_Diff_perf/run9,models/magym_PredPrey_new/9o5_Diff_perf/run3,models/magym_PredPrey_new/9o5_Diff_perf/run7" # 16 9 7 12
# model_dir="models/magym_PredPrey_new/9o5_Diff_langsup/run10,models/magym_PredPrey_new/9o5_Diff_langsup/run3,models/magym_PredPrey_new/9o5_Diff_langsup/run13,models/magym_PredPrey_new/9o5_Diff_langsup/run5" # 10 3 13 5 
# model_dir="models/magym_PredPrey_new/9o5_Diff_ec2/run9,models/magym_PredPrey_new/9o5_Diff_ec2/run5,models/magym_PredPrey_new/9o5_Diff_ec2/run8,models/magym_PredPrey_new/9o5_Diff_ec2/run4" # 9 5 8 4
# model_dir="models/magym_PredPrey_new/9o5_Diff_edl/run16,models/magym_PredPrey_new/9o5_Diff_edl/run5,models/magym_PredPrey_new/9o5_Diff_edl/run11,models/magym_PredPrey_new/9o5_Diff_edl/run12"

# model_dir="models/magym_PredPrey_new/9o5SA_noc/run4,models/magym_PredPrey_new/9o5SA_noc/run3,models/magym_PredPrey_new/9o5SA_noc/run8,models/magym_PredPrey_new/9o5SA_noc/run1"
# model_dir="models/magym_PredPrey_new/9o5SA_Diff_perf/run17,models/magym_PredPrey_new/9o5SA_Diff_perf/run19,models/magym_PredPrey_new/9o5SA_Diff_perf/run20,models/magym_PredPrey_new/9o5SA_Diff_perf/run21" 
# model_dir="models/magym_PredPrey_new/9o5SA_Diff_langsup/run4,models/magym_PredPrey_new/9o5SA_Diff_langsup/run15,models/magym_PredPrey_new/9o5SA_Diff_langsup/run5,models/magym_PredPrey_new/9o5SA_Diff_langsup/run14" 
# model_dir="models/magym_PredPrey_new/9o5SA_Diff_ec2/run9,models/magym_PredPrey_new/9o5SA_Diff_ec2/run12,models/magym_PredPrey_new/9o5SA_Diff_ec2/run8,models/magym_PredPrey_new/9o5SA_Diff_ec2/run21" 
# model_dir="models/magym_PredPrey_new/9o5SA_Diff_edl/run14,models/magym_PredPrey_new/9o5SA_Diff_edl/run5,models/magym_PredPrey_new/9o5SA_Diff_edl/run8,models/magym_PredPrey_new/9o5SA_Diff_edl/run3"

# model_dir="models/magym_PredPrey_new/Ad_9o5SA_15o5_noc/run4,models/magym_PredPrey_new/Ad_9o5SA_15o5_noc/run1,models/magym_PredPrey_new/Ad_9o5SA_15o5_noc/run2,models/magym_PredPrey_new/Ad_9o5SA_15o5_noc/run7"
# model_dir="models/magym_PredPrey_new/Ad_9o5SA_15o5_perf/run8,models/magym_PredPrey_new/Ad_9o5SA_15o5_perf/run7,models/magym_PredPrey_new/Ad_9o5SA_15o5_perf/run6,models/magym_PredPrey_new/Ad_9o5SA_15o5_perf/run2" 
# model_dir="models/magym_PredPrey_new/Ad_9o5SA_15o5_langsup/run5,models/magym_PredPrey_new/Ad_9o5SA_15o5_langsup/run6,models/magym_PredPrey_new/Ad_9o5SA_15o5_langsup/run9,models/magym_PredPrey_new/Ad_9o5SA_15o5_langsup/run7" 
# model_dir="models/magym_PredPrey_new/Ad_9o5SA_15o5_ec2/run8,models/magym_PredPrey_new/Ad_9o5SA_15o5_ec2/run7,models/magym_PredPrey_new/Ad_9o5SA_15o5_ec2/run6,models/magym_PredPrey_new/Ad_9o5SA_15o5_ec2/run2" 
# model_dir="models/magym_PredPrey_new/Ad_9o5SA_15o5_edl/run9,models/magym_PredPrey_new/Ad_9o5SA_15o5_edl/run7,models/magym_PredPrey_new/Ad_9o5SA_15o5_edl/run3,models/magym_PredPrey_new/Ad_9o5SA_15o5_edl/run6"

# model_dir="models/Foraging/18o5_lang/run4,models/magym_PredPrey_new/Ad_9o5SA_15o5_noc/run1,models/magym_PredPrey_new/Ad_9o5SA_15o5_noc/run2,models/magym_PredPrey_new/Ad_9o5SA_15o5_noc/run7"
# EEmodel_dir="models/Foraging/18o5_edl/run8,models/magym_PredPrey_new/Ad_9o5SA_15o5_perf/run7,models/magym_PredPrey_new/Ad_9o5SA_15o5_perf/run6,models/magym_PredPrey_new/Ad_9o5SA_15o5_perf/run2" 
# EEmodel_dir="models/Foraging/18o5_ec2/run5,models/magym_PredPrey_new/Ad_9o5SA_15o5_langsup/run6,models/magym_PredPrey_new/Ad_9o5SA_15o5_langsup/run9,models/magym_PredPrey_new/Ad_9o5SA_15o5_langsup/run7" 
# EEmodel_dir="models/Foraging/18o5_no_comm/run8,models/magym_PredPrey_new/Ad_9o5SA_15o5_ec2/run7,models/magym_PredPrey_new/Ad_9o5SA_15o5_ec2/run6,models/magym_PredPrey_new/Ad_9o5SA_15o5_ec2/run2" 
# EEmodel_dir="models/Foraging/18o5_perf/run9,models/magym_PredPrey_new/Ad_9o5SA_15o5_edl/run7,models/magym_PredPrey_new/Ad_9o5SA_15o5_edl/run3,models/magym_PredPrey_new/Ad_9o5SA_15o5_edl/run6"

# model_dir="models/magym_PredPrey_RGB/18s50np_noc/run11,models/magym_PredPrey_RGB/18s50np_noc/run15,models/magym_PredPrey_RGB/18s50np_noc/run10,models/magym_PredPrey_RGB/18s50np_noc/run4"
# model_dir="models/magym_PredPrey_RGB/18s50np_lang_ce0/run5,models/magym_PredPrey_RGB/18s50np_lang_ce0/run1,models/magym_PredPrey_RGB/18s50np_lang_ce0/run14,models/magym_PredPrey_RGB/18s50np_lang_ce0/run6" 
# model_dir="models/magym_PredPrey_RGB/18s50np_ec2_ae/run4,models/magym_PredPrey_RGB/18s50np_ec2_ae/run7,models/magym_PredPrey_RGB/18s50np_ec2_ae/run6,models/magym_PredPrey_RGB/18s50np_ec2_ae/run2" 
# model_dir="models/magym_PredPrey_RGB/18s50np_ec2/run12,models/magym_PredPrey_RGB/18s50np_ec2/run3,models/magym_PredPrey_RGB/18s50np_ec2/run2,models/magym_PredPrey_RGB/18s50np_ec2/run11" 
# model_dir="models/magym_PredPrey_RGB/18s50np_ec4_lg/run7,models/magym_PredPrey_RGB/18s50np_ec4_lg/run5,models/magym_PredPrey_RGB/18s50np_ec4_lg/run1,models/magym_PredPrey_RGB/18s50np_ec4_lg/run2"

# model_dir="models/magym_Foraging_RGB/18_lang_ce0/run2,models/magym_Foraging_RGB/18_lang_ce0/run4,models/magym_Foraging_RGB/18_lang_ce0/run3,models/magym_Foraging_RGB/18_lang_ce0/run1"
# model_dir="models/magym_Foraging_RGB/18_ec4_lg/run1,models/magym_Foraging_RGB/18_ec4_lg/run5,models/magym_Foraging_RGB/18_ec4_lg/run6,models/magym_Foraging_RGB/18_ec4_lg/run2"
# model_dir="models/magym_Foraging_RGB/18_ec2_ae/run7,models/magym_Foraging_RGB/18_ec2_ae/run1,models/magym_Foraging_RGB/18_ec2_ae/run1,models/magym_Foraging_RGB/18_ec2_ae/run1"

# model_dir="models/mpe_PredPrey_shape/ec2_AE_cont/run7,models/mpe_PredPrey_shape/ec2_AE_cont/run5,models/mpe_PredPrey_shape/ec2_AE_cont/run1,models/mpe_PredPrey_shape/ec2_AE_cont/run4"
# model_dir="models/mpe_PredPrey_shape/ec2_cont/run3,models/mpe_PredPrey_shape/ec2_cont/run5,models/mpe_PredPrey_shape/ec2_cont/run1,models/mpe_PredPrey_shape/ec2_cont/run4"
# model_dir="models/mpe_PredPrey_shape/ec4_LG_cont/run7,models/mpe_PredPrey_shape/ec4_LG_cont/run2,models/mpe_PredPrey_shape/ec4_LG_cont/run1,models/mpe_PredPrey_shape/ec4_LG_cont/run6"
# model_dir="models/mpe_PredPrey_shape/lang_cont/run3,models/mpe_PredPrey_shape/lang_cont/run6,models/mpe_PredPrey_shape/lang_cont/run1,models/mpe_PredPrey_shape/lang_cont/run4"
# model_dir="models/mpe_PredPrey_shape/noc_cont/run3,models/mpe_PredPrey_shape/noc_cont/run7,models/mpe_PredPrey_shape/noc_cont/run1,models/mpe_PredPrey_shape/noc_cont/run4"

model_dir="models/mpe_simple_color_reference/lang_smolbatch/run3,models/mpe_simple_color_reference/lang_smolbatch/run2,models/mpe_simple_color_reference/lang_smolbatch/run1,models/mpe_simple_color_reference/lang_smolbatch/run4"


seed=$RANDOM # 27425
comm="python algorithms/LAMARL/eval_zst.py 
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