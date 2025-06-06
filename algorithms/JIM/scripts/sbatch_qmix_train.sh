#!/bin/bash
#SBATCH --partition=hard
#SBATCH --nodelist=lizzy
#SBATCH --job-name=4a_40_09_LIM
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=7200
#SBATCH --mail-type=ALL
#SBATCH --mail-user=maxime.toquebiau@sorbonne.universite.fr
#SBATCH --output=outputs/%x-%j.out

source venv/bin/activate

n_run=3
env="algorithms/JIM/scenarios/rel_overgen.py"
model_name="qmix_4a_40_09_LIM"
sce_conf_path="configs/2a_pol.json"
n_frames=12000000
n_explo_frames=10000000
episode_length=100 # def 100
frames_per_update=100
eval_every=100000
eval_scenar_file="eval_scenarios/hard_corners_24.json"
init_explo_rate=0.1
epsilon_decay_fn="linear"
intrinsic_reward_mode="local"
intrinsic_reward_algo="e2snoveld"
int_reward_coeff=1.0
int_reward_decay_fn="constant"
gamma=0.99
int_rew_enc_dim=32 # def 16, JIM 90, LIM 30
int_rew_hidden_dim=64 # def 64, JIM 1024, LIM 256
scale_fac=0.5 # def 0.5
int_rew_lr=0.0001 # def 0.0001
state_dim=40
optimal_diffusion_coeff=0.9
suboptimal_diffusion_coeff=0.08
ro_n_agents=4
magym_env_size=8
cuda_device="cuda:0"

for n in $(seq 1 $n_run)
do
    printf "Run ${n}/${n_run}\n"
    seed=$RANDOM
    comm="python algorithms/JIM/train_qmix.py --env_path ${env} --model_name ${model_name} --sce_conf_path ${sce_conf_path} --seed ${seed} \
--n_frames ${n_frames} --cuda_device ${cuda_device} --gamma ${gamma} --episode_length ${episode_length} --frames_per_update ${frames_per_update} \
--init_explo_rate ${init_explo_rate} --n_explo_frames ${n_explo_frames} --use_per \
--intrinsic_reward_mode ${intrinsic_reward_mode} --intrinsic_reward_algo ${intrinsic_reward_algo} \
--int_reward_coeff ${int_reward_coeff} --int_reward_decay_fn ${int_reward_decay_fn} \
--scale_fac ${scale_fac} --int_rew_lr ${int_rew_lr} --int_rew_enc_dim ${int_rew_enc_dim} --int_rew_hidden_dim ${int_rew_hidden_dim} \
--eval_every ${eval_every} \
--state_dim ${state_dim} --optimal_diffusion_coeff ${optimal_diffusion_coeff} --ro_n_agents ${ro_n_agents}"
    printf "Starting training with command:\n${comm}\n\nSEED IS ${seed}\n"
    eval $comm
    printf "DONE\n\n"
done