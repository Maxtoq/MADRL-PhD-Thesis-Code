#!/bin/sh
n_run=15
experiment_name="mappo_2a_4050_JIM"
n_rollout_threads=250
n_steps=5000000
algorithm_name="mappo"
ppo_epoch=15
env_name="rel_overgen"
ro_n_agents=2
ro_state_dim=40
ro_optim_diff_coeff=50.0
ir_algo="e2s_noveld"
ir_mode="central"
ir_coeff=1.0
ir_enc_dim=64
ir_hidden_dim=128
cuda_device="cuda:3"

source venv3.8/bin/activate

for n in $(seq 1 $n_run)
do
    printf "Run ${n}/${n_run}\n"
    seed=$RANDOM
    comm="python algorithms/MAPPO/train.py --seed ${seed}\
    --experiment_name ${experiment_name}\
    --n_rollout_threads ${n_rollout_threads}\
    --n_steps ${n_steps}\
    --algorithm_name ${algorithm_name}\
    --ppo_epoch ${ppo_epoch}\
    --env_name ${env_name}\
    --ro_state_dim ${ro_state_dim}\
    --ro_n_agents ${ro_n_agents}\
    --ro_optim_diff_coeff ${ro_optim_diff_coeff}\
    --ir_algo ${ir_algo}\
    --ir_mode ${ir_mode}\
    --ir_coeff ${ir_coeff}\
    --ir_enc_dim ${ir_enc_dim}\
    --ir_hidden_dim ${ir_hidden_dim}\
    --cuda_device ${cuda_device}"
    printf "Starting training with command:\n${comm}\n\nSEED IS ${seed}\n"
    eval $comm
    printf "DONE\n\n"
done