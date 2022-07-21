#!/bin/sh
n_run=1
env="coop_push_scenario/coop_push_scenario_sparse.py"
model_name="TEST"
sce_conf_path="configs/2a_1o_fo_rel.json"
n_episodes=100000
n_exploration_eps=100000
n_updates=100000
buffer_length=1000000
lr=0.0007
gamma=0.99
tau=0.01
hidden_dim=64
batch_size=512
n_rollout_threads=1
n_training_per_updates=1
init_noise_scale=1.0
exploration_strategy="e_greedy"
cuda_device="cuda:1"

for n in $(seq 1 $n_run)
do
    printf "Run ${n}/${n_run}\n"
    seed=$RANDOM
    comm="python algorithms/MADDPG/train.py ${env} ${model_name} --sce_conf_path ${sce_conf_path} --seed ${seed} \
--n_episodes ${n_episodes} --n_exploration_eps ${n_exploration_eps} --n_updates ${n_updates} --lr ${lr} \
--hidden_dim ${hidden_dim} --n_rollout_threads ${n_rollout_threads} --n_training_per_updates ${n_training_per_updates} \
--cuda_device ${cuda_device} --gamma ${gamma} --tau ${tau} --init_noise_scale ${init_noise_scale} \
--exploration_strategy ${exploration_strategy} --buffer_length ${buffer_length} --batch_size ${batch_size} \
--discrete_action --shared_params"
    printf "Starting training with command:\n${comm}\n\nSEED IS ${seed}\n"
    eval $comm
    printf "DONE\n\n"
done
