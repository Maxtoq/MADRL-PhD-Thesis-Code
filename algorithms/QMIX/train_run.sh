#!/bin/sh
n_run=1
env="algorithms/QMIX/scenario/coop_push_scenario_closed.py"
model_name="qmix_fo_abs_disc_distrew"
sce_conf_path="configs/2a_1o_fo_abs_distrew.json"
n_episodes=600000
n_exploration_eps=600000
n_updates=150000
hidden_dim=64
n_rollout_threads=1
batch_size=1024

for n in $(seq 1 $n_run)
do
    printf "Run ${n}/${n_run}\n"
    seed=$RANDOM
    comm="python algorithms/QMIX/train.py --env_path ${env} --algorithm_name qmix \
    --model_name ${model_name} --sce_conf_path ${sce_conf_path} --seed ${seed} \
    --n_episodes ${n_episodes} --epsilon_anneal_time ${n_exploration_eps} --n_updates ${n_updates} \
    --hidden_dim ${hidden_dim} --n_rollout_threads ${n_rollout_threads} --batch_size ${batch_size}"
    printf "Starting training with command:\n${comm}\n\nSEED IS ${seed}\n"
    eval $comm
    printf "DONE\n\n"
done

