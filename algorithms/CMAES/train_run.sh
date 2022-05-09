#!/bin/sh
n_run=1
env="coop_push_scenario/coop_push_scenario_sparse.py"
model_name="cmaes_2a_fo_rel_5eps_nocol"
sce_conf_path="configs/2a_1o_fo_rel_nocol.json"
n_episodes=200000
n_eps_per_eval=15
hidden_dim=8

for n in $(seq 1 $n_run)
do
    printf "Run ${n}/${n_run}\n"
    seed=$RANDOM
    comm="python algorithms/CMAES/train_cmaes.py ${env} ${model_name} \
--sce_conf_path ${sce_conf_path} --n_episodes ${n_episodes} --seed ${seed} \
--n_eps_per_eval ${n_eps_per_eval} --hidden_dim ${hidden_dim}"
    printf "Starting training with command:\n${comm}\n\nSEED IS ${seed}\n"
    eval $comm
    printf "DONE\n\n"
done

