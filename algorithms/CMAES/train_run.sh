#!/bin/sh
n_run=2
env="coop_push_scenario/coop_push_scenario_closed.py"
model_name="cmaes_2a_fo_cont_abs_distrew"
sce_conf_path="configs/2a_1o_fo_abs_distrew.json"
n_evals=3500
n_eps_per_eval=8
hidden_dim=8

for n in $(seq 1 $n_run)
do
    printf "Run ${n}/${n_run}\n"
    seed=$RANDOM
    comm="python algorithms/CMAES/train_cmaes.py ${env} ${model_name} --sce_conf_path ${sce_conf_path} --n_eval ${n_eval} --seed ${seed} --n_eps_per_eval ${n_eps_per_eval} --hidden_dim ${hidden_dim}"
    printf "Starting training with command:\n${comm}\n\nSEED IS ${seed}\n"
    python algorithms/CMAES/train_cmaes.py ${env} ${model_name} --sce_conf_path ${sce_conf_path} --seed ${seed} --n_eps_per_eval ${n_eps_per_eval} --hidden_dim ${hidden_dim}
    printf "DONE\n\n"
done

