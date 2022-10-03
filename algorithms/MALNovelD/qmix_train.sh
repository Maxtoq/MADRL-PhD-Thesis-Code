#!/bin/sh
n_run=1
env="algorithms/MALNovelD/scenarios/coop_push_scenario_sparse_HARDER.py"
model_name="qmix_manoveld_pol"
sce_conf_path="configs/2a_1o_pol_rel.json"
n_frames=10000000
frames_per_update=100
eval_every=500000
eval_scenar_file="eval_scenarios/hard_corners_24.json"
init_explo_rate=0.6
epsilon_decay_fn="linear"
model_type="qmix_manoveld"
int_reward_coeff=1.0
cuda_device="cuda:2"

for n in $(seq 1 $n_run)
do
    printf "Run ${n}/${n_run}\n"
    seed=$RANDOM
    comm="python algorithms/MALNovelD/train_qmix.py --env_path ${env} \
--model_name ${model_name} --sce_conf_path ${sce_conf_path} --seed ${seed} \
--n_frames ${n_frames} --cuda_device ${cuda_device} \
--eval_every ${eval_every} --eval_scenar_file ${eval_scenar_file} \
--frames_per_update ${frames_per_update} --init_explo_rate ${init_explo_rate} \
--model_type ${model_type} --int_reward_coeff ${int_reward_coeff}"
    printf "Starting training with command:\n${comm}\n\nSEED IS ${seed}\n"
    eval $comm
    printf "DONE\n\n"
done
