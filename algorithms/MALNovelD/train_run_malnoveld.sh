#!/bin/sh
n_run=1
env="algorithms/MALNovelD/scenarios/coop_push_scenario_parse_harder.py"
model_name="malnoveld_fo_maddpg_disc"
sce_conf_path="configs/2a_1o_fo_rel.json"
n_frames=10000000
buffer_length=1000000
lr=0.0007
gamma=0.99
tau=0.01
frames_per_lnoveld_update=1000
frames_per_language_update=1000
explo_strat="sample"
init_explo_rate=1.0
epsilon_decay_fn="exp"
frames_per_policy_update=100
int_reward_coeff=1.0
eval_every=500000
eval_scenar_file="eval_scenarios/hard_corners_24.json"
cuda_device="cuda:0"

for n in $(seq 1 $n_run)
do
    printf "Run ${n}/${n_run}\n"
    seed=$RANDOM
    comm="python algorithms/MALNovelD/train_malnoveld.py --env_path ${env} \
--model_name ${model_name} --sce_conf_path ${sce_conf_path} --seed ${seed} \
--n_frames ${n_frames} --lr ${lr} --cuda_device ${cuda_device} --gamma ${gamma} \
--tau ${tau} --explo_strat ${explo_strat} --init_explo_rate ${init_explo_rate} \
--frames_per_lnoveld_update ${frames_per_lnoveld_update} \
--frames_per_language_update ${frames_per_language_update} \
--epsilon_decay_fn ${epsilon_decay_fn} \
--buffer_length ${buffer_length} --eval_every ${eval_every} \
--eval_scenar_file ${eval_scenar_file} \
--frames_per_policy_update ${frames_per_policy_update} \
--int_reward_coeff ${int_reward_coeff} --discrete_action"
    printf "Starting training with command:\n${comm}\n\nSEED IS ${seed}\n"
    eval $comm
    printf "DONE\n\n"
done
