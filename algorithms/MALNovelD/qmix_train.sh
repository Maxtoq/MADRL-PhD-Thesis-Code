#!/bin/sh
n_run=3
env="algorithms/MALNovelD/scenarios/click_n_push2_fixed.py"
model_name="qmix_manoveld_fo"
sce_conf_path="configs/2a_1o_fo_rel.json"
n_frames=10000000
frames_per_update=100
eval_every=1000000
eval_scenar_file="eval_scenarios/hard_corners_24.json"
init_explo_rate=0.3
epsilon_decay_fn="linear"
model_type="qmix_manoveld"
int_reward_coeff=1.0
gamma=0.99
embed_dim=16 # default 16
nd_hidden_dim=64 # default 64
nd_scale_fac=0.5 # default 0.5
nd_lr=0.0001 # default 0.0001
cuda_device="cuda:0"

for n in $(seq 1 $n_run)
do
    printf "Run ${n}/${n_run}\n"
    seed=$RANDOM
    comm="python algorithms/MALNovelD/train_qmix.py --env_path ${env} \
--model_name ${model_name} --sce_conf_path ${sce_conf_path} --seed ${seed} \
--n_frames ${n_frames} --cuda_device ${cuda_device} --gamma ${gamma} \
--frames_per_update ${frames_per_update} --init_explo_rate ${init_explo_rate} \
--nd_scale_fac ${nd_scale_fac} --nd_lr ${nd_lr} --embed_dim ${embed_dim} \
--nd_hidden_dim ${nd_hidden_dim} \
--model_type ${model_type} --int_reward_coeff ${int_reward_coeff}"
# --eval_every ${eval_every} --eval_scenar_file ${eval_scenar_file} \
    printf "Starting training with command:\n${comm}\n\nSEED IS ${seed}\n"
    eval $comm
    printf "DONE\n\n"
done
