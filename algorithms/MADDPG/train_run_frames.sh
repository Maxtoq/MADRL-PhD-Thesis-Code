#!/bin/sh
n_run=1
env="coop_push_scenario/coop_push_scenario_sparse.py"
model_name="2addpg_fo_rel_disc_frames"
sce_conf_path="configs/2a_1o_fo_rel.json"
n_frames=10000000
buffer_length=1000000
lr=0.0007
gamma=0.99
tau=0.01
init_exploration=1.0
cuda_device="cuda:0"

for n in $(seq 1 $n_run)
do
    printf "Run ${n}/${n_run}\n"
    seed=$RANDOM
    comm="python algorithms/MADDPG/train_frames.py ${env} ${model_name} --sce_conf_path ${sce_conf_path} \
--seed ${seed} --n_frames ${n_frames} --lr ${lr} --cuda_device ${cuda_device} --gamma ${gamma} \
--tau ${tau} --init_exploration ${init_exploration} --buffer_length ${buffer_length} \
--discrete_action"
    printf "Starting training with command:\n${comm}\n\nSEED IS ${seed}\n"
    eval $comm
    printf "DONE\n\n"
done
