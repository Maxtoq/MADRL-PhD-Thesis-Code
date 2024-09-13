#!/bin/bash


for n in $(seq 1 5)
do
	c="rm -r models/magym_PredPrey_new/9o5_Diff_ec2/run${n}"
	m="mv models/magym_PredPrey_new/9o5_ec2/run${n} models/magym_PredPrey_new/9o5_Diff_ec2/run${n}"
	echo $c
	eval $c
	echo $m
	eval $m
done
