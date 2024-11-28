#!/bin/bash
# bash train_and_eval.sh mug_hanging 10 3000 0 0

task_name=${1}
expert_data_num=${2}
checkpoint_num=${3}
seed=${4}
gpu_id=${5}

bash train.sh ${task_name} ${expert_data_num} ${seed} ${gpu_id}
bash eval.sh ${task_name} ${expert_data_num} ${checkpoint_num} ${seed} ${gpu_id}

