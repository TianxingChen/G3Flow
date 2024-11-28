#!/bin/bash

task_name=${1}
gpu_id=${2}

export CUDA_VISIBLE_DEVICES=${gpu_id}
echo ${task_name} | python script/run_task.py