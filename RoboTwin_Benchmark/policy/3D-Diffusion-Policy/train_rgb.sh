#!/bin/bash
# bash train.sh mug_hanging 10 0 0

task_name=${1}
expert_data_num=${2}
seed=${3}
gpu_id=${4}

echo "./data/${task_name}_${expert_data_num}.zarr" 
if [ ! -d "./data/${task_name}_${expert_data_num}.zarr" ]; then
    echo "zarr does not exist, run pkl2zarr_dp3.py"
    cd ../..
    expert_data_num_minus_one=$((expert_data_num - 1))
    if [ ! -d "./data/${task_name}_pkl/episode${expert_data_num_minus_one}" ]; then
        echo "error: expert data does not exist"
        exit 1
    else
        python script/pkl2zarr_dp3.py ${task_name} ${expert_data_num}
        cd policy/3D-Diffusion-Policy
    fi
fi

bash scripts/train_policy_rgb.sh robot_dp3 ${task_name}_${expert_data_num} train ${seed} ${gpu_id}

