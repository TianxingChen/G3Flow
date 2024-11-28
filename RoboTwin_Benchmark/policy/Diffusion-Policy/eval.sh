DEBUG=False

task_name=${1}
expert_data_num=${2}
checkpoint_num=${3}
seed=${4}
gpu_id=${5}

export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES=${gpu_id}

cd ../..
python ./script/eval_policy_dp.py "$task_name" "$checkpoint_num" "$expert_data_num" ${seed}