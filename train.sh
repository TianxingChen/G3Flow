task_name=${1}
expert_data_num=${2}
n_components=${3}
feature_type="G3Flow"
seed=${4}
gpu_id=${5}
sample_num=1024

cd G3FlowDP

if [ ! -d "./data/zarr_data/${task_name}_${expert_data_num}_${sample_num}_${n_components}_${feature_type}.zarr" ]; then
    echo "zarr does not exist, run pkl2zarr"
    expert_data_num_minus_one=$((expert_data_num - 1))
    if [ ! -d "../RoboTwin_Benchmark/data/${task_name}_pkl/episode${expert_data_num_minus_one}" ]; then
        echo "error: expert data does not exist"
        exit 1
    else
        python scripts/get_G3Flow_dataset.py ${task_name} ${expert_data_num} ${n_components} ${sample_num}
        python scripts/pkl2zarr_G3FlowDP.py ${task_name} ${expert_data_num} ${n_components} ${sample_num} ablation
    fi
fi

bash scripts/train_policy.sh G3FlowDP ${task_name} ${expert_data_num} ${n_components} ${feature_type} train ${seed} ${gpu_id}
