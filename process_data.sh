task_name=${1}
expert_data_num=${2}
n_components=${3}
gpu_id=${4}
feature_type="G3Flow"
sample_num=1024

TOKENIZERS_PARALLELISM=false

export CUDA_VISIBLE_DEVICES=${gpu_id}

cd G3FlowDP
python scripts/get_G3Flow_dataset.py ${task_name} ${expert_data_num} ${n_components} ${sample_num} ${feature_type}
python scripts/pkl2zarr_G3FlowDP.py ${task_name} ${expert_data_num} ${n_components} ${sample_num} ${feature_type}