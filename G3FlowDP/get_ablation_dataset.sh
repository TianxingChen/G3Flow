task_name=${1}
expert_data_num=${2}
n_components=${3}
gpu_id=${4}
sample_num=1024

TOKENIZERS_PARALLELISM=false

export CUDA_VISIBLE_DEVICES=${gpu_id}

python scripts/get_G3Flow_ablation_dataset.py ${task_name} ${expert_data_num} ${n_components} ${sample_num}