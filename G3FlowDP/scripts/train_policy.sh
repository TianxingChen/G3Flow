# Examples:
# bash scripts/train_policy.sh dp3 adroit_hammer_pointcloud 0112 0 0
# bash scripts/train_policy.sh simple_dp3 adroit_hammer_pointcloud 0112 0 0



DEBUG=False
save_ckpt=True

alg_name=${1}
task_name=${2}
expert_data_num=${3}
n_components=${4}
config_name=${alg_name}
feature_type=${5}
addition_info=${6}
seed=${7}
gpu_id=${8}

exp_name=${task_name}-${alg_name}-${addition_info}
run_dir="data/debug_dir/outputs/${exp_name}_seed${seed}"

# gpu_id=$(bash scripts/find_gpu.sh)
echo -e "\033[33mgpu id (to use): ${gpu_id}\033[0m"


if [ $DEBUG = True ]; then
    wandb_mode=offline
    # wandb_mode=online
    echo -e "\033[33mDebug mode!\033[0m"
    echo -e "\033[33mDebug mode!\033[0m"
    echo -e "\033[33mDebug mode!\033[0m"
else
    wandb_mode=online
    echo -e "\033[33mTrain mode\033[0m"
fi

cd dp


export HYDRA_FULL_ERROR=1 
export CUDA_VISIBLE_DEVICES=${gpu_id}
python train.py --config-name=${config_name}.yaml \
                            task=${task_name} \
                            hydra.run.dir=${run_dir} \
                            training.debug=$DEBUG \
                            training.seed=${seed} \
                            training.device="cuda:0" \
                            exp_name=${exp_name} \
                            logging.mode=${wandb_mode} \
                            checkpoint.save_ckpt=${save_ckpt} \
                            n_components=${n_components} \
                            expert_data_num=${expert_data_num} \
                            sample_num=1024 \
                            feature_type=${feature_type}

