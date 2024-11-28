# use the same command as training except the script
# for example:
# bash script/run_eval_policy_dp3.sh robot_dp3 $task_name $expert_data_num eval $seed $checkpoint_num $gpu_id
DEBUG=False
alg_name=${1}
task_name=${2}
expert_data_num=${3}
n_components=${4}
feature_type=${5}
config_name=${alg_name}
addition_info=${6}
seed=${7}
checkpoint_num=${8}
gpu_id=${9}
sample_num=${10}
exp_name=${task_name}-${alg_name}-${addition_info}
run_dir="./policy/3D-Diffusion-Policy/3D-Diffusion-Policy/diffusion_policy_3d/data/outputs/${exp_name}_seed${seed}"

export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES=${gpu_id}

python script/eval_policy_G3FlowDP_ablation.py --config-name=${config_name}.yaml \
                            task=${task_name} \
                            raw_task_name=${task_name} \
                            hydra.run.dir=${run_dir} \
                            training.debug=$DEBUG \
                            training.seed=${seed} \
                            training.device="cuda:0" \
                            exp_name=${exp_name} \
                            logging.mode=${wandb_mode} \
                            checkpoint_num=${checkpoint_num} \
                            expert_data_num=${expert_data_num} \
                            n_components=${n_components} \
                            sample_num=${sample_num} \
                            feature_type=${feature_type}