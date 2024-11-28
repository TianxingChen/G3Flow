# bash eval.sh mug_hanging 10 3000 0 0

task_name=${1}
expert_data_num=${2}
n_components=${3}
feature_type="G3Flow"
checkpoint_num=3000
seed=${4}
gpu_id=${5}
sample_num=1024

cd ./RoboTwin_Benchmark
bash script/run_eval_policy_G3FlowDP.sh G3FlowDP $task_name $expert_data_num $n_components $feature_type eval $seed $checkpoint_num $gpu_id $sample_num
