import sys
import os
current_file_path = os.path.abspath(__file__)
parent_dir = os.path.dirname(current_file_path)
sys.path.append(os.path.join(parent_dir, '../../G3FlowDP/3D-Diffusion-Policy'))
sys.path.append(os.path.join(parent_dir))
sys.path.append(os.path.join(parent_dir, '..'))
sys.path.append(os.path.join(parent_dir, '../../tools/'))
sys.path.append(os.path.join(parent_dir, '../../G3FlowDP'))
sys.path.append(os.path.join(parent_dir, '../../tools/Grounded-Segment-Anything'))
from dino import DINO
from Detect_and_Seg import GroundedSAM 
import traceback

import torch  
import sapien.core as sapien
import numpy as np
from envs import *
import hydra
import pathlib
import logging
from G3Flow_policy import *
import yaml
from datetime import datetime
import importlib
import warnings
from imagineModel import G3FlowVirtualSpace
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)


def class_decorator(task_name):
    envs_module = importlib.import_module(f'envs.{task_name}')
    try:
        env_class = getattr(envs_module, task_name)
        env_instance = env_class()
    except:
        raise SystemExit("No Task")
    return env_instance

TASK = None

@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.joinpath(
        os.path.join('../../G3FlowDP/dp/diffusion_policy_3d'), 'config'))
)
def main(cfg):
    test_scene = G3FlowVirtualSpace()
    print('render ok !')
    global TASK
    cfg.feature_dim = cfg.n_components+3
    cfg.task.shape_meta.obs.feature_point_cloud.shape = [cfg.sample_num, cfg.feature_dim]
    TASK = cfg.task.name
    seed = cfg.training.seed
    
    print('Task name:', TASK)
    checkpoint_num = cfg.checkpoint_num
    expert_dat_num = cfg.expert_data_num
    with open(f'./task_config/{cfg.raw_task_name}.yml', 'r', encoding='utf-8') as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)
    task = class_decorator(args['task_name'])
    if checkpoint_num == -1:
        st_seed = 100000 * (seed+1)
        suc_nums = []
        test_num = 20
        topk = 5
        for i in range(10):
            checkpoint_num = (i+1) * 300
            print(f'====================== checkpoint {checkpoint_num} ======================')
            feature_dp = G3FlowDP(cfg, checkpoint_num)
            st_seed, suc_num = test_policy(task, args, feature_dp, st_seed, test_num=test_num, cfg=cfg)
            suc_nums.append(suc_num)
    else:
        st_seed = 100000 * (seed+1)
        suc_nums = []
        test_num = 100
        topk = 1
        feature_dp = G3FlowDP(cfg, checkpoint_num)
        st_seed, suc_num = test_policy(task, args, feature_dp, st_seed, test_num=test_num, cfg=cfg)
        suc_nums.append(suc_num)
    topk_success_rate = sorted(suc_nums, reverse=True)[:topk]
    save_dir  = f'result_feature_dp/{cfg.raw_task_name}_{cfg.expert_data_num}_{cfg.sample_num}_{cfg.n_components}_{cfg.feature_type}_{cfg.training.seed}'
    file_path = os.path.join(save_dir, f'result_{checkpoint_num}_ckpt.txt')
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if not os.path.exists(os.path.dirname(file_path)):
        os.makedirs(os.path.dirname(file_path))
    with open(file_path, 'w') as file:
        file.write(f'Timestamp: {current_time}\n\n')
        file.write(f'Checkpoint Num: {checkpoint_num}\n')
        
        file.write('Successful Rate of Diffenent checkpoints:\n')
        file.write('\n'.join(map(str, np.array(suc_nums) / test_num)))
        file.write('\n\n')
        file.write(f'TopK {topk} Success Rate (every):\n')
        file.write('\n'.join(map(str, np.array(topk_success_rate) / test_num)))
        file.write('\n\n')
        file.write(f'TopK {topk} Success Rate:\n')
        file.write(f'\n'.join(map(str, np.array(topk_success_rate) / (topk * test_num))))
        file.write('\n\n')
    print(f'Data has been saved to {file_path}')
    
def test_policy(Demo_class, args, feature_dp, st_seed, test_num=20, cfg=None):
    global TASK
    epid = 0      
    seed_list=[]  
    suc_num = 0   
    expert_check = True
    print("Task name: ",args["task_name"])
    Demo_class.suc = 0
    Demo_class.test_num =0
    now_id = 0
    succ_seed = 0
    suc_test_seed_list = []
    
    now_seed = st_seed
    tracker = None
    dino = DINO(dino_name="dinov2", model_name='vits14').to('cuda')
    detect_and_seg = GroundedSAM(sam_checkpoint=os.path.join(parent_dir, '../../tools/weights_for_g3flow/sam_vit_h_4b8939.pth'))
    while succ_seed < test_num:
        render_freq = args['render_freq']
        args['render_freq'] = 0
           
        if expert_check:
            try:
                Demo_class.setup_demo(now_ep_num=now_id, seed = now_seed, is_test = True, ** args)
                
                info = Demo_class.play_once()
                Demo_class.close()
            except Exception as e:
                stack_trace = traceback.format_exc()
                print(' -------------')
                print('Error: ', stack_trace)
                print(' -------------')
                Demo_class.close()
                now_seed += 1
                args['render_freq'] = render_freq
                print('error occurs !')
                continue

        if (not expert_check) or ( Demo_class.plan_success and Demo_class.check_success() ):
            succ_seed +=1
            suc_test_seed_list.append(now_seed)
            
        else:
            now_seed += 1
            args['render_freq'] = render_freq
            import time
            time.sleep(1)
            continue
        
        args['render_freq'] = render_freq
        Demo_class.setup_demo(now_ep_num=now_id, seed = now_seed, is_test = True, ** args)
        
        tracker = Demo_class.apply_g3flow_dp(feature_dp, info=info, tracker=tracker, dino=dino, detect_and_seg=detect_and_seg, cfg=cfg, sample_num=cfg.sample_num)

        now_id += 1
        Demo_class.close()
        if Demo_class.render_freq:
            Demo_class.viewer.close()
        feature_dp.env_runner.reset_obs()
        print(f"{TASK} success rate: {Demo_class.suc}/{Demo_class.test_num} = {Demo_class.suc / Demo_class.test_num * 100} %, current seed: {now_seed}\n")
        Demo_class._take_picture()
        now_seed += 1
    return now_seed, Demo_class.suc

if __name__ == "__main__":
    main()
