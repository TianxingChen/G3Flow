import os, yaml

names = [
    "tool_adjust_T", 
    "tool_adjust_G", 
    "bottle_adjust_T", 
    "diverse_bottles_pick_G",
    "shoe_place_T",
    "shoe_place_G",
    "shoes_place_T",
    "shoes_place_G"
]
for task_name in names:
    task_config_path = f'../task_config/{task_name}.yml'
    if True:
        data = {
            'task_name': task_name,
            'render_freq': 0,
            'use_seed': False,
            'collect_data': True,
            'save_path': './data',
            'dual_arm': True,
            'st_episode': 0 ,
            'camera_w': 320,
            'camera_h': 240,
            'pcd_crop': True ,
            'pcd_down_sample_num': 1024,
            'episode_num': 100,
            'save_type':{
                'raw_data': False,
                'pkl': True
            },
            'data_type':{
                'rgb': True,
                'observer': True,
                'depth': True,
                'pointcloud': True,
                'conbine': False,
                'endpose': True,
                'qpos': True,
                'mesh_segmentation': False,
                'actor_segmentation': False,
            }
        }
        with open(task_config_path, 'w') as f:
            yaml.dump(data,f,default_flow_style = False,sort_keys=False)