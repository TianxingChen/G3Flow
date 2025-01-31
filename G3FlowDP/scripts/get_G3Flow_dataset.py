
import sys
import warnings
import os
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
current_file_path = os.path.abspath(__file__)
parent_dir = os.path.dirname(current_file_path)

sys.path.append(os.path.join(parent_dir, '../../tools'))
sys.path.append(os.path.join(parent_dir, '../../tools/Grounded-Segment-Anything'))
sys.path.append(os.path.join(parent_dir, '../'))
from feature_pca import FeaturePCA
from Detect_and_Seg import *
from dino import DINO
from featureTracking import G3FlowDPTracker
from imagineModel import G3FlowVirtualSpace
from Detect_and_Seg import GroundedSAM
import numpy as np
from function import *
import pdb
import open3d as o3d
import json
import pickle
import torch
import argparse
import sapien.core as sapien
import transforms3d as t3d

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('task_name', type=str)
    parser.add_argument('episode_number', type=int)
    parser.add_argument('n_components', type=int)
    parser.add_argument('num_points', type=int)
    parser.add_argument('feature_type', type=str)

    args = parser.parse_args()    
    episode_num = args.episode_number
    task_name = args.task_name
    n_components = args.n_components
    feature_type = args.feature_type

    camera = 'head_camera'
    device = "cuda"

    scene = G3FlowVirtualSpace()
    print('render ok !')

    pcd_crop_bbox = [[-0.6, -0.35, 0.7401],[0.6, 0.35, 2]]
    
    sample_num = args.num_points
    
    PCA_ed = False
    save_dir = os.path.join(parent_dir, f'../data/{task_name}_{episode_num}_{sample_num}_{n_components}_{feature_type}')
    PCA_folder = os.path.join(parent_dir, f'../PCA_Model/{task_name}_{episode_num}_{sample_num}_{n_components}_{feature_type}')
    PCA_model_path = PCA_folder + f'/pca_{n_components}.pkl'

    delete_folder(save_dir)
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(PCA_folder, exist_ok=True)

    scene_info_path = f'../RoboTwin_Benchmark/data/{task_name}_pkl/scene_info.json' # raw data
    with open(scene_info_path, 'r', encoding='utf-8') as file:
        scene_info = json.load(file)

    dino = DINO(dino_name="dinov2", model_name='vits14').to(device)
    detect_and_seg = GroundedSAM(sam_checkpoint=os.path.join(parent_dir, '../../tools/weights_for_g3flow/sam_vit_h_4b8939.pth'))

    # ================ Get PCA MODEL ================
    if not PCA_ed:
        total_feature = None
        for episode_idx in range(episode_num):
            print(f'processing PCA: {episode_idx+1} / {episode_num}', end='\r')
            folder_path = os.path.join(parent_dir, f'../../RoboTwin_Benchmark/data/{task_name}_pkl/episode{episode_idx}')

            with open(folder_path + f'/0.pkl', 'rb') as file:
                data = pickle.load(file)

            text_prompt = str(scene_info[f'{episode_idx}']['object_description'][0])
            topk = int(scene_info[f'{episode_idx}']['object_description'][1])
            object_id = scene_info[f'{episode_idx}']['object_id'] # list
            assert sample_num % topk == 0, 'error: sample_num % topk != 0'

            camera_intrinsic, cam2world_matrix = data['observation'][camera]['intrinsic_cv'], data['observation'][camera]['cam2world_gl']
            color = data['observation'][camera]['rgb'][..., :3]
            depth = data['observation'][camera]['depth'][...] / 1000
            masks = detect_and_seg.detect_and_seg(color, text_prompt, topk=topk)
            masks = sort_masks(masks)

            # load model
            glb_files_paths, glb_model_data_paths, obj_files_paths, obj_model_data_paths = [], [], [], []
            for i in range(topk):
                glb_file_path = os.path.join(parent_dir, f'../../RoboTwin_Benchmark/models/' + str(scene_info[f'{episode_idx}']['object_type'][0]) + '/base' + str(scene_info[f'{episode_idx}']['object_id'][i]) + '.glb')
                glb_model_data_path = os.path.join(parent_dir, f'../../RoboTwin_Benchmark/models/' + str(scene_info[f'{episode_idx}']['object_type'][0]) + '/model_data' + str(scene_info[f'{episode_idx}']['object_id'][i]) + '.json')
                obj_file_path = os.path.join(parent_dir, f'../../RoboTwin_Benchmark/models/' + str(scene_info[f'{episode_idx}']['object_type'][0]) + '/obj/textured' + str(scene_info[f'{episode_idx}']['object_id'][i]) + '.obj')
                obj_model_data_path = os.path.join(parent_dir, f'../../RoboTwin_Benchmark/models/' + str(scene_info[f'{episode_idx}']['object_type'][0]) + '/obj/model_data' + str(scene_info[f'{episode_idx}']['object_id'][i]) + '.json')
                
                glb_files_paths.append(glb_file_path)
                glb_model_data_paths.append(glb_model_data_path)
                obj_files_paths.append(obj_file_path)
                obj_model_data_paths.append(obj_model_data_path)
            
            tracker = G3FlowDPTracker(cam2world_matrix, camera_intrinsic, n_components=n_components, mesh_file_paths=obj_files_paths, model_data_paths=obj_model_data_paths, topk=topk, detect_and_seg=detect_and_seg, dino=dino) 

            for i in range(masks.shape[0]):
                mask = np.array(masks[i][0].cpu())
                pose = tracker.pose_trackers[i].get_pose_first_frame(color, depth, mask)
                real_obj_world_matrix = cam2world_matrix @ np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0,0,0,1]]) @ pose
                real_obj_world_pose = sapien.Pose(real_obj_world_matrix[:3,3], t3d.quaternions.mat2quat(real_obj_world_matrix[:3,:3]))

                glb_file_path = glb_files_paths[i]
                glb_model_data_path = glb_model_data_paths[i]
                scene = G3FlowVirtualSpace()
                scene.load_actor(glb_file_path, glb_model_data_path, real_obj_world_pose)
                scene.load_camera(real_obj_world_pose.p, real_obj_world_pose.q)
                observation = scene.get_obs()

                camera_key_list = list(observation.keys())
                for j in range(len(camera_key_list)):
                    sub_camera = observation[camera_key_list[j]]
                    obj_color, obj_depth, obj_mask = sub_camera['rgb'], sub_camera['depth'] / 1000, sub_camera['mask']
                    bounding_box_mask = get_bounding_box_mask(obj_mask) # hole image level
                    region_color, _, region_mask = extract_bounding_box_region(obj_color, None, bounding_box_mask, obj_mask) 
                    feature_map, feature_line = get_dino_feature(region_color, transform_size=420, model=dino)
                    if total_feature is None:
                        total_feature = feature_line[region_mask.reshape(-1)]
                    else:
                        total_feature = torch.cat((total_feature, feature_line[region_mask.reshape(-1)]), dim=0)
            

            camera_intrinsic, cam2world_matrix = data['observation'][camera]['intrinsic_cv'], data['observation'][camera]['cam2world_gl']

        total_pca = FeaturePCA(n_components=n_components)
        total_pca.fit(total_feature.cpu())
        total_pca.save_pca(PCA_model_path)


    # ================ Process Data ================

   
    tracker = None
    for episode_idx in range(0, episode_num):
        print(f'getting feature: episode {episode_idx}')
        delete_folder(os.path.join(save_dir, f'./episode{episode_idx}'))
        os.makedirs(os.path.join(save_dir, f'./episode{episode_idx}'), exist_ok=True)
        pose_dict = dict()
        first_pose = None

        folder_path = os.path.join(parent_dir, f'../../RoboTwin_Benchmark/data/{task_name}_pkl/episode{episode_idx}')

        idx = 0
        while os.path.exists(folder_path + f'/{idx}.pkl'):
            with open(folder_path + f'/{idx}.pkl', 'rb') as file:
                data = pickle.load(file)

            color = data['observation'][camera]['rgb'][..., :3]
            depth = data['observation'][camera]['depth'][...] / 1000

            camera_intrinsic, cam2world_matrix = data['observation'][camera]['intrinsic_cv'], data['observation'][camera]['cam2world_gl']

            if idx == 0:
                text_prompt = str(scene_info[f'{episode_idx}']['object_description'][0])
                topk = int(scene_info[f'{episode_idx}']['object_description'][1])
                assert sample_num % topk == 0, 'error: sample_num % topk != 0'

                PCA_model_path = PCA_model_path 
                text_prompt = str(scene_info[f'{episode_idx}']['object_description'][0])
                
                # load model
                glb_files_paths, glb_model_data_paths, obj_files_paths, obj_model_data_paths = [], [], [], []
                for i in range(topk):
                    glb_file_path = os.path.join(parent_dir, f'../../RoboTwin_Benchmark/models/' + str(scene_info[f'{episode_idx}']['object_type'][0]) + '/base' + str(scene_info[f'{episode_idx}']['object_id'][i]) + '.glb')
                    glb_model_data_path = os.path.join(parent_dir, f'../../RoboTwin_Benchmark/models/' + str(scene_info[f'{episode_idx}']['object_type'][0]) + '/model_data' + str(scene_info[f'{episode_idx}']['object_id'][i]) + '.json')
                    obj_file_path = os.path.join(parent_dir, f'../../RoboTwin_Benchmark/models/' + str(scene_info[f'{episode_idx}']['object_type'][0]) + '/obj/textured' + str(scene_info[f'{episode_idx}']['object_id'][i]) + '.obj')
                    obj_model_data_path = os.path.join(parent_dir, f'../../RoboTwin_Benchmark/models/' + str(scene_info[f'{episode_idx}']['object_type'][0]) + '/obj/model_data' + str(scene_info[f'{episode_idx}']['object_id'][i]) + '.json')
                    
                    glb_files_paths.append(glb_file_path)
                    glb_model_data_paths.append(glb_model_data_path)
                    obj_files_paths.append(obj_file_path)
                    obj_model_data_paths.append(obj_model_data_path)
                
                tracker = G3FlowDPTracker(cam2world_matrix, camera_intrinsic, n_components=n_components, mesh_file_paths=obj_files_paths, model_data_paths=obj_model_data_paths, topk=topk, detect_and_seg=detect_and_seg, dino=dino) 
                tracker.load_pca(PCA_model_path)
  
                feature_pcd = tracker.get_first_frame(data, text_prompt, camera=camera, save_path=os.path.join(save_dir, f'episode{episode_idx}'), sample_num=sample_num, feature_type=feature_type, additional_obj_file_info=[glb_files_paths, glb_model_data_paths])
                poses = tracker.get_pose(color, depth)

                o3d_pcd = o3d.geometry.PointCloud()
                o3d_pcd.points = o3d.utility.Vector3dVector(feature_pcd[..., :3])
                o3d_pcd.colors = o3d.utility.Vector3dVector(feature_pcd[..., 3:6]) 
                output_filename = os.path.join(save_dir, f'episode{episode_idx}/_orig.pcd')
                o3d.io.write_point_cloud(output_filename, o3d_pcd)
                print('pcd is saved to: ', output_filename)
                
            else:
                poses = tracker.get_pose(color, depth)
                feature_pcd = tracker.get_feature_pcd(poses)
                    
            data.pop('observation')
            data['feature_point_cloud'] = feature_pcd

            with open(os.path.join(save_dir, f'episode{episode_idx}/{idx}.pkl'), 'wb') as file:
                pickle.dump(data, file)

            idx += 1
        


