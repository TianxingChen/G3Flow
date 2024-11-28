
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
import numpy as np
from function import *
import pdb
import open3d as o3d
import json
import pickle
import torch
import argparse
import transforms3d as t3d

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('task_name', type=str)
    parser.add_argument('episode_number', type=int)
    parser.add_argument('n_components', type=int)
    parser.add_argument('num_points', type=int)

    args = parser.parse_args()    
    episode_num = args.episode_number
    task_name = args.task_name
    n_components = args.n_components
    feature_type = 'ablation'

    camera = 'head_camera'
    device = "cuda"

    pcd_crop_bbox = [[-0.6, -0.35, 0.7413],[0.6, 0.35, 2]]
    
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

    # ================ Get PCA ================
    if not PCA_ed:
        assert feature_type == 'ablation'
        total_feature = None

        for episode_idx in range(episode_num):
            print(f'processing PCA: {episode_idx+1} / {episode_num}', end='\r')
            folder_path = os.path.join(parent_dir, f'../../RoboTwin_Benchmark/data/{task_name}_pkl/episode{episode_idx}')

            with open(folder_path + f'/0.pkl', 'rb') as file:
                data = pickle.load(file)

            camera_intrinsic, cam2world_matrix = data['observation'][camera]['intrinsic_cv'], data['observation']['head_camera']['cam2world_gl']
            color = data['observation'][camera]['rgb'][..., :3]
            depth = data['observation'][camera]['depth'][...] / 1000
            pcd_tmp, pcd_rgb_tmp = tanslation_point_cloud(depth, color, camera_intrinsic, cam2world_matrix, mask=None)
            
            feature_map, feature_line = get_dino_feature(color, transform_size=420, model=dino)
            min_bound = torch.tensor(pcd_crop_bbox[0], dtype=torch.float32).to(device)
            max_bound = torch.tensor(pcd_crop_bbox[1], dtype=torch.float32).to(device)

            pcd_tensor = torch.Tensor(pcd_tmp).to(device)
            inside_bounds_mask = (pcd_tensor >= min_bound).all(dim=1) & (pcd_tensor <= max_bound).all(dim=1)
            inside_bounds_mask = inside_bounds_mask.to('cpu')
            pcd_tmp = pcd_tmp[inside_bounds_mask]
            pcd_rgb_tmp = pcd_rgb_tmp[inside_bounds_mask]
            feature_line = feature_line[inside_bounds_mask]
            if total_feature is None:
                total_feature = feature_line
            else:
                total_feature = torch.cat((total_feature, feature_line), dim=0)

        total_pca = FeaturePCA(n_components=n_components)
        total_pca.fit(total_feature)
        total_pca.save_pca(PCA_model_path)
    

    # ================ Process Data ================

    PCA_model_path = PCA_model_path 
    feature_pca = FeaturePCA(n_components)
    feature_pca.load_pca(PCA_model_path)
    for episode_idx in range(0, episode_num):
        print(f'getting feature: episode {episode_idx}')
        delete_folder(os.path.join(save_dir, f'./episode{episode_idx}'))
        os.makedirs(os.path.join(save_dir, f'./episode{episode_idx}'), exist_ok=True)
        pose_dict = dict()

        folder_path = os.path.join(parent_dir, f'../../RoboTwin_Benchmark/data/{task_name}_pkl/episode{episode_idx}')

        idx = 0
        while os.path.exists(folder_path + f'/{idx}.pkl'):
            with open(folder_path + f'/{idx}.pkl', 'rb') as file:
                data = pickle.load(file)

            color = data['observation'][camera]['rgb'][..., :3]
            depth = data['observation'][camera]['depth'][...] / 1000

            camera_intrinsic, cam2world_matrix = data['observation'][camera]['intrinsic_cv'], data['observation'][camera]['cam2world_gl']

            pcd_tmp, pcd_rgb_tmp = tanslation_point_cloud(depth, color, camera_intrinsic, cam2world_matrix, mask=None)
            
            feature_map, feature_line = get_dino_feature(color, transform_size=420, model=dino)
            min_bound = torch.tensor(pcd_crop_bbox[0], dtype=torch.float32).to(device)
            max_bound = torch.tensor(pcd_crop_bbox[1], dtype=torch.float32).to(device)

            pcd_tensor = torch.Tensor(pcd_tmp).to(device)


            inside_bounds_mask = (pcd_tensor >= min_bound).all(dim=1) & (pcd_tensor <= max_bound).all(dim=1)
            inside_bounds_mask = inside_bounds_mask.to('cpu')
            pcd_tmp = pcd_tmp[inside_bounds_mask]
            pcd_rgb_tmp = pcd_rgb_tmp[inside_bounds_mask]
            raw_feature_line = feature_line
            feature_line = feature_line[inside_bounds_mask]
            feature_line = feature_pca.transform(feature_line.cpu())
            feature_pcd = np.concatenate((pcd_tmp, feature_line), axis=1)

            data.pop('observation')
            _, indices = fps(pcd_tmp, num_points=sample_num, use_cuda=True)
            data['feature_point_cloud'] = feature_pcd[indices.to('cpu')].squeeze(0)

            with open(os.path.join(save_dir, f'episode{episode_idx}/{idx}.pkl'), 'wb') as file:
                pickle.dump(data, file)
            if idx == 0:
                o3d_pcd = o3d.geometry.PointCloud()
                o3d_pcd.points = o3d.utility.Vector3dVector(feature_pcd[..., :3])
                o3d_pcd.colors = o3d.utility.Vector3dVector(feature_pcd[..., 3:6]) 
                output_filename = os.path.join(save_dir, f'episode{episode_idx}/_orig.pcd')
                o3d.io.write_point_cloud(output_filename, o3d_pcd)
                print('pcd is saved to: ', output_filename)
                save_path = os.path.join(save_dir, f'./episode{episode_idx}')
                feature_line_pcad_rgb = feature_to_rgb(raw_feature_line[:, :3], color.shape[0], color.shape[1])
                save_image(np.array(feature_line_pcad_rgb*255, dtype=np.uint8), save_path+f'/_pcad.png', mask=None, mask_color=255)
                save_image(color, save_path+f'/_raw_scene.png')

            idx += 1

        


