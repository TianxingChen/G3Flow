import pdb, pickle, os, sys
import numpy as np
import open3d as o3d
from copy import deepcopy
import zarr, shutil
import argparse
current_file_path = os.path.abspath(__file__)
parent_dir = os.path.dirname(current_file_path)

def main():
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

    sample_num = args.num_points
    
    visualize_pcd = False

    task_name = args.task_name
    num = args.episode_number
    current_ep, num = 0, num

    load_dir = os.path.join(parent_dir, f'../data/{task_name}_{episode_num}_{sample_num}_{n_components}_{feature_type}')
    save_dir = os.path.join(parent_dir, f'../data/zarr_data/{task_name}_{episode_num}_{sample_num}_{n_components}_{feature_type}.zarr')
    
    total_count = 0


    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)

    zarr_root = zarr.group(save_dir)
    zarr_data = zarr_root.create_group('data')
    zarr_meta = zarr_root.create_group('meta')

    point_cloud_arrays, episode_ends_arrays, action_arrays, state_arrays, joint_action_arrays = [], [], [], [], []
    feature_point_cloud_arrays = []

    while os.path.isdir(load_dir+f'/episode{current_ep}') and current_ep < num:
        print(f'processing episode: {current_ep + 1} / {num}', end='\r')
        file_num = 0
        point_cloud_sub_arrays = []
        state_sub_arrays = []
        action_sub_arrays = [] 
        joint_action_sub_arrays = []
        episode_ends_sub_arrays = []
        feature_point_cloud_sub_arrays = []
        
        while os.path.exists(load_dir+f'/episode{current_ep}'+f'/{file_num}.pkl'):
            with open(load_dir+f'/episode{current_ep}'+f'/{file_num}.pkl', 'rb') as file:
                data = pickle.load(file)
            
            pcd = data['pointcloud'][:,:]
            action = data['endpose']
            feature_point_cloud = data['feature_point_cloud'][:,:]

            joint_action = data['joint_action']

            point_cloud_sub_arrays.append(pcd)
            feature_point_cloud_sub_arrays.append(feature_point_cloud)
            state_sub_arrays.append(joint_action)
            action_sub_arrays.append(action)
            joint_action_sub_arrays.append(joint_action)

            if visualize_pcd:
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(data['pcd']['points'])
                pcd.colors = o3d.utility.Vector3dVector(data['pcd']['colors'])
                o3d.visualization.draw_geometries([pcd])

            file_num += 1
            total_count += 1
            
        current_ep += 1

        episode_ends_arrays.append(deepcopy(total_count))
        point_cloud_arrays.extend(point_cloud_sub_arrays)
        action_arrays.extend(action_sub_arrays)
        state_arrays.extend(state_sub_arrays)
        joint_action_arrays.extend(joint_action_sub_arrays)
        feature_point_cloud_arrays.extend(feature_point_cloud_sub_arrays)

    print()
    episode_ends_arrays = np.array(episode_ends_arrays)
    action_arrays = np.array(action_arrays)
    state_arrays = np.array(state_arrays)
    point_cloud_arrays = np.array(point_cloud_arrays)
    joint_action_arrays = np.array(joint_action_arrays)
    feature_point_cloud_arrays = np.array(feature_point_cloud_arrays)

    compressor = zarr.Blosc(cname='zstd', clevel=3, shuffle=1)
    action_chunk_size = (100, action_arrays.shape[1])
    state_chunk_size = (100, state_arrays.shape[1])
    joint_chunk_size = (100, joint_action_arrays.shape[1])
    point_cloud_chunk_size = (100, point_cloud_arrays.shape[1], point_cloud_arrays.shape[2])
    feature_point_cloud_chunk_size = (100, feature_point_cloud_arrays.shape[1], feature_point_cloud_arrays.shape[2])


    zarr_data.create_dataset('point_cloud', data=point_cloud_arrays, chunks=point_cloud_chunk_size, dtype='float32', overwrite=True, compressor=compressor)
    zarr_data.create_dataset('feature_point_cloud', data=feature_point_cloud_arrays, chunks=feature_point_cloud_chunk_size, dtype='float32', overwrite=True, compressor=compressor)
    zarr_data.create_dataset('state', data=state_arrays, chunks=state_chunk_size, dtype='float32', overwrite=True, compressor=compressor)
    zarr_data.create_dataset('action', data=joint_action_arrays, chunks=joint_chunk_size, dtype='float32', overwrite=True, compressor=compressor)
    zarr_meta.create_dataset('episode_ends', data=episode_ends_arrays, dtype='int64', overwrite=True, compressor=compressor)

if __name__ == '__main__':
    main()
