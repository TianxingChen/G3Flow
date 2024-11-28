import os
import pdb
import sys
current_file_path = os.path.abspath(__file__)
parent_dir = os.path.dirname(current_file_path)
sys.path.append(parent_dir)
sys.path.append(os.path.join(os.path.dirname(parent_dir), './tools/FoundationPose'))
sys.path.append(os.path.join(os.path.dirname(parent_dir), './tools/Grounded-Segment-Anything'))
sys.path.append(os.path.join(os.path.dirname(parent_dir), 'Grounded-Segment-Anything'))
from TrackPose import FoundationPoseTracker
from dino import DINO
import numpy as np
from function import *
from trans_pcd import TransPCD
from feature_pca import FeaturePCA
import pdb
from Detect_and_Seg import *
from FoundationPose.TrackPose import FoundationPoseTracker
from imagineModel import G3FlowVirtualSpace
import sapien.core as sapien
import transforms3d as t3d

class G3FlowDPTracker:
    def __init__(self, cam2world_matrix, intrinsic_cv, n_components, mesh_file_paths, model_data_paths=None, debug=0, device="cuda", topk=1, detect_and_seg=None, dino=None): # camera para, dim of feature
        self.topk = topk
        self.mesh_file_paths = mesh_file_paths
        self.model_data_paths = model_data_paths
        assert len(mesh_file_paths) == topk, 'number of mesh file error'
        self.cam2world_gl = cam2world_matrix @ np.array(np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]))
        
        if detect_and_seg is None:
            self.detect_and_seg = GroundedSAM(sam_checkpoint=os.path.join(os.path.dirname(os.path.abspath(__file__)), '../tools/weights_for_g3flow/sam_vit_h_4b8939.pth'))
        else:
            self.detect_and_seg = detect_and_seg
        if dino is None:
            self.dino = DINO(dino_name="dinov2", model_name='vits14').to(device)
        else:
            self.dino = dino

        debug_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), './data/debug_dir')
        self.feature_pca = FeaturePCA(n_components)
        self.pose_trackers = []
        self.trans_pcds = []
        for i in range(topk):
            self.pose_trackers.append(FoundationPoseTracker(mesh_file_paths[i], K=np.array(intrinsic_cv, dtype=np.float64), debug_dir=debug_dir, debug=0))
            self.trans_pcds.append(TransPCD(self.cam2world_gl))
    
    def reset(self, cam2world_matrix, intrinsic_cv, n_components, mesh_file_paths, debug=0, device="cuda"):
        assert 1 == 2, "reset function is abandoned"
        self.cam2world_gl = cam2world_matrix @ np.array(np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]))
        for i in range(self.topk):
            self.trans_pcds[i] = TransPCD(self.cam2world_gl)
            debug_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), './data/debug_dir')
            self.pose_trackers[i] = FoundationPoseTracker(mesh_file_paths[i], K=np.array(intrinsic_cv, dtype=np.float64), debug_dir=debug_dir, debug=0)
    
    def load_pca(self, file_path): # ok
        assert os.path.isfile(file_path), f"PCA model File missing error: {file_path}"
        print('load pca model from: ', file_path)
        self.feature_pca.load_pca(file_path)
    
    def del_detect_and_seg(self): # ok
        import torch
        del self.detect_and_seg
        torch.cuda.empty_cache()
    
    def get_dino_feature(self, image, transform_size=420): # ok
        return get_dino_feature(image, transform_size=transform_size, model=self.dino) # function.py
    
    def get_first_frame_masks(self, observation, text_prompt, camera='head_camera', topk=1):
        def visualize_mask(mask, save_id=0):
            import matplotlib.pyplot as plt
            import numpy as np
            # Visualize the mask
            plt.imshow(mask, cmap='gray')  # Use 'gray' color map to visualize the mask
            plt.colorbar()  # Add a color bar to indicate the values
            plt.savefig(f'./{save_id}.png')

        camera_intrinsic, cam2world_matrix = observation['observation'][camera]['intrinsic_cv'], observation['observation'][camera]['cam2world_gl']
        color = observation['observation'][camera]['rgb'][..., :3]
        depth = observation['observation'][camera]['depth'][...] / 1000 # !
        masks = self.detect_and_seg.detect_and_seg(color, text_prompt, topk=topk)
        return masks
    
    def get_first_frame(self, observation, text_prompt, camera='head_camera', save_path=None, sample_num=None, feature_type=None, additional_obj_file_info=None): # additional_obj_file_info -> list
        assert feature_type is not None, 'feature_type is not defined'
        assert sample_num % self.topk == 0, 'sample num error'
        print('Current Feature Type: ', feature_type)

        camera_intrinsic, cam2world_matrix = observation['observation'][camera]['intrinsic_cv'], observation['observation'][camera]['cam2world_gl']
        color = observation['observation'][camera]['rgb'][..., :3]
        depth = observation['observation'][camera]['depth'][...] / 1000 # !
        
        feature_line_pcad_masked, pcd, pcd_rgb = None, None, None
        feature_pcd = None
        feature_pcd_list = []
        
        masks = self.get_first_frame_masks(observation, text_prompt, camera=camera, topk=self.topk) 
        masks = sort_masks(masks)
        for i in range(self.topk):
            mask = np.array(masks[i][0].cpu())
            pose = self.pose_trackers[i].get_pose_first_frame(color, depth, mask)

            real_obj_world_matrix = cam2world_matrix @ np.array([[1, 0  , 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0,0,0,1]]) @ pose
            real_obj_world_pose = sapien.Pose(real_obj_world_matrix[:3,3], t3d.quaternions.mat2quat(real_obj_world_matrix[:3,:3]))

            if additional_obj_file_info is None:
                obj_file_path = self.mesh_file_paths[i]
                obj_model_data_path = self.model_data_paths[i]
            else:
                obj_file_path = additional_obj_file_info[0][i]
                obj_model_data_path = additional_obj_file_info[1][i]
                
            scene = G3FlowVirtualSpace()
            scene.load_actor(obj_file_path, obj_model_data_path, real_obj_world_pose)
            scene.load_camera(real_obj_world_pose.p, real_obj_world_pose.q)
            observation = scene.get_obs()

            camera_key_list = list(observation.keys())
            for j in range(len(camera_key_list)):
                sub_camera = observation[camera_key_list[j]]
                obj_color, obj_depth, obj_mask = sub_camera['rgb'], sub_camera['depth'] / 1000, sub_camera['mask']
                bounding_box_mask = get_bounding_box_mask(obj_mask) # hole image level
                region_color, region_depth, region_mask = extract_bounding_box_region(obj_color, obj_depth, bounding_box_mask, obj_mask) 
                feature_map, feature_line = self.get_dino_feature(region_color) # full
                feature_line_masked = feature_line[region_mask.reshape(-1)]

                pcd_tmp_sub_camera, pcd_rgb_tmp_sub_camera = tanslation_point_cloud(obj_depth, obj_color, sub_camera['intrinsic_cv'], sub_camera['cam2world_gl'], mask=obj_mask)

                if j == 0:
                    feature_line_pcad_masked = self.pca_transform(feature_line_masked)[:] # without xyz
                    pcd_tmp, pcd_rgb_tmp = pcd_tmp_sub_camera[:], pcd_rgb_tmp_sub_camera[:]
                else:
                    res_feature = self.pca_transform(feature_line_masked)[:]
                    feature_line_pcad_masked = np.concatenate((feature_line_pcad_masked, res_feature), axis=0)
                    pcd_tmp = np.concatenate((pcd_tmp, pcd_tmp_sub_camera[:]), axis=0)
                    pcd_rgb_tmp = np.concatenate((pcd_rgb_tmp, pcd_rgb_tmp_sub_camera[:]), axis=0)
                
                if save_path is not None:
                    if not os.path.exists(save_path):
                        os.makedirs(save_path)
                    feature_line_pcad = self.pca_transform(feature_line)
                    feature_line_pcad_rgb = feature_to_rgb(feature_line_pcad[:, :3], feature_map.shape[0], feature_map.shape[1])
                    max_val = np.max(region_depth[region_mask])
                    min_val = np.min(region_depth[region_mask])
                    normalized_depth = (region_depth - min_val) / (max_val - min_val) * 255
                    normalized_depth = normalized_depth.astype(np.uint8)
                    normalized_depth[~region_mask] = 255
                    cv2.imwrite(save_path+f'/_normal_depth_{i}_{j}.png', normalized_depth)

                    mask_image = Image.fromarray(region_mask)

                    mask_image.save(save_path+f'/_mask_{i}_{j}.png')

                    save_image(region_color, save_path+f'/_seg_with_mask_{i}_{j}.png', mask=region_mask, mask_color=255)
                    save_image(np.array(feature_line_pcad_rgb*255, dtype=np.uint8), save_path+f'/_pcad_{i}_{j}.png', mask=region_mask, mask_color=255)
                    save_image(color, save_path+f'/_raw_scene.png')
            
            _, indices = fps(pcd_tmp, num_points=sample_num // self.topk, use_cuda=True)
            indices = indices[0].cpu()
            sub_feature_pcd = np.concatenate((pcd_tmp[indices], feature_line_pcad_masked[indices]), axis=1)
            feature_pcd_list.append(sub_feature_pcd)

            self.trans_pcds[i].set_first_frame(feature_pcd_list[i], pose) # set first frame pose

        for i in range(self.topk):
            if i == 0:
                feature_pcd = feature_pcd_list[i]
            else:
                feature_pcd = np.concatenate((feature_pcd, feature_pcd_list[i]), axis=0)

        return feature_pcd

    def get_pose(self, color, depth):
        poses = []
        for i in range(self.topk):
            poses.append(self.pose_trackers[i].get_pose(color, depth))
        return poses
    
    def get_feature_pcd(self, poses):
        feature_pcd = None
        for i in range(self.topk):
            assert self.trans_pcds[i].first_set == True, 'First Frame is not set'
            if i == 0:
                feature_pcd = self.trans_pcds[i].trans_pcd(poses[i])
            else:
                feature_pcd = np.concatenate((feature_pcd, self.trans_pcds[i].trans_pcd(poses[i])), axis=0)
        return feature_pcd

    def track_pose(self):
        pass

    def pca_fit(self, features):
        self.feature_pca.fit(features)

    def pca_transform(self, features):
        return self.feature_pca.transform(features.cpu())


