
import sys
import warnings
import os
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
current_file_path = os.path.abspath(__file__)
parent_dir = os.path.dirname(current_file_path)

sys.path.append(os.path.join(parent_dir, '../../tools'))
import numpy as np
import pdb
import json
import pickle
import torch
import argparse
import sapien.core as sapien
from sapien.utils.viewer import Viewer
import gymnasium as gym
import toppra as ta
import transforms3d as t3d
from collections import OrderedDict
import random

class G3FlowVirtualSpace(gym.Env):
    def __init__(self):
        super().__init__()
        ta.setup_logging("CRITICAL") # hide logging

        self.setup_scene()

    def setup_scene(self,**kwargs):
        '''
        Set the scene
            - Set up the basic scene: light source, viewer.
        '''
        self.engine = sapien.Engine()
        # declare sapien renderer
        from sapien.render import set_global_config
        set_global_config(max_num_materials = 50000, max_num_textures = 50000)
        self.renderer = sapien.SapienRenderer()
        # give renderer to sapien sim
        self.engine.set_renderer(self.renderer)
        
        sapien.render.set_camera_shader_dir("rt")
        sapien.render.set_ray_tracing_samples_per_pixel(32)
        sapien.render.set_ray_tracing_path_depth(8)
        sapien.render.set_ray_tracing_denoiser("oidn")

        # declare sapien scene
        scene_config = sapien.SceneConfig()
        self.scene = self.engine.create_scene(scene_config)

        # give some white ambient light of moderate intensity
        self.scene.set_ambient_light(kwargs.get("ambient_light", [0.5, 0.5, 0.5]))
        # default enable shadow unless specified otherwise
        shadow = kwargs.get("shadow", True)
        # default spotlight angle and intensity
        direction_lights = kwargs.get(
            "direction_lights", [[[0, 0.5, -1], [0.5, 0.5, 0.5]]]
        )
        for direction_light in direction_lights:
            self.scene.add_directional_light(
                direction_light[0], direction_light[1], shadow=shadow
            )
        # default point lights position and intensity
        point_lights = kwargs.get(
            "point_lights",
            [[[1, 0, 1.8], [1, 1, 1]], [[-1, 0, 1.8], [1, 1, 1]]]
        )
        for point_light in point_lights:
            self.scene.add_point_light(point_light[0], point_light[1], shadow=shadow)

    def _cal_camera_matrix(self, camera_pose, center_pose, camera_up_pose):
        camera_cam_forward = center_pose - camera_pose
        camera_cam_forward = camera_cam_forward / np.linalg.norm(camera_cam_forward)
        camera_cam_up = camera_up_pose / np.linalg.norm(camera_up_pose)
        camera_cam_left = np.cross(camera_cam_up, camera_cam_forward)
        camera_cam_left = camera_cam_left / np.linalg.norm(camera_cam_left)
        camera_mat44 = np.eye(4)
        camera_mat44[:3, :3] = np.stack([camera_cam_forward, camera_cam_left, camera_cam_up], axis=1)
        camera_mat44[:3, 3] = camera_pose
        return camera_mat44

    def load_camera(self, center_point, obj_quat, camera_w = 320,camera_h = 240):
        '''
            Add cameras and set camera parameters
                - Including four cameras: left, right, front, up.
        '''
        near, far = 0.1, 1
        width, height = camera_w, camera_h

        cam_height = 0.3
        delta = 0.3
        obj_matrix = t3d.quaternions.quat2mat(obj_quat) @ np.array(self.actor_data["trans_matrix"])[:3,:3]

        inv_matrix =  obj_matrix @ np.linalg.inv(np.array([[1,0,0],[0,0,1],[0,1,0]]))
        theta = 0
        local_rng = np.random.RandomState(int(np.abs(center_point[0]*1000)))
        angles = np.radians(local_rng.uniform(-20,20,3))
        print(angles)
        Rx = t3d.euler.euler2mat(angles[0], 0, 0)
        Ry = t3d.euler.euler2mat(0, angles[1], 0)
        Rz = t3d.euler.euler2mat(0, 0, angles[2])
        R = np.dot(Rz, np.dot(Ry, Rx))

        d_theta = 2 * np.pi / 3

        front_cam_pose = center_point + np.dot(inv_matrix @ (np.array([delta,delta,cam_height]) * np.array([np.cos(theta), np.sin(theta),1])), R.T)
        left_cam_pose = center_point + np.dot(inv_matrix @ (np.array([delta, delta, cam_height]) * np.array([np.cos(theta + d_theta), np.sin(theta + d_theta),1])), R.T)
        right_cam_pose = center_point + np.dot(inv_matrix @ (np.array([delta, delta, cam_height]) * np.array([np.cos(theta - d_theta), np.sin(theta - d_theta),1])), R.T)
        z_arix_up = inv_matrix @ np.array([0,0,1])
        
        # front camera
        front_mat44 = self._cal_camera_matrix(front_cam_pose, center_point, cal_up_arix(center_point, front_cam_pose, z_arix_up))
        # left camera
        left_mat44 = self._cal_camera_matrix(left_cam_pose, center_point, cal_up_arix(center_point, left_cam_pose, z_arix_up))
        # right camera
        right_mat44 = self._cal_camera_matrix(right_cam_pose, center_point, cal_up_arix(center_point, right_cam_pose, z_arix_up))

        self.camera_list = []

        self.front_camera = self.scene.add_camera(
            name="front_camera",
            width=width,
            height=height,
            fovy=np.deg2rad(60),
            near=near,
            far=far,
        )
        self.front_camera.entity.set_pose(sapien.Pose(front_mat44))
        self.camera_list.append(self.front_camera)

        self.left_camera = self.scene.add_camera(
            name="left_camera",
            width=width,
            height=height,
            fovy=np.deg2rad(60),
            near=near,
            far=far,
        )
        self.left_camera.entity.set_pose(sapien.Pose(left_mat44))
        self.camera_list.append(self.left_camera)

        self.right_camera = self.scene.add_camera(
            name="right_camera",
            width=width,
            height=height,
            fovy=np.deg2rad(60),
            near=near,
            far=far,
        )
        self.right_camera.entity.set_pose(sapien.Pose(right_mat44))
        self.camera_list.append(self.right_camera)

        self.scene.step()  # run a physical step
        self.scene.update_render()  # sync pose from SAPIEN to renderer

    # Get Camera RGBA
    def _get_camera_rgba(self, camera):
        rgba = camera.get_picture("Color")
        rgba_img = (rgba * 255).clip(0, 255).astype("uint8")
        return rgba_img
    
    # Get Camera Depth
    def _get_camera_depth(self, camera):
        position = camera.get_picture("Position")
        depth = -position[..., 2]
        depth_image = (depth * 1000.0).astype(np.float64)
        unique, counts = np.unique(position[..., 3], return_counts=True)
        most_common_index = np.argmax(counts)
        mask = position[..., 3] != unique[most_common_index]
        return depth_image, mask
    
    def _get_camera_pcd(self, camera, point_num = 0):
        rgba = camera.get_picture_cuda("Color").torch() # [H, W, 4]
        position = camera.get_picture_cuda("Position").torch()
        model_matrix = camera.get_model_matrix()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_matrix = torch.tensor(model_matrix, dtype=torch.float32).to(device)

        # Extract valid three-dimensional points and corresponding color data.
        valid_mask = position[..., 3] < 1
        points_opengl = position[..., :3][valid_mask]
        points_color = rgba[valid_mask][:,:3]
        # Transform into the world coordinate system.
        points_world = torch.bmm(points_opengl.view(1, -1, 3), model_matrix[:3, :3].transpose(0,1).view(-1, 3, 3)).squeeze(1) + model_matrix[:3, 3]
        points_world = points_world.cpu().numpy()[0]
        points_color = points_color.cpu().numpy()
        return np.hstack((points_world, points_color))


    def get_obs(self):
        self.scene.update_render()
        pkl_dic = OrderedDict()
        for camera in self.camera_list:
            camera.take_picture()
            camera_intrinsic_cv = camera.get_intrinsic_matrix()
            camera_extrinsic_cv = camera.get_extrinsic_matrix()
            camera_model_matrix = camera.get_model_matrix()
            camera_name = camera.get_name()
            pkl_dic[camera_name] = {
                "intrinsic_cv" : camera_intrinsic_cv,
                "extrinsic_cv" : camera_extrinsic_cv,
                "cam2world_gl" : camera_model_matrix
            }
            rbga = self._get_camera_rgba(camera)
            depth, mask = self._get_camera_depth(camera)
            pcd = self._get_camera_pcd(camera)
            pkl_dic[camera_name]["rgb"] = rbga[:,:,:3]
            pkl_dic[camera_name]["depth"] = depth
            pkl_dic[camera_name]["mask"] = mask
            pkl_dic[camera_name]["pcd"] = pcd
            # pkl_dic[camera_name]["rgb"][~mask] = [127,0,255]
        return pkl_dic

    def load_actor(self, actor_file_path, actor_data_path, actor_world_pose):
        self.actor, self.actor_data = create_actor(
            self.scene,
            actor_world_pose,
            actor_file_path,
            actor_data_path,
            is_static=True
        )
    
def cal_up_arix(o, p, z):
    vec1 = o - p
    vec2 = z
    # pdb.set_trace()
    r = - np.sum(vec1 ** 2) / np.sum(vec2 * vec1)
    return r * vec2 + o - p

# create obj model
def create_actor(
    scene: sapien.Scene,
    pose: sapien.Pose,
    model_path: str,
    model_data_path: str,
    is_static = True,
) -> sapien.Entity:
    file_name =  model_path
    assert file_name[-3:] == 'glb', 'need glb but obj'

    builder = scene.create_actor_builder()
    if is_static:
        builder.set_physx_body_type("static")
    else:
        builder.set_physx_body_type("dynamic")
    
    scale= (1,1,1)
    try:
        with open(model_data_path, 'r') as file:
            model_data = json.load(file)
        scale = model_data["scale"]
    except:
        model_data = None

    builder.add_visual_from_file(
        filename=file_name,
        scale= scale)
    mesh = builder.build()
    mesh.set_pose(pose)
    
    return mesh, model_data


if __name__ == '__main__':
    a = G3FlowVirtualSpace()