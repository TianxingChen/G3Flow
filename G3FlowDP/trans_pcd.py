import numpy  as np
import torch
import cv2
import pdb
import pickle
import json
import open3d as o3d
from PIL import Image, ImageColor
import imageio
import os
import time
import sys
sys.path.append('../tools')
from function import *

class TransPCD:
    def __init__(self, cam2world_gl):
        self.cam2world_gl =  cam2world_gl
        self.first_set = False
    
    def set_first_frame(self, pcd_array, pose_matrix):
        self.first_set = True
        self.first_pose_matrix = pose_matrix
        self.first_pcd_array = pcd_array
        self.trans_opt = self.cam2world_gl @ self.first_pose_matrix     # pcd tarnsform operator

    def trans_pcd(self, pose_matrix):
        assert self.first_set, "No first pcd"
        pcd_array = self.first_pcd_array
        colors = pcd_array[...,3:]
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        trans_matrix = self.trans_opt @ np.linalg.inv(pose_matrix) @ np.linalg.inv(self.cam2world_gl)
        
        trans_matrix_torch = torch.tensor(trans_matrix,device=device)
        pcd_array_torch = torch.tensor(pcd_array[...,:3], device=device)
        pcd_array_torch_homogeneous = torch.cat((pcd_array_torch, torch.ones(pcd_array.shape[0], 1, device=device)), dim=1)
        new_pcd_array_homogeneous = torch.matmul(pcd_array_torch_homogeneous, trans_matrix_torch.inverse().T)
        new_pcd_array = new_pcd_array_homogeneous[..., :3] / new_pcd_array_homogeneous[..., 3].unsqueeze(1)

        return np.hstack((new_pcd_array.cpu().numpy(), colors))
    
    def clear(self):
        # reserve camera to world matrix, clear others
        self.first_set = False
        self.first_pose_matrix = None
        self.first_pcd_array = None
        self.trans_opt = None
        