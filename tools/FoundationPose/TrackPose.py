import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
from estimater import *
from datareader import *
import argparse
import logging


class FoundationPoseTracker:
    def __init__(self, mesh_file, K, debug_dir, debug, est_refine_iter=5, track_refine_iter=2): 
        set_logging_format(level=logging.CRITICAL)
        set_seed(0)
        mesh = trimesh.load(mesh_file) # obj
        to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
        bbox = np.stack([-extents/2, extents/2], axis=0).reshape(2,3)
        scorer = ScorePredictor()
        refiner = PoseRefinePredictor()
        glctx = dr.RasterizeCudaContext()
        est = FoundationPose(model_pts=mesh.vertices, model_normals=mesh.vertex_normals, mesh=mesh, scorer=scorer, refiner=refiner, debug_dir=debug_dir, debug=debug, glctx=glctx)
        logging.info("estimator initialization done")

        self.if_first = True
        self.K = K
        self.est_refine_iter, self.track_refine_iter = est_refine_iter, track_refine_iter
        self.est = est
        self.to_origin, self.bbox = to_origin, bbox
    
    def reset(self, mesh_file, K, debug_dir, debug, est_refine_iter=5, track_refine_iter=2):
        mesh = trimesh.load(mesh_file) # obj
        self.reset_object(model_pts=mesh.vertices, model_normals=mesh.vertex_normals, mesh=mesh)
    
    # color: (H, W, 3), np.int8
    # depth: (H, W), float64
    # mask: (H, W), True or False
    # reader.K: 3*3
    
    def get_pose_first_frame(self, color, depth, mask):
        self.if_first = False
        pose = self.est.register(K=self.K, rgb=color, depth=depth, ob_mask=mask, iteration=self.est_refine_iter)
        return pose

    def get_pose(self, color, depth):
        assert not self.if_first, 'first frame should be set'
        pose = self.est.track_one(rgb=color, depth=depth, K=self.K, iteration=self.track_refine_iter)
        return pose

    def vis_result(self, color, depth, pose, save_path):
        center_pose = pose@np.linalg.inv(self.to_origin)
        vis = draw_posed_3d_box(self.K, img=color, ob_in_cam=center_pose, bbox=self.bbox)
        vis = draw_xyz_axis(color, ob_in_cam=center_pose, scale=0.1, K=self.K, thickness=3, transparency=0, is_input_rgb=True)
        imageio.imwrite(save_path, vis)
