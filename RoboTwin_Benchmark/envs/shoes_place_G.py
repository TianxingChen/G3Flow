
from .base_task import Base_task
from .utils import *
import math
import sapien

class shoes_place_G(Base_task):
    def setup_demo(self,is_test = False, **kwags):
        super()._init(**kwags)
        self.create_table_and_wall()
        self.load_robot()
        self.setup_planner()
        self.load_camera(kwags.get('camera_w', 640),kwags.get('camera_h', 480))
        self.pre_move()
        if is_test:
            self.id_list = [3, 5, 10, 11, 12, 13, 14]
        else:
            self.id_list = [0, 4, 6]
        self.load_actors()
        self.step_lim = 600
    
    def pre_move(self):
        render_freq = self.render_freq
        self.render_freq=0
        self.together_open_gripper(save_freq=None)
        self.render_freq = render_freq

    def load_actors(self, **kwargs):
        # super().setup_scene()
        self.target = create_visual_box(
            scene = self.scene,
            pose = sapien.Pose([0,-0.13,0.74],[1,0,0,0]),
            half_size=(0.13,0.1,0.0005),
            color=(0,0,1),
            name="box"
        )


        # left shoe
        shoes_pose = rand_pose(
            xlim=[-0.25,-0.1],
            ylim=[-0.1,0.05],
            zlim=[0.8],
            ylim_prop=True,
            rotate_rand=True,
            rotate_lim=[0,3.14,0],
            qpos=[0.707,0.707,0,0]
        )

        while np.sum(pow(shoes_pose.get_p()[:2] - np.zeros(2),2)) < 0.0225:
            shoes_pose = rand_pose(
                xlim=[-0.25,-0.1],
                ylim=[-0.1,0.05],
                zlim=[0.8],
                ylim_prop=True,
                rotate_rand=True,
                rotate_lim=[0,3.14,0],
                qpos=[0.707,0.707,0,0]
            )
        
        sample_id = np.random.choice(self.id_list)
        self.sample_id = int(sample_id)

        self.left_shoe, self.shoe_data = create_actor(
            self.scene,
            pose=shoes_pose,
            modelname="shoes",
            convex=True,
            model_id = sample_id,
            model_z_val = True
        )

        # right shoe
        shoes_pose = rand_pose(
            xlim=[0.1,0.25],
            ylim=[-0.1,0.05],
            zlim=[0.8],
            ylim_prop=True,
            rotate_rand=True,
            rotate_lim=[0,3.14,0],
            qpos=[0.707,0.707,0,0]
        )

        while np.sum(pow(shoes_pose.get_p()[:2] - np.zeros(2),2)) < 0.0225:
            shoes_pose = rand_pose(
                xlim=[0.1,0.25],
                ylim=[-0.1,0.05],
                zlim=[0.8],
                ylim_prop=True,
                rotate_rand=True,
                rotate_lim=[0,3.14,0],
                qpos=[0.707,0.707,0,0]
            )
        

        self.right_shoe, _ = create_actor(
            self.scene,
            pose=shoes_pose,
            modelname="shoes",
            convex=True,
            model_id = sample_id,
            model_z_val = True
        )

        self.left_shoe.find_component_by_type(sapien.physx.PhysxRigidDynamicComponent).mass = 0.1
        self.right_shoe.find_component_by_type(sapien.physx.PhysxRigidDynamicComponent).mass = 0.1

    def get_target_grap_pose(self,shoe_rpy):
        if math.fmod(math.fmod(shoe_rpy[2] + shoe_rpy[0], 2 * math.pi) + 2 * math.pi, 2*math.pi) < math.pi:
            grasp_matrix = np.array([[-1,0,0,0],[0,1,0,0],[0,0,-1,0],[0,0,0,1]])
            target_quat = [-0.707,0,-0.707,0]
        else:
            grasp_matrix = np.eye(4)
            target_quat = [0,0.707,0,-0.707]
        return grasp_matrix, target_quat

    def play_once(self):
        left_shoe_rpy = self.left_shoe.get_pose().get_rpy()
        right_shoe_rpy = self.right_shoe.get_pose().get_rpy()

        left_grasp_matrix, left_target_quat = self.get_target_grap_pose(left_shoe_rpy)
        right_grasp_matrix, right_target_quat = self.get_target_grap_pose(right_shoe_rpy)
        
        # use right arm move
        left_pose1 = self.get_grasp_pose_w_labeled_direction(self.left_shoe,self.shoe_data, grasp_matrix = left_grasp_matrix, pre_dis=0.1)
        right_pose1 = self.get_grasp_pose_w_labeled_direction(self.right_shoe,self.shoe_data, grasp_matrix = right_grasp_matrix, pre_dis=0.1)
        self.together_move_to_pose_with_screw(left_target_pose=left_pose1, right_target_pose = right_pose1, save_freq = 15)
        left_pose1 = self.get_grasp_pose_w_labeled_direction(self.left_shoe,self.shoe_data, grasp_matrix = left_grasp_matrix, pre_dis=0)
        right_pose1 = self.get_grasp_pose_w_labeled_direction(self.right_shoe,self.shoe_data, grasp_matrix = right_grasp_matrix, pre_dis=0)
        self.together_move_to_pose_with_screw(left_target_pose=left_pose1, right_target_pose = right_pose1, save_freq = 15)
        self.together_close_gripper(save_freq = 15)
        left_pose1[2] += 0.07
        right_pose1[2] += 0.07            
        self.together_move_to_pose_with_screw(left_target_pose=left_pose1, right_target_pose = right_pose1, save_freq = 15)

        right_temp_pose = [0.25,-0.07,right_pose1[2]] + right_target_quat
        left_target_pose = [0,-0.19,left_pose1[2]] + left_target_quat
        right_target_pose = [0.01,-0.07,right_pose1[2]] + right_target_quat

        self.together_move_to_pose_with_screw(left_target_pose = left_target_pose, right_target_pose=right_temp_pose, save_freq = 15)
        left_target_pose[2] -=0.06
        self.left_move_to_pose_with_screw(pose = left_target_pose,save_freq = 15)
        self.open_left_gripper(save_freq = 15)
        left_target_pose[2] +=0.06
        self.left_move_to_pose_with_screw(pose = left_target_pose,save_freq = 15)
        self.together_move_to_pose_with_screw(left_target_pose=self.left_original_pose, right_target_pose = right_target_pose,save_freq = 15)
        right_target_pose[2] -=0.06
        self.right_move_to_pose_with_screw(pose = right_target_pose,save_freq = 15)
        self.open_right_gripper(save_freq = 15)

        info = dict()
        info['object_id'] = [self.sample_id, self.sample_id]
        info['object_description'] = ['white shoe', 2]
        info['object_type'] = ['shoes']
        return info

    def check_success(self):
        left_shoe_pose_p = np.array(self.left_shoe.get_pose().p)
        left_shoe_pose_q = np.array(self.left_shoe.get_pose().q)
        right_shoe_pose_p = np.array(self.right_shoe.get_pose().p)
        right_shoe_pose_q = np.array(self.right_shoe.get_pose().q)
        if left_shoe_pose_q[0] < 0:
            left_shoe_pose_q *= -1
        if right_shoe_pose_q[0] < 0:
            right_shoe_pose_q *= -1
        target_pose_p = np.array([0,-0.13])
        target_pose_q = np.array([0.5,0.5,-0.5,-0.5])
        eps = np.array([0.02,0.02,0.05,0.05,0.05,0.05])
        return np.all(abs(left_shoe_pose_p[:2] - (target_pose_p - [0,0.06])) < eps[:2]) and np.all(abs(left_shoe_pose_q - target_pose_q) < eps[-4:]) and \
               np.all(abs(right_shoe_pose_p[:2] - (target_pose_p + [0,0.06])) < eps[:2]) and np.all(abs(right_shoe_pose_q - target_pose_q) < eps[-4:]) and self.is_left_gripper_open() and self.is_right_gripper_open()
        
