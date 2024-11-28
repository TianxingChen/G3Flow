from .base_task import Base_task
from .utils import *
import math
import sapien

class shoe_place_T(Base_task):
    def setup_demo(self, is_test = False, **kwags):
        super()._init(**kwags)
        self.create_table_and_wall()
        self.load_robot()
        self.setup_planner()
        self.load_camera(kwags.get('camera_w', 640),kwags.get('camera_h', 480))
        self.pre_move()
        if is_test:
            self.id_list = [2*i+1 for i in range(5)]
        else:
            self.id_list = [2*i for i in range(5)]
        self.load_actors()
        self.step_lim = 450
            
    def pre_move(self):
        render_freq = self.render_freq
        self.render_freq=0
        self.together_open_gripper(save_freq=None)
        self.render_freq = render_freq

    def load_actors(self, **kwargs):
        # super().setup_scene()
        self.target = create_visual_box(
            scene = self.scene,
            pose = sapien.Pose([0,-0.08,0.74],[1,0,0,0]),
            half_size=(0.13,0.05,0.0005),
            color=(0,0,1),
            name="box"
        )

        shoes_pose = rand_pose(
            xlim=[-0.25,0.25],
            ylim=[-0.1,0.05],
            zlim=[0.8],
            ylim_prop=True,
            rotate_rand=True,
            rotate_lim=[0,3.14,0],
            qpos=[0.707,0.707,0,0]
        )

        while np.sum(pow(shoes_pose.get_p()[:2] - np.zeros(2),2)) < 0.0225:
            shoes_pose = rand_pose(
                xlim=[-0.25,0.25],
                ylim=[-0.1,0.05],
                zlim=[0.8],
                ylim_prop=True,
                rotate_rand=True,
                rotate_lim=[0,3.14,0],
                qpos=[0.707,0.707,0,0]
            )
        
        sample_id = np.random.choice(self.id_list)
        self.sample_id = int(sample_id)
        self.shoe, self.shoe_data = create_actor(
            self.scene,
            pose=shoes_pose,
            modelname="shoes",
            convex=True,
            model_id = sample_id,
            # model_id = self.ep_num,
            model_z_val = True
        )

        self.shoe.find_component_by_type(sapien.physx.PhysxRigidDynamicComponent).mass = 0.1


    def play_once(self):
        shoe_rpy = self.shoe.get_pose().get_rpy()
        if math.fmod(math.fmod(shoe_rpy[2] + shoe_rpy[0], 2 * math.pi) + 2 * math.pi, 2*math.pi) < math.pi:
            grasp_matrix = np.array([[-1,0,0,0],[0,1,0,0],[0,0,-1,0],[0,0,0,1]])
            target_quat = [-0.707,0,-0.707,0]
        else:
            grasp_matrix = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])
            target_quat = [0,0.707,0,-0.707]

        if self.shoe.get_pose().p[0] > 0:
            # use right arm move
            right_pose1 = self.get_grasp_pose_w_labeled_direction(self.shoe,self.shoe_data, grasp_matrix = grasp_matrix, pre_dis=0.1)
            self.right_move_to_pose_with_screw(pose = right_pose1,save_freq = 15)
            right_pose1 = self.get_grasp_pose_w_labeled_direction(self.shoe,self.shoe_data, grasp_matrix = grasp_matrix, pre_dis=0)
            self.right_move_to_pose_with_screw(pose = right_pose1,save_freq = 15)
            self.close_right_gripper(save_freq = 15)
            right_pose1[2] += 0.07
            self.right_move_to_pose_with_screw(pose = right_pose1,save_freq = 15)
            target_pose = [0,-0.08,right_pose1[2]] + target_quat
            self.right_move_to_pose_with_screw(pose = target_pose,save_freq = 15)
            target_pose[2] -=0.06
            self.right_move_to_pose_with_screw(pose = target_pose,save_freq = 15)
            self.open_right_gripper(save_freq = 15)
            target_pose[2] +=0.06
            self.right_move_to_pose_with_screw(pose = target_pose,save_freq = 15)
        else:
            # use left arm move
            left_pose1 = self.get_grasp_pose_w_labeled_direction(self.shoe,self.shoe_data, grasp_matrix = grasp_matrix, pre_dis=0.1)
            self.left_move_to_pose_with_screw(pose = left_pose1,save_freq = 15)
            left_pose1 = self.get_grasp_pose_w_labeled_direction(self.shoe,self.shoe_data, grasp_matrix = grasp_matrix, pre_dis=0)
            self.left_move_to_pose_with_screw(pose = left_pose1,save_freq = 15)
            self.close_left_gripper(save_freq = 15)
            left_pose1[2] += 0.07
            self.left_move_to_pose_with_screw(pose = left_pose1,save_freq = 15)
            target_pose = [0,-0.08,left_pose1[2]] + target_quat
            self.left_move_to_pose_with_screw(pose = target_pose,save_freq = 15)
            target_pose[2] -=0.06
            self.left_move_to_pose_with_screw(pose = target_pose,save_freq = 15)
            self.open_left_gripper(save_freq = 15)
            target_pose[2] +=0.06
            self.left_move_to_pose_with_screw(pose = target_pose,save_freq = 15)
        info = dict()
        info['object_id'] = [self.sample_id]
        info['object_description'] = ['white shoe', 1]
        info['object_type'] = ['shoes']
        return info

    def check_success(self):
        shoe_pose_p = np.array(self.shoe.get_pose().p)
        shoe_pose_q = np.array(self.shoe.get_pose().q)
        
        if shoe_pose_q[0] < 0:
            shoe_pose_q *= -1

        target_pose_p = np.array([0,-0.08])
        target_pose_q = np.array([0.5,0.5,-0.5,-0.5])
        eps = np.array([0.05,0.02,0.05,0.05,0.05,0.05])
        endpose_z = max(self.get_right_endpose_pose().p[2], self.get_left_endpose_pose().p[2])
        return np.all(abs(shoe_pose_p[:2] - target_pose_p) < eps[:2]) and np.all(abs(shoe_pose_q - target_pose_q) < eps[-4:] )and endpose_z > 0.96 and self.is_left_gripper_open() and self.is_right_gripper_open()
