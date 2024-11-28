from .base_task import Base_task
from .utils import *
import sapien

class diverse_bottles_pick_G(Base_task):
    def setup_demo(self,is_test = False,**kwags):
        super()._init(**kwags)
        self.create_table_and_wall()
        self.load_robot()
        self.setup_planner()
        self.load_camera(kwags.get('camera_w', 640), kwags.get('camera_h', 480))
        self.pre_move()
        if is_test:
            self.id_list = [2*i + 1 for i in range(11)]
        else:
            self.id_list = [2*i for i in range(11)]
        self.load_actors()
        self.step_lim = 400
    
    def pre_move(self):
        render_freq = self.render_freq
        self.render_freq=0

        self.together_close_gripper(save_freq=None)
        self.together_open_gripper(save_freq=None)

        self.render_freq = render_freq

    def load_actors(self, **kwargs):
        # super().setup_scene()
        self.left_bottle = np.random.choice(self.id_list)
        self.bottle1, self.bottle1_data = rand_create_actor(
            self.scene,
            xlim=[-0.25,-0.05],
            ylim=[0.03,0.23],
            zlim=[0.865],
            modelname="bottles",
            rotate_rand=False,
            qpos=[0.66, 0.66, -0.25, -0.25],
            convex=False,
            model_id = self.left_bottle,
            model_z_val = True
        )

        self.right_bottle = np.random.choice(self.id_list)
        self.bottle2, self.bottle2_data = rand_create_actor(
            self.scene,
            xlim=[0.05,0.25],
            ylim=[0.03,0.23],
            zlim=[0.865],
            modelname="bottles",
            rotate_rand=False,
            qpos=[0.65, 0.65, 0.27, 0.27],
            convex=False,
            model_id = self.right_bottle,
            model_z_val = True
        )

        self.bottle1.find_component_by_type(sapien.physx.PhysxRigidDynamicComponent).mass = 0.01
        self.bottle2.find_component_by_type(sapien.physx.PhysxRigidDynamicComponent).mass = 0.01

        render_freq = self.render_freq
        self.render_freq = 0
        for _ in range(4):
            self.together_open_gripper(save_freq=None)
        self.render_freq = render_freq

    def play_once(self,save_freq=None):
        left_pose0 = self.get_grasp_pose_w_given_direction(self.bottle1, self.bottle1_data,grasp_qpos=[-0.906,0,0,-0.424], pre_dis=0.1)
        right_pose0 = self.get_grasp_pose_w_given_direction(self.bottle2, self.bottle2_data,grasp_qpos=[-0.415,0,0,-0.910], pre_dis=0.11)
        left_pose1 = self.get_grasp_pose_w_given_direction(self.bottle1, self.bottle1_data,grasp_qpos=[-0.906,0,0,-0.424], pre_dis=0)
        right_pose1 = self.get_grasp_pose_w_given_direction(self.bottle2, self.bottle2_data,grasp_qpos=[-0.415,0,0,-0.910], pre_dis=0.01)
        left_target_pose = [-0.19,-0.12,0.92,1,0,0,0]
        right_target_pose = [0.19,-0.12,0.92,-0.01,0.01,0.03,-1]
        
        self.together_move_to_pose_with_screw(left_pose0,right_pose0,save_freq=15)

        self.together_move_to_pose_with_screw(left_pose1,right_pose1,save_freq=15)
        
        self.together_close_gripper(left_pos=0.005,right_pos=0.005,save_freq=15)

        left_pose1[2]+=0.08
        right_pose1[2]+=0.08
        self.together_move_to_pose_with_screw(left_pose1,right_pose1,save_freq=15)
        self.together_move_to_pose_with_screw(left_target_pose,right_target_pose,save_freq=15)
        info = dict()
        info['object_id'] = [int(self.left_bottle), int(self.right_bottle)]
        info['object_description'] = ['bottles', 2]
        info['object_type'] = ['bottles']
        return info

    def check_success(self):
        red_target = [-0.06,-0.105]
        green_target = [0.057,-0.11]
        eps = 0.025
        bottle1_pose = self.bottle1.get_pose().p
        bottle2_pose = self.bottle2.get_pose().p
        if bottle1_pose[2] < 0.78 or bottle2_pose[2] < 0.78:
            self.actor_pose = False
        return abs(bottle1_pose[0]-red_target[0])<eps and abs(bottle1_pose[1]-red_target[1])<eps and bottle1_pose[2]>0.89 and \
               abs(bottle2_pose[0]-green_target[0])<eps and abs(bottle2_pose[1]-green_target[1])<eps and bottle2_pose[2]>0.89
