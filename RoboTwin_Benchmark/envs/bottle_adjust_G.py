
from .base_task import Base_task
from .utils import *
import sapien
import math

class bottle_adjust_G(Base_task):
    def setup_demo(self, is_test=False, **kwags):
        super()._init(**kwags)
        self.create_table_and_wall()
        self.load_robot()
        self.setup_planner()
        self.load_camera(kwags.get('camera_w', 640),kwags.get('camera_h', 480))
        self.pre_move()
        if is_test:
            self.id_list = [5, 7, 8, 10, 17, 18, 21]
        else:
            self.id_list = [13]
        self.load_actors()
        self.step_lim = 400

        self.right_p = []
        self.left_p = []
        np.set_printoptions(suppress=True, precision=3)
    
    def pre_move(self):
        render_freq = self.render_freq
        self.render_freq=0

        self.together_close_gripper(save_freq=None)
        self.together_open_gripper(save_freq=None)

        self.render_freq = render_freq

    def load_actors(self, **kwargs):
        # super().setup_scene()

        self.qpose_tag = np.random.randint(0,2)
        qposes = [[0.707,0.05,0.05,-0.707],[0.707,-0.05,0.05,0.707]]
        
        self.sample_id = int(np.random.choice(self.id_list))

        self.bottle, self.bottle_data = rand_create_actor(
            self.scene,
            xlim=[0,0],
            ylim=[-0.13,-0.08],
            zlim=[0.78],
            rotate_rand=True,
            qpos=qposes[self.qpose_tag],
            modelname="bottles",
            convex=False,
            rotate_lim=(0,0,0.4),
            model_id=self.sample_id
        )
        
        self.bottle.find_component_by_type(sapien.physx.PhysxRigidDynamicComponent).mass = 0.01

        render_freq = self.render_freq
        self.render_freq = 0
        for _ in range(4):
            self.together_open_gripper(save_freq=None)
        self.render_freq = render_freq

        

    def play_once(self):
        grasp_pose = self.get_grasp_pose_w_labeled_direction(self.bottle, self.bottle_data,grasp_matrix=np.array([[0,0,1,0],[1,0,0,0],[0,1,0,0],[0,0,0,1]]), pre_dis=0)
        pre_pose = self.get_grasp_pose_w_labeled_direction(self.bottle, self.bottle_data,grasp_matrix=np.array([[0,0,1,0],[1,0,0,0],[0,1,0,0],[0,0,0,1]]), pre_dis=0.08)

        left_target_pose = [-0.25,-0.12,0.92,0.707,-0.05 ,0.,0.5]
        right_target_pose = [0.25,-0.12,0.92,0.5,-0.05,0.,0.707]
        if self.qpose_tag==1:  # right
            # move to pre pose
            self.right_move_to_pose_with_screw(pre_pose,save_freq=15)
            # move to grasp pose
            self.right_move_to_pose_with_screw(grasp_pose,save_freq=15)
            self.close_right_gripper(pos = 0.01,save_freq=15)
            # raise
            self.right_move_to_pose_with_screw(pre_pose,save_freq=15)
            # move to target
            self.right_move_to_pose_with_screw(right_target_pose,save_freq=15)

        else:#left
            # move to pre pose
            self.left_move_to_pose_with_screw(pre_pose,save_freq=15)
            # move to grasp pose
            self.left_move_to_pose_with_screw(grasp_pose,save_freq=15)
            self.close_left_gripper(pos = 0.01,save_freq=15)
            # raise
            self.left_move_to_pose_with_screw(pre_pose,save_freq=15)
            # move to target
            self.left_move_to_pose_with_screw(left_target_pose,save_freq=15)
        info = dict()
        info['object_id'] = [int(self.sample_id)]
        info['object_description'] = ['bottle', 1]
        info['object_type'] = ['bottles']
        return info

    def check_success(self):
        target_hight = 0.83 
        bottle_pose = self.bottle.get_pose().p
        return ((self.qpose_tag == 0 and bottle_pose[0]<-0.15) or (self.qpose_tag == 1 and bottle_pose[0]> 0.15)) \
               and bottle_pose[2]>target_hight