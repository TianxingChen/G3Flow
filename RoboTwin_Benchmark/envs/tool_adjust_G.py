
from .base_task import Base_task
from .utils import *
import sapien
import math

class tool_adjust_G(Base_task):
    def setup_demo(self, is_test=False, **kwags):
        super()._init(**kwags)
        self.create_table_and_wall()
        self.load_robot()
        self.setup_planner()
        self.load_camera(kwags.get('camera_w', 640),kwags.get('camera_h', 480))
        self.pre_move()
        if is_test:
            self.id_list = [1, 4, 7, 8]
        else:
            self.id_list = [3, 5]
        self.load_actors()
        self.step_lim = 300

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
        self.hand_tag = np.random.randint(0,2) 
        self.model_id = int(np.random.choice(self.id_list))
        qpos_list = [[[0.077,0.703,-0.703,0.077],[0.707,0,0,0.707],[-0.465,-0.533,-0.471,0.527],[0,0.707,-0.707,0],
                      [-0.5,-0.5,0.5,0.5],[0.5,-0.5,0.5,0.5],[0.5,0.5,0.5,0.5],[0.5,0.5,-0.5,0.5],[0.707,0,0,0.707], [0.707,0,0,0.707]],
                     [[0.077,0.703,0.703,0.077],[0.707,0,0,-0.707],[-0.465,0.533,-0.471,-0.527],[0,0.707,0.707,0],
                      [0.5,0.5,0.5,0.5],[0.5,0.5,0.5,-0.5],[-0.5,-0.5,0.5,0.5],[-0.5,0.5,0.5,0.5],[0.707,0,0,-0.707], [0.707,0,0,-0.707]]]
        zlim_list = [0.785, 0.78, 0.79, 0.79, 0.79, 0.79,0.79,0.78, 0.769, 0.77]
        convex_list = [False,True,False,False,False,False,False,False, False, True]
        now_qpos = qpos_list[self.hand_tag][self.model_id]

        self.object, self.object_data = rand_create_glb(
            self.scene,
            xlim=[0., 0.],
            ylim=[-0.13,0],
            zlim=[zlim_list[self.model_id]],
            rotate_rand=False,
            qpos=now_qpos,
            modelname="tools",
            convex=convex_list[self.model_id],
            rotate_lim=(0,0,0),
            model_id=self.model_id,
        )
        
        self.object.find_component_by_type(sapien.physx.PhysxRigidDynamicComponent).mass = 0.01
        render_freq = self.render_freq
        self.render_freq = 0
        for _ in range(4):
            self.together_open_gripper(save_freq=None)
        self.render_freq = render_freq

    def play_once(self):
        if self.hand_tag == 0: # right 1,left 0
            grasp_qpos=[-0,0.707,0,-0.707]
            grasp_pose = self.get_grasp_pose_w_given_direction(self.object, self.object_data,grasp_qpos=grasp_qpos)
            pre_pose = self.get_grasp_pose_w_given_direction(self.object, self.object_data,grasp_qpos=grasp_qpos, pre_dis=0.08)
            self.right_move_to_pose_with_screw(pre_pose,save_freq=15)
            grasp_pose[2] = 0.9
            self.right_move_to_pose_with_screw(grasp_pose,save_freq=15)
            self.close_right_gripper(save_freq=15)
            grasp_pose[2] += 0.1
            self.right_move_to_pose_with_screw(grasp_pose,save_freq=15)
        else:
            grasp_qpos=[0.707,0,0.707,-0]
            grasp_pose = self.get_grasp_pose_w_given_direction(self.object, self.object_data,grasp_qpos=grasp_qpos)
            pre_pose = self.get_grasp_pose_w_given_direction(self.object, self.object_data,grasp_qpos=grasp_qpos, pre_dis=0.08)
            self.left_move_to_pose_with_screw(pre_pose,save_freq=15)
            grasp_pose[2] = 0.9
            self.left_move_to_pose_with_screw(grasp_pose,save_freq=15)
            self.close_left_gripper(save_freq=15)
            grasp_pose[2] += 0.1
            self.left_move_to_pose_with_screw(grasp_pose,save_freq=15)
        
        object_description = ['hammer', 'screwdriver', 'flashlight', 'electric shaver', 'electric shaver', 'toothbrush', 'digital camera', 'hair dryer', 'ultraman transforming device', 'torch']
        info = dict()
        info['object_id'] = [self.model_id]
        info['object_description'] = [object_description[self.model_id], 1]
        info['object_type'] = ['tools']
        return info

    def check_success(self):
        target_hight = 0.78
        endpose_x_limit = 0.1
        object_pose = self.object.get_pose().p
        if self.hand_tag == 1:
            endpose_x = abs(self.left_endpose.global_pose.p[0])
        else :
            endpose_x = abs(self.right_endpose.global_pose.p[0])
        return object_pose[2] > target_hight and endpose_x < endpose_x_limit
