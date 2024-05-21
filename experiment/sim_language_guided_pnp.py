import os, sys
# os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"   
# os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

from jax.experimental.compilation_cache import compilation_cache as cc
cc.set_cache_dir("./__jax_cache__")
# cc.initialize_cache("./__jax_cache__")

from pathlib import Path
import glob
import jax.numpy as jnp
import jax
import pickle
import torch
import numpy as np
import yaml
import pybullet as pb
import time
from functools import partial
from scipy.spatial.transform import Rotation as sciR
from typing import List, Dict
import random
import matplotlib.pyplot as plt
import open3d as o3d

BASEDIR = Path(__file__).parent.parent
if BASEDIR not in sys.path:
    sys.path.insert(0, str(BASEDIR))

from imm_pb_util.bullet_envs.robot import FrankaPanda, PandaGripper, PandaGripperVisual
from pybullet_utils.bullet_client import BulletClient
from examples.latent_language_guided_pnp import LanguageGuidedPnP, VirtualWorld
from typing import Tuple
from imm_pb_util.bullet_envs.env import BulletEnv
from imm_pb_util.bullet_envs.camera import SimpleCamera, compute_gl_view_matrix_from_gl_camera_pose
from examples.visualize_occ import create_mesh_from_latent
import util.cvx_util as cxutil
import experiment.exp_utils as exutil

def configure_bullet(config: Dict, bc=None):
    # Configuration
    DEBUG_SHOW_GUI         = config["project_params"]["debug_show_gui"]
    DEBUG_GUI_DELAY        = config["project_params"]["debug_delay_gui"]
    CONTROL_HZ             = config["sim_params"]["control_hz"]
    GRAVITY                = config["sim_params"]["gravity"]
    CAMERA_DISTANCE        = config["sim_params"]["debug_camera"]["distance"]
    CAMERA_YAW             = config["sim_params"]["debug_camera"]["yaw"]
    CAMERA_PITCH           = config["sim_params"]["debug_camera"]["pitch"]
    CAMERA_TARGET_POSITION = config["sim_params"]["debug_camera"]["target_position"]
    if bc is None:
        if DEBUG_SHOW_GUI:
            bc = BulletClient(connection_mode=pb.GUI)
        else:
            bc = BulletClient(connection_mode=pb.DIRECT)

    # Sim params
    CONTROL_DT = 1. / CONTROL_HZ
    bc.setTimeStep(CONTROL_DT)
    bc.setGravity(0, 0, GRAVITY)
    # bc.resetDebugVisualizerCamera(
    #     cameraDistance       = CAMERA_DISTANCE, 
    #     cameraYaw            = CAMERA_YAW, 
    #     cameraPitch          = CAMERA_PITCH, 
    #     cameraTargetPosition = CAMERA_TARGET_POSITION )
    bc.resetDebugVisualizerCamera(
        cameraDistance       = 1.00,
        cameraYaw            = -60.20,
        cameraPitch          = -22.80, 
        cameraTargetPosition = [0.10, 0.06, -0.02] )
    bc.configureDebugVisualizer(bc.COV_ENABLE_RENDERING)
    bc.configureDebugVisualizer(bc.COV_ENABLE_GUI,0)
    bc.setPhysicsEngineParameter(enableFileCaching=0)

    return bc

class SimPnPEnv(BulletEnv):
    """
    ?
    """
    def __init__(self, bc: BulletClient, env_type='table'):
        """Load ground plane.

        Args:
            bc (BulletClient): _
        """
        super().__init__(bc, False)

        if env_type=='table':
            table_id = bc.createMultiBody(
                    baseMass = 0.0, 
                    basePosition = [-0.2,0,-0.1],
                    baseCollisionShapeIndex = bc.createCollisionShape(bc.GEOM_BOX, halfExtents=[1.0, 0.8, 0.1]),
                    baseVisualShapeIndex = bc.createVisualShape(bc.GEOM_BOX, halfExtents=[1.0, 0.8, 0.1]))
                #   Table height randomization
            # bc.changeVisualShape(table_id, -1, rgbaColor=[0.9670298390136767, 0.5472322491757223, 0.9726843599648843, 1.0])
            bc.changeVisualShape(table_id, -1, rgbaColor=list(np.random.uniform(0,1,size=(3,)))+[1,])
            # Register
            self.env_assets['table'] = table_id
        elif env_type=='shelf':
            table_id = bc.loadURDF(fileName='imm_pb_util/urdf/cabinet/cabinet_topless.urdf')
            self.env_assets['shelf'] = table_id


    @property
    def env_uids(self) -> List[int]:
        return list(self.env_assets.values())


def log_stats(stats, case):
    stats[case] += 1
    if case == "trial":
        return
    print(stats)


def step(bc, steps=-1):
    while steps != 0: 
        bc.stepSimulation()
        steps -= 1


class PnPEnv(object):

    def __init__(self, config: Dict, env_type='table'):
        self.config = config
        """Demo main function

        Args:
            config (Dict): Configuration dict
        """

        # Random control
        SEED = 5
        np.random.seed(SEED)
        random.seed(SEED)
        torch.manual_seed(SEED)

        self.env_type = env_type
        
        self.bc = None
        self.reset_env()

    def reset_env(self):
        self.bc = configure_bullet(self.config, self.bc)
        # Config
        NUM_OBJ = 4
        remove_list = ['data/NOCS_val/32_64_1_v4/bottle-e29e8121d93af93bba803759c05367b0.obj']
        bottle_list = [fn for fn in glob.glob('data/NOCS_val/32_64_1_v4/bottle*.obj') if fn not in remove_list]
        bowl_list = [fn for fn in glob.glob('data/NOCS_val/32_64_1_v4/bowl*.obj') if fn not in remove_list]
        mug_list = [fn for fn in glob.glob('data/NOCS_val/32_64_1_v4/mug*.obj') if fn not in remove_list]
        can_list = [fn for fn in glob.glob('data/NOCS_val/32_64_1_v4/can*.obj') if fn not in remove_list]
        OBJ_FILE_PATH_LIST = list(np.random.choice(bowl_list, size=10, replace=False)) + \
            list(np.random.choice(bottle_list, size=10, replace=False)) + \
            list(np.random.choice(mug_list, size=10, replace=False))
            # list(np.random.choice(can_list, size=3, replace=False))
        # OBJ_FILE_PATH_LIST = glob.glob('data/NOCS_val/32_64_1_v4/*.obj')

        # Main loop
        # PB needs reset. Max vis shape sucks.
        # Simulation initialization

        self.panda_gripper = PandaGripperVisual(self.bc, Path("imm_pb_util")/"urdf"/"panda_gripper"/"panda_gripper_visual.urdf")
        self.panda_gripper.reset_finger(0.04)
        self.panda_gripper.freeze()
        self.panda_gripper.transparent(0.7)
        self.panda_gripper.set_gripper_pose(np.array([0,0,5.0]), np.array([0,0,0,1]))
        
        env = SimPnPEnv(self.bc, env_type=self.env_type)

        cameras = [SimpleCamera(i, self.bc, config) for i in range(0, 3)]
        # nv_renderer = BaseNVRenderer(bc, show_gui=False)
        # Visualize camera frames
        for cam in cameras:
            cam.show_camera_pose()
        # Stabilize
        for i in range(100):
            self.bc.stepSimulation()

        self.cameras = cameras

        # Define start and goal position
        # object_region = np.array([-0.15, -0.15, 0.05]), np.array([0.15, 0.15, 0.10])
        if self.env_type == 'table':
            object_region = np.array([-0.2, -0.2, 0.05]), np.array([0.2, 0.2, 0.15])
        elif self.env_type == 'shelf':
            object_region = np.array([-0.15, -0.32, 0.05]), np.array([0.15, 0.32, 0.15])
        
        # Spawn random object
        pb_obj_list, pb_obj_path_list, pb_obj_scale_list \
            = spawn_random_collision_free_objects_sim_grasp(self.bc, OBJ_FILE_PATH_LIST, NUM_OBJ, object_region)

        # Stabilize
        step(self.bc, 2000)
        for pb_obj in pb_obj_list:
            self.bc.changeDynamics(pb_obj.uid, -1, lateralFriction=0.6)

        # Check any dropped object
        pos_list = []
        quat_list = []
        for pb_obj, path, scale in zip(pb_obj_list, pb_obj_path_list, pb_obj_scale_list):
            pos, quat = self.bc.getBasePositionAndOrientation(pb_obj.uid)
            pos_list.append(pos)
            quat_list.append(quat)

        self.robot = FrankaPanda(self.bc, config)
        for lidx in self.robot.finger_indices:
            self.bc.changeDynamics(self.robot.uid, lidx, rollingFriction=0.7, spinningFriction=0.7, lateralFriction=1.6)
        self.init_q = self.robot.rest_pose[self.robot.nonfixed_joint_indices_arm]


    def take_images(self, padding=0, visualize=False):

        # Capture images
        pb_image_list = []
        pb_depth_list = []
        cam_intrinsics = []
        cam_posquats = []
        for cam in self.cameras:
            rgb, depth, seg = cam.capture_rgbd_image()
            intrinsics = cam.intrinsic.formatted # (W, H, Fx, Fy, Cx, Cy)
            pos = cam.extrinsic.camera_pos
            quat = cam.extrinsic.camera_quat
            cam_intrinsics.append(intrinsics)
            cam_posquats.append(np.concatenate([pos, quat]))
            pb_image_list.append(rgb)
            pb_depth_list.append(depth)
        for _ in range(padding-len(self.cameras)):
            pb_image_list.append(np.zeros_like(pb_image_list[0]))
            cam_intrinsics.append(cam_intrinsics[0])
            cam_posquats.append(cam_posquats[0])
        pb_image_list = np.array(pb_image_list)
        cam_intrinsics = np.array(cam_intrinsics)
        cam_posquats = np.array(cam_posquats)

        if visualize:
            fig, axs = plt.subplots(1, 3, figsize=(15, 5))
            for i, (ax, img) in enumerate(zip(axs, [pb_image_list[0], pb_image_list[1], pb_image_list[2]])):
                ax.imshow(img)    
                ax.set_title(f"Camera {i}")
            plt.tight_layout()
            plt.show()

        return pb_image_list, cam_intrinsics, cam_posquats

    def hand_take_image(self):
        # take picture
        eepos, eequat = self.robot.get_endeffector_pose()
        camera_offset = np.array([0.057373, 0.0, 0.010]) # should be fixed
        cam_pos, cam_quat = pb.multiplyTransforms(eepos, eequat, camera_offset, (sciR.from_euler('x', np.pi)*sciR.from_euler('z', -np.pi/2)).as_quat())
        intrinsic = self.cameras[0].intrinsic
        gl_view_matrix = compute_gl_view_matrix_from_gl_camera_pose(
                bc  = self.robot.bc, 
                pos = cam_pos,
                orn = cam_quat)

        (w, h, px, px_d, px_id) = self.robot.bc.getCameraImage(
            width            = intrinsic.width,
            height           = intrinsic.height,
            viewMatrix       = gl_view_matrix,
            projectionMatrix = intrinsic.gl_proj_matrix,
            renderer         = self.robot.bc.ER_BULLET_HARDWARE_OPENGL)
        
        rgb_array   = np.array(px, dtype=np.uint8)
        rgb_array   = rgb_array[:, :, :3]  
        return rgb_array, np.concatenate([cam_pos, cam_quat], -1), intrinsic.formatted

    def visualize_pred_obj(self, obj_preds, models, id):
        # load mesh
        if id == 0:
            color = [0.4,0.4,0.4,0.5]
        else:
            color = [0.9,0.6,0.2,0.5]
        objs_uid = []
        for i in range(obj_preds.outer_shape[0]):
            obj_ = jax.tree_util.tree_map(lambda x: x[i], obj_preds)
            if np.all(np.abs(obj_.pos) < 1.0):
                mesh_pred = create_mesh_from_latent(jkey, models, obj_.translate(-obj_.pos), density=128, visualize=False)
                o3d.io.write_triangle_mesh(f'tmp/latent_mesh{id}_{i}.obj', mesh_pred)
                obj_id = self.bc.createMultiBody(baseMass=0.0, baseVisualShapeIndex=pb.createVisualShape(pb.GEOM_MESH, fileName=f'tmp/latent_mesh{id}_{i}.obj', rgbaColor=color))
                objs_uid.append(obj_id)
                self.bc.resetBasePositionAndOrientation(obj_id, np.array(obj_.pos), [0,0,0,1])
                print(f'{i} loaded')
        return objs_uid

import util.pb_util as pbutil
from imm_pb_util.bullet_envs.objects import BulletObject
def spawn_random_collision_free_objects_sim_grasp(
        bc,
        obj_file_path_list: List[Path],
        num_obj: int,
        object_region: Tuple[np.ndarray, np.ndarray] ,
        trial: int = 5000
):
    """?"""
    
    obj_path_list = np.random.choice(obj_file_path_list, num_obj)
    pb_obj_list = []
    pb_obj_path_list = []
    pb_obj_scale_list = []
    pb_obj_color_list = []
    for fp in obj_path_list:
        base_len = 0.65
        scale_ = 1/base_len
        categorical_scale = {'can':1/base_len*0.15, 'bottle':1/base_len*0.15, 'bowl':1/base_len*0.2,
                                'camera':1/base_len*0.15, 'laptop':1/base_len*0.3, 'mug':1/base_len*0.12}
        # categorical_scale = {'can':1/base_len*0.08, 'bottle':1/base_len*0.11, 'bowl':1/base_len*0.15,
        #                         'camera':1/base_len*0.09, 'laptop':1/base_len*0.3, 'mug':1/base_len*0.12}
        
        categorical_orn_ratio = {'can':0.6, 'bottle':0.9, 'bowl':0,
                                'camera':0, 'laptop':1, 'mug':0} # ratio of layed object
        
        for key, value in categorical_scale.items():
            if key in fp:
                scale_ = value
                ratio_ = categorical_orn_ratio[key]

        # scale_ = scale_*np.random.uniform(0.6, 0.8, size=(1,))
        scale_ = scale_*np.random.uniform(0.7, 1.3, size=(1,))
        scale_ = scale_*np.ones(3)

        color = pbutil.looking_good_color()
        uid = pbutil.create_pb_multibody_from_file(bc, str(fp), [0,0,0], [0,0,0,1], scale=scale_, color=color, use_visual_mesh=True)
        for i in range(trial):
            pos_rand = np.random.uniform(*object_region, size=(3,))
            # TODO: stochasticity for stand objects
            obj_quat = np.where(np.random.uniform() > ratio_, np.array([0.707, 0, 0, 0.707]), np.array([0, 0, 0, 1]))
            z_ang = sciR.from_euler('z', np.random.uniform(-np.pi, np.pi, size=()))*sciR.from_quat(obj_quat)
            ori_rand = z_ang.as_quat()
            bc.resetBasePositionAndOrientation(uid, pos_rand, ori_rand)
            bc.performCollisionDetection()
            cp_res = bc.getContactPoints(uid)
            if len(cp_res) == 0:
                break
        pb_obj = BulletObject(bc, uid)
        pb_obj_list.append(pb_obj)
        pb_obj_path_list.append(fp)
        pb_obj_scale_list.append(scale_)
        pb_obj_color_list.append(color)

    return pb_obj_list, pb_obj_path_list, pb_obj_scale_list

    
if __name__=="__main__":

    jkey = jax.random.PRNGKey(0)

    # Open yaml config file
    # ENV_TYPE = 'table'
    ENV_TYPE = 'shelf'
    # config_file_path = Path(__file__).parent.parent / 'examples/pb_examples' / "pb_cfg" / "pb_sim_pnp.yaml"
    config_file_path = Path(__file__).parent.parent / 'examples/pb_examples' / "pb_cfg" / "pb_sim_pnp_shelf.yaml"
    with open(config_file_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # Main
    pnp_env = PnPEnv(config, ENV_TYPE)
    robot = pnp_env.robot



    # logs_dir = 'logs_pnp/04162024-191008'
    logs_dir = None
    language_guided_pnp = LanguageGuidedPnP(np.array(pnp_env.robot.base_pos), np.array(pnp_env.robot.base_orn), env_type=ENV_TYPE, 
                                            scene_obj_no=6, logs_dir=logs_dir, cam_posquats=None)
    
    # virtual_world.register_latent_obj(obj_pred, language_guided_pnp.models, jkey)
    sim_dt = pb.getPhysicsEngineParameters()['fixedTimeStep']
    def follow_traj(traj_):
        traj_dt = 0.04
        for q in traj_:
            pnp_env.robot.update_arm_control(q)
            for _ in range(int(traj_dt/sim_dt)):
                pb.stepSimulation()
            time.sleep(traj_dt)

    for _ in range(100):
        pnp_env.bc.resetSimulation()
        pnp_env.reset_env()
        robot.reset(jnp.array([0,-np.pi/2,0,0,0,0,0.]))
        pnp_env.panda_gripper.set_gripper_pose(np.array([0,0,5.0]), np.array([0,0,0,1]))
        # robot.reset(exutil.IMG_TANKING_Q)
        images, intrinsics, cam_posquats = pnp_env.take_images(padding=3, visualize=False)
        obj_pred, language_aligned_obj_feat, clip_aux_info, inf_aux_info = language_guided_pnp.estimation(images, cam_posquats, intrinsics, jkey)
        _, jkey = jax.random.split(jkey)
        robot.reset(exutil.RRT_INITIAL_Q)

        # skill 1
        target_obj_text = 'a bowl'
        goal_obj_text = None
        goal_obj_text2 = None
        goal_relation_type = None
        # traj_list, final_grasp_pq = language_guided_pnp.pnp_to_with_relation(obj_pred, language_aligned_obj_feat, 
        #                          target_obj_text, goal_obj_text, goal_relation_type, pnp_env.init_q, jkey, clip_aux_info, goal_obj_text2=goal_obj_text2, rrt=1)
        # with open(os.path.join(language_guided_pnp.logs_dir, f'skil1_{target_obj_text}.pkl'), 'wb') as f:
        #     pickle.dump(traj_list, f)

        # with open(os.path.join(language_guided_pnp.logs_dir, f'skil1_{target_obj_text}.pkl'), 'rb') as f:
        #     traj_list = pickle.load(f)
        valid_target_mask = jnp.ones((obj_pred.outer_shape[0],), dtype=jnp.bool_)
        jkey, subkey = jax.random.split(jkey)
        pregrasp_traj, initial_grasp_pq, pre_grasp_aux_info, target_obj, target_idx, target_text_query_aux_info = language_guided_pnp.pregrasp_traj_with_target(obj_pred, language_aligned_obj_feat, target_obj_text, pnp_env.init_q, subkey, 
                                                                                                                        clip_aux_info=clip_aux_info, inf_aux_info=inf_aux_info, valid_target_mask=valid_target_mask)
        valid_target_mask = valid_target_mask.at[target_idx].set(False)

        if jnp.any(jnp.abs(obj_pred.pos[target_idx]) > 8.0):
            print('target object is not in the scene')
            continue
        
        follow_traj(pregrasp_traj)
        # second grasp
        # intermediate_traj = np.stack([IMG_TANKING_Q, np.array([ 0.1221, -0.5215, -0.0480, -0.8196, -0.1164,  1.4949,  0.8300])], 0)
        # intermediate_traj = IMG_TANKING_Q[None]
        intermediate_traj = None
        jkey, subkey = jax.random.split(jkey)
        grasp_traj, pq_oe, grasp_q, grasp_pq, grasp_aux_info = language_guided_pnp.grasp_motion(obj_pred, target_idx, pregrasp_traj[-1], subkey, inf_aux_info=inf_aux_info)


        # grasp_traj, pq_oe, grasp_q, grasp_pq, target_idx = language_guided_pnp.grasp_motion(obj_pred, language_aligned_obj_feat, target_obj_text, pnp_env.init_q, jkey, clip_aux_info=clip_aux_info)
        robot.reset(grasp_traj[0])
        pnp_env.panda_gripper.set_gripper_pose(grasp_pq[0], grasp_pq[1])
        follow_traj(grasp_traj)
        robot.close_gripper(sleep=True)


        jkey, subkey = jax.random.split(jkey)
        grasp_to_place_traj = language_guided_pnp.place_to_target(target_idx, obj_pred, language_aligned_obj_feat, goal_obj_text, goal_obj_text2, 
                                            goal_relation_type, pq_oe, grasp_q, pnp_env.init_q, jkey, intermediate_traj=None, clip_aux_info=clip_aux_info)
        # follow_traj(grasp_to_place_traj)

        traj_list = grasp_to_place_traj[:-1]
        follow_traj(traj_list[0])
        follow_traj(traj_list[1])


        # approach grasp predictions
        # jkey, subkey = jax.random.split(jkey)
        # pregrasp_traj, initial_grasp_pq, target_obj, target_idx1 = language_guided_pnp.pregrasp_traj_with_target(obj_pred, language_aligned_obj_feat, target_obj_text, exutil.IMG_TANKING_Q, subkey, clip_aux_info=clip_aux_info)

        # # move
        # robot.reset(pregrasp_traj[0])
        # follow_traj(pregrasp_traj)

        # hand_img, hand_cam_posquat, hand_cam_intrinsic = pnp_env.hand_take_image()
        # pnp_env.panda_gripper.set_gripper_pose(initial_grasp_pq[0], initial_grasp_pq[1])
        # pred_pb_obj1 = pnp_env.visualize_pred_obj(target_obj.extend_outer_shape(0), language_guided_pnp.models, id=0)

        # images[-1] = hand_img
        # cam_posquats[-1] = hand_cam_posquat
        # intrinsics[-1] = hand_cam_intrinsic

        # jkey, subkey = jax.random.split(jkey)
        # inf_outputs = language_guided_pnp.inf_cls.estimation(jkey, language_guided_pnp.models.pixel_size, images, intrinsics, cam_posquats, previous_obj_pred=obj_pred,
        #                                                     out_valid_obj_only=False, filter_before_opt=True, verbose=1, plane_params=language_guided_pnp.plane_params[0][:1])
        # obj_pred2 = inf_outputs[0]
        # target_idx2 = jnp.argmin(jnp.linalg.norm(target_obj.pos - obj_pred2.pos, axis=-1))
        # target_obj2 = jax.tree_util.tree_map(lambda x: x[target_idx2], obj_pred2)

        # obj_pred = jax.tree_util.tree_map(lambda x,y: x.at[target_idx1].set(y[target_idx2]), obj_pred, obj_pred2)

        # pred_pb_obj2 = pnp_env.visualize_pred_obj(target_obj2.extend_outer_shape(0), language_guided_pnp.models, id=1)

        # # second grasp
        # intermediate_traj = np.stack([pnp_env.init_q, pnp_env.init_q-np.array([0,0.5,0,0,0,0,0])], 0)
        # traj_list, final_grasp_pq = language_guided_pnp.postgrasp_pnp_with_relation(obj_pred, language_aligned_obj_feat, 
        #                         target_idx1, goal_obj_text, goal_relation_type, pregrasp_traj[-1], pnp_env.init_q, jkey, 
        #                         intermediate_traj=intermediate_traj, clip_aux_info=None, goal_obj_text2=None, rrt=1)
        




        # ## skil 2
        # target_obj_text = 'a white bowl'
        # goal_obj_text = 'a pink bowl'
        # # goal_obj_text2 = 'a blue bowl'
        # goal_obj_text2 = None
        # # goal_relation_type = 'between'
        # goal_relation_type = 'up'
        # # traj_list = language_guided_pnp.pnp_to_with_relation(obj_pred, language_aligned_obj_feat, 
        # #                          target_obj_text, goal_obj_text, goal_relation_type, pnp_env.init_q, jkey, clip_aux_info, goal_obj_text2=goal_obj_text2)
        # # with open(os.path.join(language_guided_pnp.logs_dir, f'skil2_{target_obj_text}_{goal_obj_text}_{goal_relation_type}.pkl'), 'wb') as f:
        # #     pickle.dump(traj_list, f)

        # with open(os.path.join(language_guided_pnp.logs_dir, f'skil2_{target_obj_text}_{goal_obj_text}_{goal_relation_type}.pkl'), 'rb') as f:
        #     traj_list = pickle.load(f)


        ## skil 3
        # target_category = 'bowl'
        # category_per_obj = language_guided_pnp.category_overlay(language_aligned_obj_feat)
        # traj_list = language_guided_pnp.rearrange_categorical_objects(obj_pred, language_aligned_obj_feat, category_per_obj, target_cat=target_category, rest_q = pnp_env.init_q, jkey=jkey)
        # with open(os.path.join(language_guided_pnp.logs_dir, f'cat_pnp_traj_res_{target_category}.pkl'), 'wb') as f:
        #     pickle.dump(traj_list, f)

        # with open(os.path.join(language_guided_pnp.logs_dir, f'cat_pnp_traj_res_{target_category}.pkl'), 'rb') as f:
        #     traj_list = pickle.load(f)

        ## skill 4
        # target_obj_text = 'a white bowl'
        # goal_obj_text = 'a pink bowl'
        # traj_list = language_guided_pnp.pour_water(obj_pred, language_aligned_obj_feat, target_obj_text, goal_obj_text, pnp_env.init_q, jkey, clip_aux_info=clip_aux_info)
        # with open(os.path.join(language_guided_pnp.logs_dir, f'skill4_{target_obj_text}_{goal_obj_text}.pkl'), 'wb') as f:
        #     pickle.dump(traj_list, f)

        # Execute
        
        # for traj_ in traj_list:
        #     traj1, traj2, traj3 = traj_
        #     print('motion solution')
        #     robot.reset(traj1[0])
        #     robot.reset_finger(robot.finger_travel/2.0)
        #     follow_traj(traj1)
        #     robot.close_gripper(sleep=True)
        #     follow_traj(traj2)
        #     robot.release()
        #     for _ in range(int(0.5/sim_dt)):
        #         pb.stepSimulation()
        #     follow_traj(traj3)


        ### skill 3 - target fetching
        # target_obj_text = 'a bowl for soup'
        # planning_output = language_guided_pnp.fetch_target_obj(target_obj_text, obj_pred, language_aligned_obj_feat, 
        #                                                        pnp_env.init_q, pnp_env.init_q, jkey, clip_aux_info, visualize=False)

        # language_guided_pnp.robot_execution_in_sim(pnp_env.robot, planning_output)


