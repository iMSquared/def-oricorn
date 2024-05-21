from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
import torch
import random
import pybullet as pb
from pybullet_utils.bullet_client import BulletClient
from scipy.spatial.transform import Rotation as sciR
from copy import deepcopy
import open3d as o3d
import einops
import os, pickle
from dataclasses import dataclass
import time

from typing import Dict, Tuple, List, Optional, NamedTuple

import yaml
import util.io_util as ioutil
import util.pb_util as pbutil
import util.cvx_util as cxutil
import util.model_util as mutil
import util.transform_util as tutil
import util.inference_util as ifutil
import logging

# Environment
from imm_pb_util.imm.pybullet_util.common import get_link_pose, imagine
from imm_pb_util.imm.pybullet_util.typing_extra import TranslationT
from imm_pb_util.imm.pybullet_util.vision import CoordinateVisualizer
from imm_pb_util.bullet_envs.env import SimpleEnv
from imm_pb_util.bullet_envs.robot import FrankaPanda, PandaGripper
from imm_pb_util.bullet_envs.manipulation import FrankaManipulation, get_linear_interpolated_trajectory
from imm_pb_util.bullet_envs.objects import BulletObject
from imm_pb_util.bullet_envs.camera import SimpleCamera, RobotAttachedCamera
from imm_pb_util.bullet_envs.nvisii import BaseNVRenderer
from imm_pb_util.bullet_envs.env import BulletEnv

# ?
from examples.pb_examples.common.common import configure_bullet, change_dynamics_slippery, change_dynamics_stiff, step


class SimGraspEnv(BulletEnv):
    """
    ?
    """
    def __init__(self, bc: BulletClient, config: Dict):
        """Load ground plane.

        Args:
            bc (BulletClient): _
        """
        super().__init__(bc, add_plane=False)

        table_id = bc.createMultiBody(
                baseMass = 0.0, 
                basePosition = [0,0,-0.1],
                baseCollisionShapeIndex = bc.createCollisionShape(bc.GEOM_BOX, halfExtents=[0.4, 0.4, 0.1]),
                baseVisualShapeIndex = bc.createVisualShape(bc.GEOM_BOX, halfExtents=[0.4, 0.4, 0.1]))
            #   Table height randomization
        bc.resetBasePositionAndOrientation(
            table_id, 
            posObj = [0,0,-0.1], 
            ornObj = [0,0,0,1])
        # bc.changeVisualShape(table_id, -1, rgbaColor=[0.9670298390136767, 0.5472322491757223, 0.9726843599648843, 1.0])
        bc.changeVisualShape(table_id, -1, rgbaColor=[0.1, 0.1, 0.1, 1.0])

        # Register
        self.env_assets['table'] = table_id

    @property
    def env_uids(self) -> List[int]:
        return list(self.env_assets.values())



@dataclass
class PybulletStateGrasp:
    """Class to manage pybullet conveniently"""
    bc: BulletClient
    env: SimGraspEnv
    obj_list: List[BulletObject]
    cameras: List[SimpleCamera]
    nv_renderer: BaseNVRenderer
    config: Dict

    @classmethod
    def init_from_config(cls, config):
        bc = configure_bullet(config)
        env, cameras, nv_renderer = initialize_grasp_domain(bc, config)
        obj_list = []

        return cls(bc, env, obj_list, cameras, nv_renderer, config)
    

    def reset_hard(self):
        """Hard reset"""
        self.bc.disconnect()
        bc = configure_bullet(self.config)
        env, cameras, nv_renderer = initialize_grasp_domain(bc, self.config)
        obj_list = []
        self.bc = bc
        self.env = env
        self.cameras = cameras
        self.nv_renderer = nv_renderer
        self.obj_list = obj_list


    def reset_soft(self):
        """Clear environment + objects"""
        raise NotImplementedError()


    def reset_objects(self):
        """Soft reset"""
        for pb_obj in self.obj_list:
            self.bc.removeBody(pb_obj.uid)
        # self.bc.removeAllUserDebugItems()
        self.obj_list = []


    def reset_hard_periodically(self, counter: int, period: int):
        """I hate pybullet visual shape bug"""
        if counter != 0 and counter % period == 0:
            self.reset_hard()
            return True
        else:
            return False


def initialize_grasp_domain(bc: BulletClient, config):
    """Initialize skeleton environment for grasp problem. No objects are initialized"""
    env = SimGraspEnv(bc, config)
    cameras = [SimpleCamera(i, bc, config) for i in range(0, 3)]
    nv_renderer = BaseNVRenderer(bc, show_gui=False)
    # Visualize camera frames
    for cam in cameras:
        cam.show_camera_pose()

    return env, cameras, nv_renderer


@dataclass
class GraspObjectConfig:
    mesh_name: str
    scale: np.ndarray
    pos: np.ndarray
    quat: np.ndarray
    color: np.ndarray


def load_grasp_problem_from_config(
        pb_state: PybulletStateGrasp,
        config_dir_path: Path,
        obj_dir_path: Path,
        i: int,
):
    """Load objects in the environment"""
    # Open
    print((config_dir_path/f"sim_grasp_{i}.yaml"))
    with (config_dir_path/f"sim_grasp_{i}.yaml").open("r") as f:
        data = yaml.safe_load(f)
        objs = data["objects"]
        target_idx = int(data["target_idx"])

    pb_obj_list = []
    obj_config_list: list[GraspObjectConfig] = []
    for idx, obj in enumerate(objs):
        # Parse
        mesh_name = obj["obj"]
        scale = np.asarray(obj["scale"])
        pos = np.asarray(obj["pos"])
        quat = np.asarray(obj["quat"])
        color = np.asarray(obj["color"])
        obj_path = obj_dir_path/mesh_name
        # Load obj
        uid = pbutil.create_pb_multibody_from_file(
            pb_state.bc, str(obj_path), pos, quat, scale=scale, color=color)
        pb_obj = BulletObject(pb_state.bc, uid)
        # Compose config
        obj_config = GraspObjectConfig(mesh_name, scale, pos, quat, color)
        # Aggregate
        pb_obj_list.append(pb_obj)
        obj_config_list.append(obj_config)
    # Update state
    pb_state.obj_list = pb_obj_list
    
    return target_idx, obj_config_list


def save_grasp_problem_to_config(
        config_dir_path,
        i,
        target_idx,
        pb_obj_pos_list,
        pb_obj_quat_list,
        pb_obj_path_list,
        pb_obj_scale_list,
        pb_obj_color_list,
):
    items = []
    loop = zip(pb_obj_pos_list, pb_obj_quat_list, pb_obj_path_list, pb_obj_scale_list, pb_obj_color_list)
    for pos, quat, path, scale, color in loop:
        # pos, quat = bc.getBasePositionAndOrientation(pb_obj.uid)
        item = {
            "obj": Path(path).name,
            "scale": scale.tolist(),
            "pos": list(pos),
            "quat": list(quat),
            "color": color.tolist()
        }
        items.append(item)

    data = {
        "target_idx": int(target_idx),
        "objects": items,
    }

    data_yaml_str = yaml.dump(data)
    filename = f"sim_grasp_{i}.yaml"
    with (config_dir_path/filename).open("w") as f:
        f.write(data_yaml_str)
    

def grasp(
        pb_state: PybulletStateGrasp, 
        pb_obj_gt_target: BulletObject, 
        virtual_base_pq: tuple[jnp.ndarray, jnp.ndarray]
) -> bool:

    # Reset initial pose
    panda_gripper = PandaGripper(pb_state.bc, Path("imm_pb_util")/"urdf"/"panda_gripper"/"panda_gripper_prismatic.urdf")
    # panda_gripper.set_gripper_pose(virtual_base_pq[0], tutil.qmulti(virtual_base_pq[1], tutil.aa2q(jnp.array([0,0,-np.pi/4]))))
    panda_gripper.set_gripper_pose(virtual_base_pq[0], virtual_base_pq[1])
    panda_gripper.reset_z(-0.2)
    step(pb_state.bc, 500)
    panda_gripper.release()
    step(pb_state.bc, 500)
    
    # Approach
    N_INTER = 7000
    for i in range(N_INTER+1):
        z = -0.2 + (i/N_INTER * 0.2)
        panda_gripper.set_z(z)
        step(pb_state.bc, 1)
    step(pb_state.bc, 500)

    # Grasp (smooth trajectory)
    joint_q_start = [0.04, 0.04]
    joint_q_end = [0.0, 0.0]
    traj = get_linear_interpolated_trajectory(joint_q_start, joint_q_end, action_duration=1, control_dt=0.001)
    for q in traj:
        panda_gripper.last_pose[panda_gripper.finger_indices] = q
        panda_gripper.activate()
        step(pb_state.bc, 1)
    step(pb_state.bc, 1000)

    # Backward
    for i in range(N_INTER+1):
        z = -i/N_INTER * 1.0
        panda_gripper.set_z(z)
        step(pb_state.bc, 1)

    # Stabilize
    step(pb_state.bc, 1000)

    # Check grasp height
    target_pos, _ = pb_state.bc.getBasePositionAndOrientation(pb_obj_gt_target.uid)
    if target_pos[2] < 0.5:
        print("fail")
        pb_state.bc.removeBody(panda_gripper.uid)
        return False
    
    print("success")
    pb_state.bc.removeBody(panda_gripper.uid)
    return True





def spawn_random_collision_free_objects_sim_grasp(
        pb_state: PybulletStateGrasp,
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
        # categorical_scale = {'can':1/base_len*0.15, 'bottle':1/base_len*0.15, 'bowl':1/base_len*0.2,
        #                         'camera':1/base_len*0.15, 'laptop':1/base_len*0.3, 'mug':1/base_len*0.12}
        categorical_scale = {'can':1/base_len*0.08, 'bottle':1/base_len*0.11, 'bowl':1/base_len*0.15,
                                'camera':1/base_len*0.09, 'laptop':1/base_len*0.3, 'mug':1/base_len*0.12}
        
        categorical_orn_ratio = {'can':0.6, 'bottle':0.9, 'bowl':0,
                                'camera':0, 'laptop':1, 'mug':0} # ratio of layed object
        
        for key, value in categorical_scale.items():
            if key in fp:
                scale_ = value
                ratio_ = categorical_orn_ratio[key]

        scale_ = scale_*np.random.uniform(0.6, 0.8, size=(1,))
        scale_ = scale_*np.ones(3)

        color = pbutil.looking_good_color()
        uid = pbutil.create_pb_multibody_from_file(pb_state.bc, str(fp), [0,0,0], [0,0,0,1], scale=scale_, color=color)
        for i in range(trial):
            pos_rand = np.random.uniform(*object_region, size=(3,))
            # TODO: stochasticity for stand objects
            obj_quat = np.where(np.random.uniform() > ratio_, np.array([0.707, 0, 0, 0.707]), np.array([0, 0, 0, 1]))
            z_ang = sciR.from_euler('z', np.random.uniform(-np.pi, np.pi, size=()))*sciR.from_quat(obj_quat)
            ori_rand = z_ang.as_quat()
            pb_state.bc.resetBasePositionAndOrientation(uid, pos_rand, ori_rand)
            pb_state.bc.performCollisionDetection()
            cp_res = pb_state.bc.getContactPoints(uid)
            if len(cp_res) == 0:
                break
        pb_obj = BulletObject(pb_state.bc, uid)
        pb_obj_list.append(pb_obj)
        pb_obj_path_list.append(fp)
        pb_obj_scale_list.append(scale_)
        pb_obj_color_list.append(color)

    change_dynamics_stiff(pb_state.bc, pb_obj_list)
    pb_state.obj_list = pb_obj_list    

    return obj_path_list, pb_obj_scale_list, pb_obj_color_list



@dataclass
class GraspItem:
    best_pq: tuple[np.ndarray, np.ndarray]
    approaching_pq: tuple[np.ndarray, np.ndarray]
    virtual_base_pq: tuple[np.ndarray, np.ndarray]

    def dump_grasp(self, logs_dir_path: Path, idx: int):
        """Dump grasp to numpy for replay"""
        out_file_name = logs_dir_path/f"grasp_{idx}.npz"
        with out_file_name.open("wb") as f:
            np.savez_compressed(
                file = f, 
                best_pos = self.best_pq[0],
                best_quat = self.best_pq[1],
                approaching_pos = self.approaching_pq[0],
                approaching_quat = self.approaching_pq[1],
                virtual_base_pos = self.virtual_base_pq[0],
                virtual_base_quat = self.virtual_base_pq[1],
            )
    
    @classmethod
    def load_grasp(cls, grasp_file_path: Path):
        """Load grasp for replay"""
        with grasp_file_path.open("rb") as f:
            loaded = np.load(f)
            best_pos = loaded["best_pos"]
            best_quat = loaded["best_quat"]
            approaching_pos = loaded["approaching_pos"]
            approaching_quat = loaded["approaching_quat"]
            virtual_base_pos = loaded["virtual_base_pos"]
            virtual_base_quat = loaded["virtual_base_quat"]
        item = GraspItem(
            (best_pos, best_quat),
            (approaching_pos, approaching_quat),
            (virtual_base_pos, virtual_base_quat)
        )
        return item

    def __str__(self):
        return f"best_pos={self.best_pq[0].tolist()}, best_quat={self.best_pq[1].tolist()}"



@dataclass
class GraspState:
    grasp_item: GraspItem
    success: bool


@dataclass
class GraspResultState:

    grasp_state_records: dict[int, GraspState]

    def __init__(self):
        self.grasp_state_records = {}

    def update(self, idx: int, grasp_state: GraspState):
        self.grasp_state_records[idx] = grasp_state

    def summarize(self):
        
        failure_indices = [idx for idx, state in self.grasp_state_records.items() if state.success==False]
        num_success = np.sum([state.success for state in self.grasp_state_records.values()])
        num_total = len(self.grasp_state_records)
        print(f"success: {num_success}/{num_total} ({num_success/num_total*100:.1f}%)")
        print(f"failed testset indices: {failure_indices}")
        logging.info(f"success: {num_success}/{num_total} ({num_success/num_total*100:.1f}%)")
        logging.info(f"failed testset indices: {failure_indices}")