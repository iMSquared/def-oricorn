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
import trimesh
import open3d as o3d
import einops
import os, pickle
from dataclasses import dataclass

from typing import Dict, Tuple, List, Optional, NamedTuple

import yaml
import util.io_util as ioutil
import util.pb_util as pbutil
import util.cvx_util as cxutil
import util.model_util as mutil
import util.transform_util as tutil

# Environment
from imm_pb_util.imm.pybullet_util.common import get_link_pose, imagine
from imm_pb_util.imm.pybullet_util.typing_extra import TranslationT
from imm_pb_util.imm.pybullet_util.vision import CoordinateVisualizer
from imm_pb_util.bullet_envs.env import SimpleEnv
from imm_pb_util.bullet_envs.robot import FrankaPanda, PandaGripper
from imm_pb_util.bullet_envs.manipulation import FrankaManipulation
from imm_pb_util.bullet_envs.objects import BulletObject
from imm_pb_util.bullet_envs.camera import SimpleCamera, RobotAttachedCamera, DebugCamera
from imm_pb_util.bullet_envs.nvisii import BaseNVRenderer
from imm_pb_util.bullet_envs.env import SimGraspEnv

# ?
from examples.pb_examples.common.common import (
    step,
    configure_bullet, 
    change_dynamics_slippery, 
    change_dynamics_stiff,
    draw_box
)


@dataclass
class PybulletStateEstimation:
    """asdf"""
    bc: BulletClient
    env: SimpleEnv
    obj_list: List[BulletObject]
    cameras: List[SimpleCamera]
    nv_renderer: BaseNVRenderer
    config: Dict
    debug_camera: DebugCamera

    @classmethod
    def init_from_config(cls, config):
        # Simulation initialization
        bc = configure_bullet(config)
        env, cameras, nv_renderer = init_estimation_domain(bc, config)
        debug_camera = DebugCamera.from_debugger(bc)
        return cls(bc, env, [], cameras, nv_renderer, config, debug_camera)
    
    
    def reset_hard(self):
        """Hard reset"""
        # Simulation initialization
        self.bc.disconnect()
        bc = configure_bullet(self.config)
        env, cameras, nv_renderer = init_estimation_domain(bc, self.config)
        debug_camera = DebugCamera.from_debugger(bc)
        self.bc = bc
        self.env = env
        self.obj_list = []
        self.cameras = cameras
        self.nv_renderer = nv_renderer
        self.debug_camera = debug_camera


    def reset_objects(self):
        """Soft reset"""
        for pb_obj in self.obj_list:
            self.bc.removeBody(pb_obj.uid)
        self.bc.removeAllUserDebugItems()
        self.obj_list = []


    def reset_hard_periodically(self, counter: int, period: int):
        """I hate pybullet visual shape bug"""
        if counter != 0 and counter % period == 0:
            self.reset_hard()



def get_object_region():
    """Define object spawn region"""
    box_corner_A = np.array([0.1, 0.15, 0.2])
    box_corner_B = np.array([-0.1, -0.15, 0.25])
    region = (box_corner_A, box_corner_B)

    return region



def init_estimation_domain(bc, config):
    """Configure basic properties for estimation domain"""

    # Items...
    env = SimpleEnv(bc, config)
    cameras = [SimpleCamera(i, bc, config) for i in range(0, 3)]
    nv_renderer = BaseNVRenderer(bc, show_gui=False)
    # Visualize camera frames
    for cam in cameras:
        cam.show_camera_pose()
    # Stabilize
    for i in range(100):
        bc.stepSimulation()

    return env, cameras, nv_renderer


@dataclass
class ObjectsConfiguration:
    pb_obj_list: List[BulletObject]
    obj_path_list: List[Path]
    obj_scale_list: List[np.ndarray]
    obj_color_list: List[np.ndarray]


def save_configuration_to_yaml(
        output_file_path: Path,
        bc: BulletClient,
        obj_configs: ObjectsConfiguration
):
    items = []
    loop = zip(obj_configs.pb_obj_list, obj_configs.obj_path_list, obj_configs.obj_scale_list, obj_configs.obj_color_list)
    for pb_obj, path, scale, color in loop:
        pos, quat = bc.getBasePositionAndOrientation(pb_obj.uid)
        item = {
            "obj": Path(path).name,
            "scale": scale.tolist(),
            "color": color.tolist(),
            "pos": list(pos),
            "quat": list(quat),}
        items.append(item)

    data = {
        "objects": items
    }

    data_yaml_str = yaml.dump(data)
    with output_file_path.open("w") as f:
        f.write(data_yaml_str)


def load_yaml_to_configuration(
        input_file_path: Path,
        obj_dir_path: Path,
        bc: BulletClient, 
):
    # Open
    print(str(input_file_path))
    with input_file_path.open("r") as f:
        data = yaml.safe_load(f)
        objs = data["objects"]
    
    pb_obj_list =[]
    for obj in objs:
        name = obj["obj"]
        scale = np.array(obj["scale"])
        color = np.array(obj["color"])
        pos = np.array(obj["pos"])
        quat = np.array(obj["quat"])
        obj_path = obj_dir_path/name
        uid = pbutil.create_pb_multibody_from_file(
            bc, str(obj_path), pos, quat, scale=scale, color=color, fixed_base=True)
        pb_obj = BulletObject(bc, uid)
        pb_obj_list.append(pb_obj)
    
    return pb_obj_list



def spawn_object_with_scaling(bc: BulletClient, fp: Path, large: bool = False) -> int:
    # Apply default mesh scaling
    base_len = 0.65
    scale_ = 1/base_len
    categorical_scale = {
        'can':1/base_len*0.15, 'bottle':1/base_len*0.15, 'bowl':1/base_len*0.2,
        'camera':1/base_len*0.15, 'laptop':1/base_len*0.3, 'mug':1/base_len*0.12
    }
    for key, value in categorical_scale.items():
        if key in fp.name:
            scale_ = value
    # Apply random scaling
    if large:
        scale_ = scale_*np.random.uniform(1.1, 1.3, size=(1,))
    else:
        scale_ = scale_*np.random.uniform(0.7, 1.3, size=(1,))
    scale_ = scale_*np.ones(3)
    # Create pb object
    color = pbutil.looking_good_color()
    uid = pbutil.create_pb_multibody_from_file(bc, str(fp), [0,0,0], [0,0,0,1], scale=scale_, color=color)

    return uid, scale_, color



def spawn_random_collision_free_objects_handpick(
        pb_state: PybulletStateEstimation,
        obj_file_path_list: List[Path],
        object_region: Tuple[np.ndarray, np.ndarray] ,
        trial: int = 5000
) -> ObjectsConfiguration:
    """Spawn hand-picked configurations"""
    
    occluders_path_list = [p for p in obj_file_path_list if "bottle" in p.name or "mug" in p.name or "bowl" in p.name or "can" in p.name]
    occludees_path_list = [p for p in obj_file_path_list if "bottle" in p.name or "can" in p.name]
    
    # Select random objects
    occluder_path = np.random.choice(occluders_path_list, 1).item()
    occludee_path = np.random.choice(occludees_path_list, 1).item()
    
    # TODO
    # Spawn occluder first.
    # Spawn occludee second.
    # Check occluder is in front of occludee.
    object_region_min = object_region[0]
    object_region_max = object_region[1]
    y_mean = (object_region_min[1]+object_region_max[1])/2.
    # Occluder region
    occluder_region_min = np.copy(object_region_min)
    occluder_region_max = np.copy(object_region_max)
    occluder_region_min[1] = y_mean
    draw_box(pb_state.bc, occluder_region_min, occluder_region_max)

    # Occludee region
    occludee_region_min = np.copy(object_region_min)
    occludee_region_max = np.copy(object_region_max)
    occludee_region_max[1] = y_mean
    draw_box(pb_state.bc, occludee_region_min, occludee_region_max)


    # Spawn occludee
    occludee_uid, occludee_scale, occludee_color = spawn_object_with_scaling(pb_state.bc, occludee_path)
    pb_obj_occludee = BulletObject(pb_state.bc, occludee_uid)
    for i in range(trial):
        pos_rand = np.random.uniform(occludee_region_min, occludee_region_max, size=(3,))
        # TODO: stochasticity for stand objects
        obj_quat = np.where(np.random.uniform() > 0.4, np.array([0.707, 0, 0, 0.707]), np.array([0, 0, 0, 1]))
        z_ang = sciR.from_euler('z', np.random.uniform(-np.pi, np.pi, size=()))*sciR.from_quat(obj_quat)
        ori_rand = z_ang.as_quat()
        pb_state.bc.resetBasePositionAndOrientation(occludee_uid, pos_rand, ori_rand)
        pb_state.bc.performCollisionDetection()
        cp_res = pb_state.bc.getContactPoints(occludee_uid)
        if len(cp_res) == 0:
            break


    # Spawn occluder
    occluder_uid, occluder_scale, occluder_color = spawn_object_with_scaling(pb_state.bc, occluder_path, large=True)
    pb_obj_occluder = BulletObject(pb_state.bc, occluder_uid)
    for i in range(trial):
        pos_rand = np.random.uniform(occluder_region_min, occluder_region_max, size=(3,))
        # TODO: stochasticity for stand objects
        obj_quat = np.where(np.random.uniform() > 0.4, np.array([0.707, 0, 0, 0.707]), np.array([0, 0, 0, 1]))
        z_ang = sciR.from_euler('z', np.random.uniform(-np.pi, np.pi, size=()))*sciR.from_quat(obj_quat)
        ori_rand = z_ang.as_quat()
        pb_state.bc.resetBasePositionAndOrientation(occluder_uid, pos_rand, ori_rand)
        pb_state.bc.performCollisionDetection()
        cp_res = pb_state.bc.getContactPoints(occluder_uid)
        if len(cp_res) == 0:
            break


    # Collect
    pb_obj_list = [pb_obj_occluder, pb_obj_occludee]
    obj_path_list = [occluder_path, occludee_path]
    obj_scale_list = [occluder_scale, occludee_scale]
    obj_color_list = [occluder_color, occludee_color]
    obj_configs = ObjectsConfiguration(
        pb_obj_list,
        obj_path_list, 
        obj_scale_list, 
        obj_color_list
    )

    # Stabilize
    step(pb_state.bc, 1000)
    change_dynamics_slippery(pb_state.bc, pb_obj_list)
    step(pb_state.bc, 1000)
    change_dynamics_stiff(pb_state.bc, pb_obj_list)

    # Check difference
    difference = check_mask_difference(
        pb_state, 
        pb_obj_occluder,
        occluder_path,
        occluder_scale,
        occluder_color,
        pb_obj_occludee,
        pixels_threshold = 50
    )
    if difference:
        valid = True
    else:
        valid = False


    return obj_configs, valid


def check_mask_difference(
        pb_state: PybulletStateEstimation,
        pb_obj_occluder: BulletObject,
        pb_obj_occluder_path: Path,
        pb_obj_occluder_scale: np.ndarray,
        pb_obj_occluder_color: np.ndarray,
        pb_obj_occludee: BulletObject,
        pixels_threshold: int
):
    
    # Capture segmentation after occlusion
    occludee_segmask_list_after = []
    for cam in pb_state.cameras:
        _, _, seg = cam.capture_rgbd_image()
        masks = (seg == pb_obj_occludee.uid)
        occludee_segmask_list_after.append(masks)
    occludee_segmask_list_after = np.array(occludee_segmask_list_after)

    # Remove body
    occluder_pos, occluder_quat = pb_state.bc.getBasePositionAndOrientation(pb_obj_occluder.uid)
    pb_state.bc.removeBody(pb_obj_occluder.uid)
    
    # Capture segmentation before occlusion
    occludee_segmask_list_before = []
    for cam in pb_state.cameras:
        _, _, seg = cam.capture_rgbd_image()
        masks = (seg == pb_obj_occludee.uid)
        occludee_segmask_list_before.append(masks)
    occludee_segmask_list_before = np.array(occludee_segmask_list_before)

    # Check difference
    difference = (occludee_segmask_list_before!=occludee_segmask_list_after)
    difference = np.sum(difference, axis=(-2, -1))
    if np.all(difference > pixels_threshold):
        flag = True
    else:
        flag = False

    # Restore
    pb_obj_occluder.uid = pbutil.create_pb_multibody_from_file(
        pb_state.bc, 
        str(pb_obj_occluder_path), 
        [0,0,0], 
        [0,0,0,1], 
        scale = pb_obj_occluder_scale, 
        color = pb_obj_occluder_color
    )
    pb_state.bc.resetBasePositionAndOrientation(pb_obj_occluder.uid, occluder_pos, occluder_quat)
    
    return flag

