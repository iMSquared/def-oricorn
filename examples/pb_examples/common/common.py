from __future__ import annotations
import numpy as np
import pybullet as pb
from pybullet_utils.bullet_client import BulletClient
from scipy.spatial.transform import Rotation as sciR
from pathlib import Path
import jax
import jax.numpy as jnp
import flax
from typing import Dict, List
import pickle

from abc import ABC

import util.cvx_util as cxutil

# Environment
from imm_pb_util.imm.pybullet_util.common import get_link_pose
from imm_pb_util.imm.pybullet_util.vision import CoordinateVisualizer
from imm_pb_util.bullet_envs.env import SimpleEnv
from imm_pb_util.bullet_envs.robot import FrankaPanda, PandaGripper
from imm_pb_util.bullet_envs.manipulation import FrankaManipulation
from imm_pb_util.bullet_envs.objects import BulletObject
from imm_pb_util.bullet_envs.camera import SimpleCamera, RobotAttachedCamera



def configure_bullet(config: Dict):
    # Configuration
    DEBUG_SHOW_GUI         = config["project_params"]["debug_show_gui"]
    DEBUG_GUI_DELAY        = config["project_params"]["debug_delay_gui"]
    CONTROL_HZ             = config["sim_params"]["control_hz"]
    GRAVITY                = config["sim_params"]["gravity"]
    CAMERA_DISTANCE        = config["sim_params"]["debug_camera"]["distance"]
    CAMERA_YAW             = config["sim_params"]["debug_camera"]["yaw"]
    CAMERA_PITCH           = config["sim_params"]["debug_camera"]["pitch"]
    CAMERA_TARGET_POSITION = config["sim_params"]["debug_camera"]["target_position"]
    if DEBUG_SHOW_GUI:
        bc = BulletClient(connection_mode=pb.GUI)
    else:
        bc = BulletClient(connection_mode=pb.DIRECT)

    # Sim params
    CONTROL_DT = 1. / CONTROL_HZ
    bc.setTimeStep(CONTROL_DT)
    bc.setGravity(0, 0, GRAVITY)
    bc.resetDebugVisualizerCamera(
        cameraDistance       = CAMERA_DISTANCE, 
        cameraYaw            = CAMERA_YAW, 
        cameraPitch          = CAMERA_PITCH, 
        cameraTargetPosition = CAMERA_TARGET_POSITION )
    bc.configureDebugVisualizer(bc.COV_ENABLE_RENDERING)
    bc.configureDebugVisualizer(bc.COV_ENABLE_GUI,0)
    bc.setPhysicsEngineParameter(enableFileCaching=0)

    return bc





def step(bc, steps=-1):
    while steps != 0: 
        bc.stepSimulation()
        steps -= 1


def draw_box(bc, corner1_cuboid, corner2_cuboid, ids=[]):
    # Calculate the coordinates of the other six corners of the cuboid
    corner3_cuboid = np.array([corner1_cuboid[0], corner2_cuboid[1], corner1_cuboid[2]])
    corner4_cuboid = np.array([corner2_cuboid[0], corner1_cuboid[1], corner1_cuboid[2]])
    corner5_cuboid = np.array([corner1_cuboid[0], corner1_cuboid[1], corner2_cuboid[2]])
    corner6_cuboid = np.array([corner2_cuboid[0], corner2_cuboid[1], corner1_cuboid[2]])
    corner7_cuboid = np.array([corner1_cuboid[0], corner2_cuboid[1], corner2_cuboid[2]])
    corner8_cuboid = np.array([corner2_cuboid[0], corner1_cuboid[1], corner2_cuboid[2]])

    # Edges of the cuboid, defined by pairs of corners
    edges_cuboid = [
        (corner1_cuboid, corner3_cuboid),
        (corner1_cuboid, corner4_cuboid),
        (corner1_cuboid, corner5_cuboid),
        (corner2_cuboid, corner6_cuboid),
        (corner2_cuboid, corner7_cuboid),
        (corner2_cuboid, corner8_cuboid),
        (corner3_cuboid, corner6_cuboid),
        (corner3_cuboid, corner7_cuboid),
        (corner4_cuboid, corner6_cuboid),
        (corner4_cuboid, corner8_cuboid),
        (corner5_cuboid, corner7_cuboid),
        (corner5_cuboid, corner8_cuboid)
    ]

    new = (len(ids) == 0)
    new_ids = []
    for i, e in enumerate(edges_cuboid):
        if new:
            id = bc.addUserDebugLine(*e, lineColorRGB=[0, 1, 0], lineWidth=2)
        else:
            id = bc.addUserDebugLine(*e, lineColorRGB=[0, 1, 0], lineWidth=2, replaceItemUniqueId=ids[i])
        new_ids.append(id)
    return new_ids


def change_dynamics_stiff(bc: BulletClient, pb_obj_list: List[BulletObject]):
    for pb_obj in pb_obj_list:
        bc.changeDynamics(pb_obj.uid, -1, rollingFriction=0.1, spinningFriction=0.1, lateralFriction=1.0)


def change_dynamics_slippery(bc: BulletClient, pb_obj_list: List[BulletObject]):
    for pb_obj in pb_obj_list:
        bc.changeDynamics(pb_obj.uid, -1, rollingFriction=0.05, spinningFriction=0.05)


def dump_latent_objects_to_file(file_path: Path, latent_obj: cxutil.LatentObjects, pred_conf: jnp.ndarray):
    """asdf"""
    with file_path.open("wb") as f:
        pickle_dict = {
            "latent_obj": latent_obj,
            "pred_conf": pred_conf
        }
        pickle.dump(pickle_dict, f)


def load_latent_objects_from_file(file_path: Path) -> cxutil.LatentObjects:
    """asdf"""
    with file_path.open("rb") as f:
        pickle_dict = pickle.load(f)
        latent_obj = pickle_dict["latent_obj"]
        pred_conf = pickle_dict["pred_conf"]

    return latent_obj, pred_conf