from __future__ import annotations
import os
# os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"   
# os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"    # NVISII will crash when showed multiple devices with jax.

import torch
import random
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict
from PIL import Image
import pybullet as pb
import yaml
import jax.numpy as jnp
import jax
import nvisii
from functools import partial
from scipy.spatial.transform import Rotation as sciR
import einops
import logging
import pickle
import time
import argparse
from dataclasses import dataclass


from jax.experimental.compilation_cache import compilation_cache as cc
cc.set_cache_dir("./cache")

# PyBullet
import pybullet as pb
from pybullet_utils.bullet_client import BulletClient

# Setup import path
import sys
BASEDIR = Path(__file__).parent.parent.parent
if str(BASEDIR) not in sys.path:
    sys.path.insert(0, str(BASEDIR))

import util.io_util as ioutil
import util.camera_util as cutil
import util.cvx_util as cxutil
import util.transform_util as tutil
import util.pb_util as pbutil
import util.environment_util as envutil
import util.franka_util as fkutil
import util.render_util as rutil
import util.inference_util as ifutil

# Environment
from imm_pb_util.imm.pybullet_util.common import get_link_pose, imagine
from imm_pb_util.imm.pybullet_util.vision import CoordinateVisualizer
from imm_pb_util.bullet_envs.env import SimpleEnv
from imm_pb_util.bullet_envs.robot import FrankaPanda, PandaGripper
from imm_pb_util.bullet_envs.manipulation import FrankaManipulation
from imm_pb_util.bullet_envs.objects import BulletObject
from imm_pb_util.bullet_envs.camera import SimpleCamera, RobotAttachedCamera
from imm_pb_util.bullet_envs.nvisii import BaseNVRenderer

# 
from examples.pb_examples.common.common import configure_bullet, draw_box, dump_latent_objects_to_file
from examples.pb_examples.common.domain_rrt import (
    PybulletStateRRT, load_rrt_problem_from_config, get_shelf_params, NodeState, TrajectoryState, RRTResultState
)
from examples.pb_examples.common.planning_modules_ours import IntegratedModulesOurs



def load_pre_rendered_rgb_images(
    pb_state: PybulletStateRRT,
    testset_config_dir_path: Path,
    testset_index: int
):
    """Get pre-rendered RGBs for a testset"""
    image_save_dir_path = testset_config_dir_path/"images"
    nv_rgb_list = []
    for i, cam in enumerate(pb_state.cameras):
        file_path = image_save_dir_path/f"sim_rrt_{testset_index}_rgb_view{i}.png"
        image = np.array(Image.open(file_path))
        nv_rgb_list.append(image)

    return np.array(nv_rgb_list)


def evaluate(
        jkey: jax.Array,
        pb_state: PybulletStateRRT,
        integrated_modules: IntegratedModulesOurs,
        testset_config_dir_path: Path,
        testset_index: int,
        obj_dir_path: Path,
):
    logging.info(f'start exp id: {testset_index}')

    # Load reaching environment
    joint_pos_start, joint_pos_goal, shelf_idx = load_rrt_problem_from_config(
        config_dir_path = testset_config_dir_path,
        obj_dir_path = obj_dir_path,
        i = testset_index, 
        pb_state = pb_state
    )
    nv_rgb_list = load_pre_rendered_rgb_images(pb_state, testset_config_dir_path, testset_index)

    # Motion plan
    initial_q = pb_state.robot.rest_pose
    shelf_params = get_shelf_params(shelf_idx)
    obj_pred_sorted, conf_sorted, traj = integrated_modules.plan(
        jkey, 
        nv_rgb_list,
        pb_state.cameras,
        initial_q,
        joint_pos_start,
        joint_pos_goal,
        shelf_params,
    )

    # Check ground truth collision
    pb_state.robot.reset(joint_pos_start)
    pb_state.manip.draw_trajectory(traj)
    traj_state = TrajectoryState.get_collision_along_trajectory(pb_state, traj)

    # Reset 
    pb_state.reset_objects()

    return obj_pred_sorted, conf_sorted, traj_state



def main(args: argparse.Namespace, config: Dict):
    """Demo main function

    Args:
        config (Dict): Configuration dict
    """
    # Random control
    SEED = args.seed
    np.random.seed(SEED)
    random.seed(SEED)
    torch.manual_seed(SEED)
    jkey = jax.random.key(SEED)

    # Config
    DATASET_START_IDX = 0
    DATASET_END_IDX = 200
    DATASET_RANGE = range(DATASET_START_IDX, DATASET_END_IDX)
    TESTSET_CONFIG_DIR_PATH = Path("examples/pb_examples/testset/rrt")
    OBJ_DIR_PATH = Path('data/NOCS_val/32_64_1_v4')
    PB_RESET_PERIOD = 5


    # Logging configuration
    date_time = ioutil.get_current_timestamp()
    LOGS_DIR_PATH = Path('logs_eval_rrt')/f"{date_time}_{Path(args.ckpt_dir).name}"
    LOGS_DIR_PATH.mkdir(exist_ok=True, parents=True)
    logging.basicConfig(
        filename = LOGS_DIR_PATH/'exp_logs.log', 
        encoding = 'utf-8', 
        level = logging.INFO
    )

    # Init state
    pb_state = PybulletStateRRT.init_from_config(config)

    # Get estimator
    robot_base_pos = pb_state.robot.base_pos
    robot_base_quat = pb_state.robot.base_orn
    CKPT_DIR: str = args.ckpt_dir 
    USE_UNCERTAINTY: bool = bool(args.use_uncertainty)
    integrated_modules = IntegratedModulesOurs(
        CKPT_DIR, robot_base_pos, robot_base_quat, USE_UNCERTAINTY, str(LOGS_DIR_PATH))
    
    # Evaluate
    rrt_result_state = RRTResultState()
    for idx in DATASET_RANGE:
        # Periodic pybullet reset
        pb_state.reset_hard_periodically(idx, PB_RESET_PERIOD)

        jkey, subkey = jax.random.split(jkey)
        obj_pred_sorted, conf_sorted, traj_state = evaluate(
            jkey = subkey,
            pb_state = pb_state, 
            integrated_modules = integrated_modules, 
            testset_config_dir_path = TESTSET_CONFIG_DIR_PATH,
            testset_index = idx,
            obj_dir_path = OBJ_DIR_PATH,
        )

        # Summarize after every episode.
        rrt_result_state.update(pb_state, idx, traj_state, verbose=1)
        rrt_result_state.summarize()
        # Dump
        traj_state.dump_trajectory(logs_dir_path=LOGS_DIR_PATH, idx=idx)
        dump_latent_objects_to_file(file_path=(LOGS_DIR_PATH/f"pred_{idx}.pkl"), latent_obj=obj_pred_sorted, pred_conf=conf_sorted)




if __name__=="__main__":

    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ckpt_dir", type=str, default="checkpoints/dif_model/02262024-180216_final")
    parser.add_argument("--use_uncertainty", type=int, default=1)
    args = parser.parse_args()

    # Open yaml config file
    config_file_path = Path(__file__).parent.resolve() / "pb_cfg" / "pb_sim_RRT.yaml"
    with open(config_file_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # Main
    main(args, config)


