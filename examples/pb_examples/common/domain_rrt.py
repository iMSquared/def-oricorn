from __future__ import annotations
from pathlib import Path
import jax
import numpy as np
import torch
import random
import pybullet as pb
from pybullet_utils.bullet_client import BulletClient
from scipy.spatial.transform import Rotation as sciR
from copy import deepcopy
from dataclasses import dataclass
import logging
import itertools
import pickle
import time

from typing import Dict, Tuple, List, Optional, NamedTuple

import yaml
import util.io_util as ioutil
import util.pb_util as pbutil

# Environment
from imm_pb_util.imm.pybullet_util.common import get_link_pose, imagine
from imm_pb_util.imm.pybullet_util.typing_extra import TranslationT
from imm_pb_util.imm.pybullet_util.vision import CoordinateVisualizer
from imm_pb_util.bullet_envs.env import SimpleEnv
from imm_pb_util.bullet_envs.robot import FrankaPanda, PandaGripper
from imm_pb_util.bullet_envs.manipulation import FrankaManipulation
from imm_pb_util.bullet_envs.objects import BulletObject
from imm_pb_util.bullet_envs.camera import SimpleCamera, RobotAttachedCamera
from imm_pb_util.bullet_envs.nvisii import BaseNVRenderer

# ?
from examples.pb_examples.common.common import configure_bullet, change_dynamics_slippery, change_dynamics_stiff





class ShelfEnv(SimpleEnv):
    
    class Meta(NamedTuple):
        name: str
        scale: float
        x_length: float
        shelf_height: float
        
        @property
        def gripper_min_dist(self):
            return self.x_length-0.05

        @property
        def gripper_region(self):
            return np.array([-self.x_length+0.05, -0.15, 0.60]), np.array([self.x_length-0.05, 0.00, 0.65])
        
        @property
        def object_region(self):
            return np.array([-self.x_length, -0.13, 0.61]), np.array([self.x_length, 0.12, 0.65])

    def __init__(self, bc: BulletClient, config: Dict):
        super().__init__(bc, config)
        self.shelf_list = [
            self.Meta('shelf-040', 1.8, 0.55, 0.5),
            self.Meta('shelf-040', 1.8, 0.55, 0.02),
            self.Meta('shelf-045', 1.8, 0.4, 0.14),
            self.Meta('shelf-045', 1.8, 0.4, -0.15),
            self.Meta('wall-shelf-028', 1.8, 0.55, -0.48),
            self.Meta('wall-shelf-028', 1.8, 0.60, -0.02),
        ]
        
    def create_shelf(self, i):
        """?"""
        shelf_name, shelf_scale, xlen, shelf_height = self.shelf_list[i]
        shelf_id = pb.createMultiBody(
            baseMass = 0.0, 
            basePosition = [0,0,shelf_height], 
            baseOrientation = sciR.from_euler('x',np.pi/2).as_quat(),
            baseCollisionShapeIndex = pb.createCollisionShape(
                pb.GEOM_MESH, fileName=f'data/shelf/32_64_1_v4/{shelf_name}.obj', meshScale=[shelf_scale,shelf_scale,shelf_scale]),
            baseVisualShapeIndex = pb.createVisualShape(
                pb.GEOM_MESH, fileName=f'data/shelf/{shelf_name}-obj/{shelf_name}.obj', meshScale=[shelf_scale,shelf_scale,shelf_scale]))
        self.env_assets['shelf'] = shelf_id

        # Change color 
        SHELF_RGBA_COLOR = [0.7, 0.7, 0.7, 1]
        self.bc.changeVisualShape(self.env_assets['shelf'], -1, rgbaColor=SHELF_RGBA_COLOR)

    def reset_shelf(self, i):
        """?"""
        if "shelf" in self.env_assets.keys():
            uid = self.env_assets.pop("shelf")
            self.bc.removeBody(uid)
        self.create_shelf(i)



def get_shelf_params(idx: int) -> List[np.ndarray]:
    """Parameters for plane collision"""
    shelf_param_dict = {
        0: None,
        1: np.array([
            [0,0,1,0.5],
            [0,0,1,0.5],
            [0,-1,0,-0.1],
            [0,-1,0,-0.1]]),
        2: None,
        3: np.array([
            [0,0,1,0.52],
            [1,0,0,-0.48],
            [0,-1,0,-0.13],
            [-1,0,0,-0.48]]),
        4: np.array([
            [0,0,1,0.53],
            [0,0,1,0.53],
            [0,0,1,0.53],
            [0,0,1,0.53]]),
        5:np.array([
            [0,0,1,0.49],
            [0,0,1,0.49],
            [0,0,-1,-0.965],
            [0,-1,0,-0.11]])
    }
    return shelf_param_dict[idx]



@dataclass
class PybulletStateRRT:
    """asdf"""
    bc: BulletClient
    robot: FrankaPanda
    manip: FrankaManipulation
    env: ShelfEnv
    obj_list: List[BulletObject]
    cameras: List[SimpleCamera]
    nv_renderer: BaseNVRenderer
    robot_view_pose: np.ndarray
    config: Dict

    @classmethod
    def init_from_config(cls, config):
        # Simulation initialization
        bc = configure_bullet(config)
        robot, manip, env, cameras, nv_renderer, robot_view_pose = initialize_rrt_domain(bc, config)
        obj_list = []
        return cls(bc, robot, manip, env, obj_list, cameras, nv_renderer, robot_view_pose, config)
    
    
    def reset_hard(self):
        """Hard reset"""
        # Simulation initialization
        self.bc.disconnect()
        bc = configure_bullet(self.config)
        robot, manip, env, cameras, nv_renderer, robot_view_pose = initialize_rrt_domain(bc, self.config)
        self.bc = bc
        self.manip = manip
        self.robot = robot
        self.env = env
        self.obj_list = []
        self.cameras = cameras
        self.nv_renderer = nv_renderer
        self.robot_view_pose = robot_view_pose


    def reset_soft(self):
        """Clear environment + objects"""
        raise NotImplementedError()


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


def initialize_rrt_domain(bc, config):
    """Initialize skeleton environment for RRT problem. No objects are initialized"""
    env = ShelfEnv(bc, config)
    robot = FrankaPanda(bc, config)
    manip = FrankaManipulation(bc, robot, config)
    # Stabilize
    for i in range(100):
        bc.stepSimulation()
    # Configure cameras
    cameras = [SimpleCamera(i, bc, config) for i in range(0, 3)]
    nv_renderer = BaseNVRenderer(bc, show_gui=False)
    # Visualize camera frames
    for cam in cameras:
        cam.show_camera_pose()
    # Get safe robot rest_pose which do not interfere with perception.
    robot_attached_camera = cameras[0]
    cam_pos = robot_attached_camera.extrinsic.camera_pos
    cam_orn = robot_attached_camera.extrinsic.camera_quat
    attach_gl_pos = [0.05, 0.0, 0.0]
    attach_gl_orn = bc.getQuaternionFromEuler([3.1415927, 0, 1.5707963])
    _, robot_view_pose = manip.solve_ik_numerical(
        *RobotAttachedCamera.compute_ee_pos_from_camera_pose(
            bc, cam_pos, cam_orn, attach_gl_pos, attach_gl_orn))
    
    # Reset robot gripper
    robot.reset_finger(0)

    return robot, manip, env, cameras, nv_renderer, robot_view_pose


def save_rrt_problem_to_config(
        output_path: Path,
        i: int,
        pb_state: PybulletStateRRT,
        joint_pos_start: np.ndarray,
        joint_pos_goal: np.ndarray,
        shelf_idx: int,
        pb_obj_path_list: list[Path],
        pb_obj_scale_list: list[np.ndarray],
        pb_obj_color_list: list[np.ndarray],
):
    robot = {
        "joint_pos_start": joint_pos_start.tolist(),
        "joint_pos_goal": joint_pos_goal.tolist() }

    items = []
    for pb_obj, path, scale, color in zip(pb_state.obj_list, pb_obj_path_list, pb_obj_scale_list, pb_obj_color_list):
        pos, quat = pb_state.bc.getBasePositionAndOrientation(pb_obj.uid)
        item = {
            "obj": Path(path).name,
            "scale": scale.tolist(),
            "pos": list(pos),
            "quat": list(quat),
            "color": color.tolist(),
        }
        items.append(item)

    data = {
        "shelf_idx": shelf_idx,
        "robot": robot,
        "objects": items
    }

    data_yaml_str = yaml.dump(data)
    filename = f"sim_rrt_{i}.yaml"
    with (output_path/filename).open("w") as f:
        f.write(data_yaml_str)


def load_rrt_problem_from_config(
        pb_state: PybulletStateRRT,
        config_dir_path: Path,
        obj_dir_path: Path,
        i: int,
):
    # Open
    print((config_dir_path/f"sim_rrt_{i}.yaml"))
    with (config_dir_path/f"sim_rrt_{i}.yaml").open("r") as f:
        data = yaml.safe_load(f)
        shelf_idx = data["shelf_idx"]
        robot_cfgs = data["robot"]
        objs = data["objects"]
    
    pb_state.reset_objects()
    pb_state.env.reset_shelf(shelf_idx)
    joint_pos_start = robot_cfgs["joint_pos_start"]
    joint_pos_goal = robot_cfgs["joint_pos_goal"]
    pb_state.robot.reset(joint_pos_start)

    pb_obj_list =[]
    for obj in objs:
        name = obj["obj"]
        scale = obj["scale"]
        pos = obj["pos"]
        quat = obj["quat"]
        color = obj["color"]
        obj_path = obj_dir_path/name
        uid = pbutil.create_pb_multibody_from_file(
            pb_state.bc, str(obj_path), pos, quat, 
            scale=scale, color=np.array(color), fixed_base=True)
        pb_obj = BulletObject(pb_state.bc, uid)
        pb_obj_list.append(pb_obj)

    # Return
    pb_state.obj_list = pb_obj_list
    return joint_pos_start, joint_pos_goal, shelf_idx





@dataclass(frozen=True)
class NodeState:
    """For evaluation"""
    collision: bool
    penetration_depth: float
    collision_uid: int|None

@dataclass(frozen=True)
class TrajectoryState:
    """For evaluation"""
    # Core
    traj: np.ndarray
    success: bool
    node_states: tuple[NodeState]

    # Aux
    collision_node_indices: tuple[int]
    penetration_depth_list: tuple[float]
    collision_uid_list: tuple[int]

    @classmethod
    def get_collision_along_trajectory(
        cls,
        pb_state: PybulletStateRRT,
        traj: np.ndarray,
        sleep: bool=False,
    ) -> "TrajectoryState":
        """Get collision result from trajectory"""

        # Flags
        success = True
        node_states: list[NodeState] = []

        # Iterate through nodes
        for i, q in enumerate(traj):
            # Check collision at q.
            pb_state.robot.reset(q)
            pb.performCollisionDetection()
            col_result = pb.getContactPoints(pb_state.robot.uid)
            
            # Check penetration depth
            max_penetration_depth = 0.
            collision_uid = None
            for col in col_result:
                penetration_depth = col[8]
                uid = col[2]
                if penetration_depth < max_penetration_depth:
                    max_penetration_depth = penetration_depth
                    collision_uid = uid

            # Compose node state
            if max_penetration_depth < 0:
                node_state = NodeState(
                    collision = True, 
                    penetration_depth = max_penetration_depth, 
                    collision_uid = collision_uid)
                success = False
            else:
                node_state = NodeState(
                    collision = False, 
                    penetration_depth = 0.,
                    collision_uid = None
                )
            node_states.append(node_state)
            if sleep:
                time.sleep(sleep)

        # Return collision results
        node_states = tuple(node_states)
        collision_node_indices = tuple([i for i, node in enumerate(node_states) if node.collision==True])
        penetration_depth_list = tuple([node.penetration_depth for node in node_states if node.collision==True])
        collision_uid_list = tuple([node.collision_uid for node in node_states if node.collision==True])

        return cls(traj, success, node_states, collision_node_indices, penetration_depth_list, collision_uid_list)

    def dump_trajectory(self, logs_dir_path: Path, idx: int):
        """Dump trajectory to numpy for replay"""
        out_file_name = logs_dir_path/f"traj_{idx}.npz"
        with out_file_name.open("wb") as f:
            np.savez_compressed(f, traj=np.asarray(self.traj))

@dataclass
class RRTResultState:
    """For evaluation"""

    # Core
    trajectory_state_records: dict[int, TrajectoryState] # key=idx

    def __init__(self):
        self.trajectory_state_records = {}

    def update(
            self,
            pb_state: PybulletStateRRT, 
            idx: int,
            trajectory_state: TrajectoryState,
            print_log: bool = True,
            verbose: int = 0,
    ):
        """Update and summarize episode evaluation result."""
        
        self.trajectory_state_records[idx] = trajectory_state

        if print_log:
            # Print number of nodes in collision
            traj_length = len(trajectory_state.node_states)
            num_col_nodes = len(trajectory_state.collision_node_indices)
            num_safe_nodes = traj_length - num_col_nodes
            collsion_log_str = (
                f"Collision results for the trajectory: {trajectory_state.success}\n"
                f"\tSpecific nodes: {trajectory_state.collision_node_indices}\n"
                f"\tNum col nodes : {num_col_nodes}\n"
                f"\tNum safe nodes: {num_safe_nodes}\n"
            )
            print(collsion_log_str)
            logging.info(collsion_log_str)
            
            # In more detail...
            if verbose >= 1:
                # Environment info
                shelf_id = pb_state.env.env_assets['shelf']
                obj_uids = [obj.uid for obj in pb_state.obj_list]

                print(f'Shelf ID: {shelf_id}')
                print(f'obj_uids: {obj_uids}')
                logging.info(f'Shelf ID: {shelf_id}')
                logging.info(f'obj_uids: {obj_uids}')
                # Collision detail
                loop = zip(
                    trajectory_state.collision_node_indices,
                    trajectory_state.penetration_depth_list,
                    trajectory_state.collision_uid_list
                )
                for node_idx, depth, uid in loop:
                    str_to_log = f'Collision in index {node_idx:4d}/{traj_length:4d}, depth: {depth*1000:4.2f}mm, uid: {uid}'
                    print(str_to_log)
                    logging.info(str_to_log)

    def summarize(self):
        """Summarize final evaluation result."""

        trajectory_states_list = list(self.trajectory_state_records.values())
        # Success rate
        num_iterations = len(trajectory_states_list)
        num_success = len([traj for traj in trajectory_states_list if traj.success])
        # Col nodes
        num_col_nodes_list = [len(traj.collision_node_indices) for traj in trajectory_states_list]
        avg_num_col_nodes = np.mean(num_col_nodes_list)
        std_num_col_nodes = np.std(num_col_nodes_list)
        # Penetration depth
        min_depth_list = np.array([min(ts.penetration_depth_list) if len(ts.penetration_depth_list)>0 else 0 for ts in trajectory_states_list])
        min_depth_list = min_depth_list * 1000.
        avg_min_depth = np.mean(min_depth_list)
        std_min_depth = np.std(min_depth_list)
        # Penetration depth (among col traj)
        min_depth_list_col = np.array([min(ts.penetration_depth_list) for ts in trajectory_states_list if len(ts.penetration_depth_list)>0])
        min_depth_list_col = min_depth_list_col * 1000.
        avg_min_depth_col = np.mean(min_depth_list_col)
        std_min_depth_col = np.std(min_depth_list_col)


        fail_indices = [idx for idx, traj_state in self.trajectory_state_records.items() if traj_state.success==False]

        # Logging
        log_str = (
            f"RRT experiment result\n"
            f"Success: {num_success} / {num_iterations} ({num_success*100/num_iterations:.1f}%)\n"
            f"Avg nodes in collision (among all traj): {avg_num_col_nodes:4.2f} ({std_num_col_nodes:4.2f})\n"
            f"Avg max penetation depth (among all traj): {avg_min_depth:4.2f} ({std_min_depth:4.2f}) mm\n"
            f"Avg max penetation depth (among col traj): {avg_min_depth_col:4.2f} ({std_min_depth_col:4.2f}) mm\n"
            f"Failed testset indices: {fail_indices}"
        )
        print(log_str)
        logging.info(log_str)