from __future__ import annotations
import math
import time
import numpy as np
import numpy.typing as npt
import importlib
from contextlib import contextmanager
from typing import List, Tuple, Callable, Optional
import random

# Bullet
from pybullet_utils.bullet_client import BulletClient
from imm.pybullet_util.typing_extra import TranslationT, QuaternionT
from imm.pybullet_util.common import ( imagine, 
                                       get_joint_positions,
                                       get_link_pose,
                                       get_relative_transform )
from imm.pybullet_util.collision import ( ContactBasedCollision,
                                          SetRobotState,
                                          GraspAffordance,
                                          LinkPair )
from imm.pybullet_util.vision import CoordinateVisualizer

# Env
from bullet_envs.robot import BulletRobot, FrankaPanda

# Custom packages...
rrt_connect = importlib.import_module("imm.motion-planners.motion_planners.rrt_connect")
rrt_star = importlib.import_module("imm.motion-planners.motion_planners.rrt_star")
from imm.franka_analytical_ik_wrapped import franka_ik_pybind



def get_linear_interpolated_trajectory(cur: List[float], 
                                       goal: List[float], 
                                       action_duration: float, 
                                       control_dt: float) -> Tuple[npt.NDArray, ...]:
    """
    This function returns linear-interpolated (dividing straight line)
    trajectory between current and goal pose.
    Acc, Jerk is not considered.
    """
    # Interpolation steps
    steps = math.ceil(action_duration/control_dt)
    # Calculate difference
    delta = [ goal[i] - cur[i] for i in range(len(cur)) ]
    # Linear interpolation
    trajectory: Tuple[npt.NDArray, ...] = ([
        np.array([
            cur[j] + ( delta[j] * float(i)/float(steps) ) 
            for j in range(len(cur))
        ])
        for i in range(1, steps+1)
    ])

    return trajectory


@contextmanager
def reset_and_restore_robot(bc: BulletClient, 
                          robot: BulletRobot, 
                          joint_indices: npt.NDArray,
                          joint_pos_reset: npt.NDArray,
                          joint_pos_restore: npt.NDArray, ):
    for i, v in zip(joint_indices, joint_pos_reset):
        bc.resetJointState(robot.uid, i, v)
    yield
    for i, v in zip(joint_indices, joint_pos_restore):
        bc.resetJointState(robot.uid, i, v)
    

class Manipulation:

    def __init__(self, bc    : BulletClient,
                       robot : BulletRobot,
                       config: dict): 
        
        # TODO: pure class
        

        self.config     = config
        self.bc         = bc
        self.robot      = robot
        
        # Sim params
        sim_params = config["sim_params"]
        self.DEBUG_SHOW_GUI  = config["project_params"]["debug_show_gui"]
        self.DEBUG_DELAY_GUI = config["project_params"]["debug_delay_gui"]
        self.CONTROL_HZ      = sim_params["control_hz"]
        self.CONTROL_DT      = 1. / sim_params["control_hz"]
        # Manipulation params
        manipulation_params = config["manipulation_params"]
        # For numerical IK solver
        self.IK_MAX_NUM_ITERATIONS = manipulation_params["inverse_kinematics"]["max_num_iterations"]
        self.IK_RESIDUAL_THRESHOLD = manipulation_params["inverse_kinematics"]["residual_threshold"]
        # RRT params
        self.RRT_RESTARTS   = manipulation_params["rrt_restarts"]
        self.RRT_ITERATIONS = manipulation_params["rrt_iterations"]
        self.RRT_STAR_MAX_ITERATIONS = manipulation_params["rrt_star_max_iterations"]
        # Default RRT callbacks
        self.TOUCH_TOL = manipulation_params["touch_tol"]
        distance_fn, sample_fn, extend_fn = self.__get_default_callbacks()
        self.default_distance_fn  = distance_fn
        self.default_sample_fn    = sample_fn
        self.default_extend_fn    = extend_fn


    def __get_default_callbacks(self) -> Tuple[Callable, Callable, Callable]:
        """Init default sample, extend, distance functions

        Returns:
            distance_fn (Callable): _
            sample_fn (Callable): _
            extend_fn (Callable): _
        """
        def distance_fn(q0: np.ndarray, q1: np.ndarray):
            return np.linalg.norm(np.subtract(q1, q0))
        # def distance_fn(q0: np.ndarray, q1: np.ndarray):
        #     for i, v in zip(self.robot.nonfixed_joint_indices_arm, q0):
        #         self.bc.resetJointState(self.robot.uid, i, v)
        #     ee_pos0, ee_orn0 = get_link_pose(
        #         self.bc, self.robot.uid, 
        #         self.robot.link_index_endeffector_base,
        #         computeForwardKinematics=True)
        #     for i, v in zip(self.robot.nonfixed_joint_indices_arm, q1):
        #         self.bc.resetJointState(self.robot.uid, i, v)
        #     ee_pos1, ee_orn1 = get_link_pose(
        #         self.bc, self.robot.uid, 
        #         self.robot.link_index_endeffector_base,
        #         computeForwardKinematics=True)
        #     return np.linalg.norm(np.subtract(ee_pos1, ee_pos0))
        def sample_fn():
            # return np.random.uniform(
            #     self.robot.nonfixed_joint_limits[0]/2.0, 
            #     self.robot.nonfixed_joint_limits[1]/2.0)
            return np.random.uniform(
                self.robot.nonfixed_joint_limits[0], 
                self.robot.nonfixed_joint_limits[1])
        def extend_fn(q0: np.ndarray, q1: np.ndarray):
            dq = np.subtract(q1, q0)  # Nx6
            return q0 + np.linspace(0, 1, num=200)[:, None] * dq
        # def extend_fn(q0: np.ndarray, q1: np.ndarray, expand_length=0.4):
        #     dir = (q1 - q0) 
        #     dir = dir / np.linalg.norm(dir)
        #     extend_to = expand_length * dir / np.linalg.norm(dir) + q0
        #     return [q0, extend_to]
            
        return distance_fn, sample_fn, extend_fn


    def define_collision_fn(
            self, 
            touch_uids: List[int] = [],
            allow_uids: List[int] = []
    ) -> Callable:
        """Get new collision function without grasping

        Args:
            grasped_uid (int): Uid of grasped object
            touch_uids (List[int], optional): Uids of touch-allowed object. Defaults to [].
            allow_uids (List[int], optional): Uids of penetration-allowed object. Defaults to [].

        Returns:
            Callable: New collision function
        """
            # Configure grasped object
        #   Attached object will be moved together when searching the path
        # Optional allow pairs
        allow_pairs = []
        for allow_uid in allow_uids:            
            allow_pair_ar = LinkPair(
                body_id_a=allow_uid,
                link_id_a=None,
                body_id_b=self.robot.uid,
                link_id_b=None)
            allow_pair_ra = LinkPair(
                body_id_a=self.robot.uid,
                link_id_a=None,
                body_id_b=allow_uid,
                link_id_b=None)
            allow_pairs += [allow_pair_ar, allow_pair_ra]
        # Optional touch pairs
        touch_pairs = []
        for touch_uid in touch_uids:
            touch_pair_ar = LinkPair(
                body_id_a=touch_uid,
                link_id_a=None,
                body_id_b=self.robot.uid,
                link_id_b=None)
            touch_pair_ra = LinkPair(
                body_id_a=self.robot.uid,
                link_id_a=None,
                body_id_b=touch_uid,
                link_id_b=None)
            touch_pairs += [touch_pair_ar, touch_pair_ra]

        # Compose collision fn
        collision_fn = ContactBasedCollision(
            bc           = self.bc,
            robot_id     = self.robot.uid,
            joint_ids    = self.robot.nonfixed_joint_indices_arm,
            attachlist   = [],
            allowlist    = [*allow_pairs],
            touchlist    = [*touch_pairs],
            joint_limits = self.robot.nonfixed_joint_limits,
            tol          = {},
            touch_tol    = self.TOUCH_TOL)   #TODO: config

        return collision_fn


    def define_grasp_collision_fn(
            self, 
            grasped_uid: int, 
            touch_uids: List[int] = [],
            allow_uids: List[int] = []
    ) -> Callable:
        """Get new collision function for grasping.

        Args:
            grasped_uid (int): Uid of grasped object
            touch_uids (List[int], optional): Uids of touch-allowed object. Defaults to [].
            allow_uids (List[int], optional): Uids of penetration-allowed object. Defaults to [].

        Returns:
            Callable: New collision function
        """
        # Configure grasped object
        #   Attached object will be moved together when searching the path
        grasp_attach_pair = LinkPair(
            body_id_a=self.robot.uid,
            link_id_a=self.robot.link_index_endeffector_base,
            body_id_b=grasped_uid,
            link_id_b=-1)
        #   Allow pair is not commutative
        grasp_allow_pair_a = LinkPair(
            body_id_a=self.robot.uid,
            link_id_a=None,
            body_id_b=grasped_uid,
            link_id_b=None)
        grasp_allow_pair_b = LinkPair(
            body_id_a=grasped_uid,
            link_id_a=None,
            body_id_b=self.robot.uid,
            link_id_b=None)
        # Optional allow pairs
        allow_pairs = []
        for allow_uid in allow_uids:            
            allow_pair_ab = LinkPair(   
                body_id_a=allow_uid,
                link_id_a=None,
                body_id_b=grasped_uid,
                link_id_b=None)
            allow_pair_ba = LinkPair(   
                body_id_a=grasped_uid,
                link_id_a=None,
                body_id_b=allow_uid,
                link_id_b=None)
            allow_pair_ar = LinkPair(
                body_id_a=allow_uid,
                link_id_a=None,
                body_id_b=self.robot.uid,
                link_id_b=None)
            allow_pair_ra = LinkPair(
                body_id_a=self.robot.uid,
                link_id_a=None,
                body_id_b=allow_uid,
                link_id_b=None)
            allow_pairs += [allow_pair_ab, allow_pair_ba, 
                            allow_pair_ar, allow_pair_ra]
        # Optional touch pairs
        touch_pairs = []
        for touch_uid in touch_uids:
            touch_pair_ab = LinkPair(
                body_id_a=touch_uid,
                link_id_a=None,
                body_id_b=grasped_uid,
                link_id_b=None)
            touch_pair_ba = LinkPair(
                body_id_a=grasped_uid,
                link_id_a=None,
                body_id_b=touch_uid,
                link_id_b=None)
            touch_pair_ar = LinkPair(
                body_id_a=touch_uid,
                link_id_a=None,
                body_id_b=self.robot.uid,
                link_id_b=None)
            touch_pair_ra = LinkPair(
                body_id_a=self.robot.uid,
                link_id_a=None,
                body_id_b=touch_uid,
                link_id_b=None)
            touch_pairs += [touch_pair_ab, touch_pair_ba,
                            touch_pair_ar, touch_pair_ra]

        # Compose collision fn
        collision_fn = ContactBasedCollision(
            bc           = self.bc,
            robot_id     = self.robot.uid,
            joint_ids    = self.robot.nonfixed_joint_indices_arm,
            attachlist   = [grasp_attach_pair],
            allowlist    = [grasp_allow_pair_a, grasp_allow_pair_b, *allow_pairs],
            touchlist    = [*touch_pairs],
            joint_limits = self.robot.nonfixed_joint_limits,
            tol          = {},
            touch_tol    = self.TOUCH_TOL)   #TODO: config

        return collision_fn


    def move(self, traj: List[npt.NDArray], real_time=True):
        """Move the robot along the trajectory

        Args:
            traj (List[npt.NDArray]): Trajectory of nonfixed robot arm joints.
        """
        # Execute the trajectory
        for joint_pos in traj:
            # Control
            self.robot.update_arm_control(joint_pos)
            self.bc.stepSimulation()
            # Visualize
            if self.DEBUG_SHOW_GUI and self.DEBUG_DELAY_GUI:
                time.sleep(self.DEBUG_DELAY_GUI)
        # Wait until control completes
        self.wait(steps=240, real_time=real_time)
    

    def wait(self, steps=-1, real_time=True):
        """General wait function for hold or stabilization.

        Args:
            steps (int): Infinite loop when -1. Defaults to -1.
        """
        while steps != 0: 
            self.bc.stepSimulation()
            # Visualize
            if self.DEBUG_SHOW_GUI and self.DEBUG_DELAY_GUI and real_time:
                time.sleep(self.DEBUG_DELAY_GUI)
            steps -= 1
        

    def solve_ik_numerical(self, pos: TranslationT, 
                                 orn_q: QuaternionT) -> Tuple[npt.NDArray, npt.NDArray]:
        """Solve the inverse kinematics of the robot given the end-effector base position.
        
        Args:
            pos (TranslationT) : R^3 position
            orn_q (QuaternionT): Quaternion
        
        Returns:
            joint_pos_cur (npt.NDArray): Current position of NONFIXED arm joints in bullet. 
            joint_pos_tgt (npt.NDArray): Target position of NONFIXED arm joints from IK.
        """
        # Get current nonfixed joint state first.
        joint_pos_cur = np.array(get_joint_positions(
            self.bc, self.robot.uid,
            self.robot.nonfixed_joint_indices_arm))
        # Reset joint to the rest pose. This will increase the ik stability.
        with reset_and_restore_robot(bc    = self.bc, 
                                     robot = self.robot,
                                     joint_indices     = self.robot.nonfixed_joint_indices_arm,
                                     joint_pos_reset   = self.robot.rest_pose[self.robot.nonfixed_joint_indices_arm],
                                     joint_pos_restore = joint_pos_cur):
            ik = self.bc.calculateInverseKinematics(
                bodyUniqueId         = self.robot.uid, 
                endEffectorLinkIndex = self.robot.link_index_endeffector_base, 
                targetPosition       = pos, 
                targetOrientation    = orn_q,
                lowerLimits          = self.robot.nonfixed_joint_limits[0],     # Upper
                upperLimits          = self.robot.nonfixed_joint_limits[1],     # Lower
                jointRanges          = self.robot.nonfixed_joint_ranges,
                restPoses            = self.robot.rest_pose[self.robot.nonfixed_joint_indices_arm],
                maxNumIterations     = self.IK_MAX_NUM_ITERATIONS,
                residualThreshold    = self.IK_RESIDUAL_THRESHOLD)
        # Set destination nonfixed joint positions.
        joint_pos_tgt = np.array(ik)[:len(self.robot.nonfixed_joint_indices_arm)]
        
        return joint_pos_cur, joint_pos_tgt


    def plan_motion_trajectory(self, joint_pos_src: npt.NDArray, 
                                     joint_pos_dst: npt.NDArray,
                                     grasped_obj_uid: Optional[int] = None,
                                     touch_uids: List[int] = [],
                                     allow_uids: List[int] = [],
                                     use_interpolation: bool = False,
                                     star: bool = False
                                     ) -> List[npt.NDArray]|None:
        """Get a motion plan trajectory.

        Args:
            joint_pos_src (npt.NDArray): Source NONFIXED arm joint position
            joint_pos_dst (npt.NDArray): Destination NONFIXED arm joint position
            grasped_obj_uid (Optional[int], optional): Uid of grapsed object. Defaults to None.
            touch_uids (List[int], optional): Uids of touch-allowed object. Defaults to None.
            allow_uids (List[int], optional): Uids of penetration-allowed object. Defaults to None.
            use_interpolation (bool, optional): Return linear interpolated trajectory when True. Defaults to False.
            star (bool, optional): ...
        Returns:
            List[npt.NDArray]|None: Generated trajectory. None when no feasible trajectory is found.
        """
        # Plan linear interpolated trajectory
        if use_interpolation:
            trajectory = get_linear_interpolated_trajectory(
                cur             = joint_pos_src, 
                goal            = joint_pos_dst, 
                action_duration = 5.0, 
                control_dt      = self.CONTROL_DT)
        # Plan RRT trajectory
        else:
            with imagine(self.bc):
                # Compose collision function
                if grasped_obj_uid is not None:
                    collision_fn = self.define_grasp_collision_fn(grasped_obj_uid, touch_uids, allow_uids)
                else:
                    collision_fn = self.define_collision_fn(touch_uids, allow_uids)
                # Plan
                if star:
                    def star_sample_fn():
                        fwd = self.default_sample_fn()
                        bwd = joint_pos_dst + np.random.normal(0, 0.05, size=len(joint_pos_dst))
                        # return fwd if np.random.uniform() > 0.3 else bwd
                        return fwd if np.random.uniform() > 0.15 else bwd        
                    def star_extend_fn(q0: np.ndarray, q1: np.ndarray):
                        dq = np.subtract(q1, q0)  # Nx6
                        dq_norm = np.linalg.norm(dq, axis=-1, keepdims=True)
                        dq = dq/dq_norm * dq_norm.clip(0,0.1)
                        return q0 + np.linspace(0, 1, num=40)[:, None] * dq
                    waypoints = rrt_star.rrt_star(
                        joint_pos_src,
                        joint_pos_dst,
                        self.default_distance_fn,
                        star_sample_fn,
                        star_extend_fn,
                        collision_fn,
                        # radius=0.2,
                        K=5,
                        max_iterations=self.RRT_STAR_MAX_ITERATIONS,
                        # goal_probability=0.1,
                        informed=False
                    )
                    trajectory = way_points_to_trajectory_np(np.array(waypoints), 200)
                else:
                    trajectory = rrt_connect.birrt(
                        joint_pos_src,
                        joint_pos_dst,
                        self.default_distance_fn,
                        self.default_sample_fn,
                        self.default_extend_fn,
                        collision_fn,
                        restarts=self.RRT_RESTARTS,
                        iterations=self.RRT_ITERATIONS)
            
        return trajectory
    

    def draw_trajectory(self, traj: List[npt.NDArray]) -> List[int]:
        """Draw trajectory

        Args:
            traj (List[npt.NDArray]): Trajectory to draw

        Returns:
            List[int]: Line uids
        """
        list_ee_pos = []
        with imagine(self.bc):
            for q in traj:
                for i, v in zip(self.robot.nonfixed_joint_indices_arm, q):
                    self.bc.resetJointState(self.robot.uid, i, v)
                list_ee_pos.append(get_link_pose(self.bc, self.robot.uid, self.robot.link_index_endeffector_base)[0])
        
        line_uids = []
        for prv_ee_pos, cur_ee_pos in zip(list_ee_pos[:-2], list_ee_pos[1:]):
             line_uid = self.bc.addUserDebugLine(
                prv_ee_pos,
                cur_ee_pos,
                lineColorRGB = [0.8, 0, 0.8],
                lineWidth    = 5)
             line_uids.append(line_uid)

        return line_uids
    


class FrankaManipulation(Manipulation):

    def __init__(self, bc    : BulletClient,
                       robot : FrankaPanda,
                       config: dict): 
        self.robot: FrankaPanda
        super().__init__(bc, robot, config)
        # Manipulation params
        manipulation_params = config["manipulation_params"]
        self.IK_MAX_TRIALS     = 100
        self.IK_MAX_VARIATIONS = manipulation_params["inverse_kinematics"]["max_num_variations"]
        # Debug
        self.draw_tgt_coordinate = CoordinateVisualizer(self.bc, brightness=0.7)


    def open_with_constraint(self):
        """Open the gripper and remove constraint"""
        # Remove constraint
        self.robot.remove_grasp_constraint()
        # Open
        self.robot.release()
        # Wait until control completes
        self.wait(steps=240)


    def close_with_constraint(self, graspable_obj_uids: List[int]) -> int|None:
        """
        Args:
            graspable_obj_uids(List[int]): Uids of graspable objects

        Returns:
            int|None: Returns uid of object with min-dist contact. Returns to None when not in contact.
        """
        # Close
        self.robot.activate()
        for i in range(240):
            # Wait until control completes
            self.wait(steps=1)
            # Get uid with the min dist.
            cps_all = self.robot.detect_gripper_contact()
            # Keep track of contacts of object with fingers.
            cp_uids_in_contact_with_finger = {
                cp[2]: {fi: False for fi in self.robot.finger_indices}
                for cp in cps_all}
            # Parse all contacts
            for cp in cps_all:
                cp_uid, cp_robot_link, cp_dist = cp[2], cp[3], cp[8]
                if cp_uid not in graspable_obj_uids:
                    continue
                if cp_dist < 0:
                    cp_uids_in_contact_with_finger[cp_uid][cp_robot_link] = True
            # Break when there exists an object that contacts both fingers.
            contact_obj_uid = None
            for cp_uid, contacts in cp_uids_in_contact_with_finger.items():
                if all(contacts.values()) == True:
                    contact_obj_uid = cp_uid
                    break            
            if contact_obj_uid is not None:
                break

        # Heuristic grasp
        self.robot.freeze()
        if contact_obj_uid is not None:
            self.robot.create_grasp_constraint(contact_obj_uid)
        return contact_obj_uid



    def solve_ik_analytical(self, pos: TranslationT, 
                                  orn_q: QuaternionT) -> Tuple[npt.NDArray, npt.NDArray|None]:
        """Solve the inverse kinematics of the robot given the end-effector base position using analytic solver.
        
        Args:
            pos (TranslationT) : R^3 position
            orn_q (QuaternionT): Quaternion

        Raises:
            ValueError: Unreachable pose
        
        Returns:
            joint_pos_cur (npt.NDArray): Current position of NONFIXED arm joints in bullet. 
            joint_pos_tgt (npt.NDArray|None): Target position of NONFIXED arm joints from IK.
        """

        # Convert world frame to robot frame.
        inverse_robot_pos, inverse_robot_orn = self.bc.invertTransform(
            self.robot.base_pos, self.robot.base_orn)
        pos, orn_q = self.bc.multiplyTransforms(
            inverse_robot_pos, inverse_robot_orn,
            pos, orn_q)

        # Get current nonfixed joint state first.
        joint_pos_cur = np.array(get_joint_positions(
            self.bc, self.robot.uid,
            self.robot.nonfixed_joint_indices_arm))
        
        # Try analytic IK until the solution is found. May return NaN pos.
        for i in range(self.IK_MAX_TRIALS):
            joint_pos_tgt = franka_ik_pybind.franka_IKCC(
                np.asarray(pos), np.asarray(orn_q), 
                random.uniform(-2.8973, 2.8973), joint_pos_cur)
            if not np.isnan(np.min(joint_pos_tgt)) and joint_pos_tgt[0] > -np.pi/2 and joint_pos_tgt[0] < np.pi/2:
                break
        
        # Handle NaN IK pose.
        if np.isnan(np.min(joint_pos_tgt)):
            joint_pos_tgt = None

        return joint_pos_cur, joint_pos_tgt


    def plan_with_ik_variations(self, grasp_pos: TranslationT, 
                                      grasp_orn: QuaternionT,
                                      debug: bool = False,
                                      **kwargs) -> List[npt.NDArray]|None:
        """Combined IK and planning

        Args:
            grasp_pos (TranslationT): _
            grasp_orn (QuaternionT): _
            debug (bool): Show target coordinate

        Returns:
            List[npt.NDArray]|None: Trajectory if exists.
        """
        if debug:
            self.draw_tgt_coordinate.draw(grasp_pos, grasp_orn)
        # Try different IKs for the same grasp pose.
        traj = None
        for i in range(self.IK_MAX_VARIATIONS):
            # Try IK
            joint_pos_cur, joint_pos_tgt = self.solve_ik_analytical(grasp_pos, grasp_orn)
            if joint_pos_tgt is None:
                continue
            # Try planning if IK exists.
            traj = self.plan_motion_trajectory(joint_pos_cur, joint_pos_tgt, **kwargs)
            if traj is not None:
                break                   
        
        # None if IK not exists or plan not exists
        return traj


    def resolve_end_effector_collision(self, joint_pos: npt.NDArray, max_steps=1000, min_dist = 0.01) -> npt.NDArray|None:
        """Adjust the end effector pose to resolve collision
        
        Args:
            joint_pos (npt.NDArray): Joint position that is possibly in collision

        Return:
            npt.NDArray|None: New joint position. None if failed.
        """


        current_joint_pos = np.copy(joint_pos)
        new_joint_pos = None

        for i in range(max_steps):
            # Use context for save 
            with imagine(self.bc):
                # Parse contact points
                self.robot.reset(current_joint_pos)
                self.bc.performCollisionDetection()    
                contact_points = []
                link_indices_to_check = [12] + list(self.robot.nonfixed_joint_indices_arm[-3:]) + [self.robot.link_index_endeffector_base] + self.robot.finger_indices.tolist() # camera
                for link_i in link_indices_to_check:
                    points = self.bc.getContactPoints(bodyA=self.robot.uid, linkIndexA=link_i)
                    contact_points += points

                # Aggregate contact normals and scales
                scaled_normal_list = []
                for point in contact_points:
                    normal_world_frame = point[7]
                    normal_scale = point[8]
                    if normal_scale < 0:
                        scaled_normal = np.asarray(normal_world_frame) * -normal_scale
                        scaled_normal_list.append(scaled_normal)

                # End if no collision exists.                 
                if len(scaled_normal_list) == 0:
                    new_joint_pos = current_joint_pos
                    break

                # Calculate new pose and ik
                # Numerical IK is appropriate as it can optimize from the current position.
                scaled_normal_list = np.array(scaled_normal_list)
                delta = np.average(scaled_normal_list, axis=0)
                delta += (delta/np.linalg.norm(delta))*min_dist

                ee_pos, ee_orn = self.robot.get_endeffector_pose()
                ee_new_pose = np.array(ee_pos) + delta
                ik = self.bc.calculateInverseKinematics(
                    bodyUniqueId         = self.robot.uid, 
                    endEffectorLinkIndex = self.robot.link_index_endeffector_base, 
                    targetPosition       = ee_new_pose, 
                    targetOrientation    = ee_orn,
                    lowerLimits          = self.robot.nonfixed_joint_limits[0],     # Upper
                    upperLimits          = self.robot.nonfixed_joint_limits[1],     # Lower
                    jointRanges          = self.robot.nonfixed_joint_ranges,
                    restPoses            = joint_pos,
                    maxNumIterations     = self.IK_MAX_NUM_ITERATIONS,
                    residualThreshold    = self.IK_RESIDUAL_THRESHOLD)

                current_joint_pos = np.array(ik)[self.robot.nonfixed_joint_indices_arm]

        return new_joint_pos



def way_points_to_trajectory_np(waypnts, resolution):
    """copy pasted from experiments.exp_util"""

    # # goal reaching traj
    # goal = waypnts[-1]
    # traj_norm = np.linalg.norm(goal - waypnts, axis=-1)
    # traj_min_idx = np.min(np.arange(traj_norm.shape[0], dtype=np.int32)*(traj_norm<0.04).astype(np.int32) + 100000*(traj_norm>=0.04).astype(np.int32))
    # waypnts = waypnts[:traj_min_idx+1]
    # waypnts = np.concatenate([waypnts, goal[None]], axis=0)

    wp_len = np.linalg.norm(waypnts[1:] - waypnts[:-1], axis=-1)
    wp_len = wp_len/np.sum(wp_len).clip(1e-5)
    wp_len = np.where(wp_len<1e-4, 0, wp_len)
    wp_len_cumsum = np.cumsum(wp_len)
    wp_len_cumsum = np.concatenate([np.array([0]),wp_len_cumsum], 0)
    wp_len_cumsum[-1] = 1.0
    indicator = np.linspace(0, 1, resolution)
    indicator = (-np.cos(indicator*np.pi)+1)/2.
    included_idx = np.sum(indicator[...,None] > wp_len_cumsum[1:], axis=-1)
    included_idx = included_idx.clip(0, included_idx.shape[0]-2)

    upper_residual = (wp_len_cumsum[included_idx+1] - indicator)/wp_len[included_idx].clip(1e-5)
    bottom_residual = (indicator - wp_len_cumsum[included_idx])/wp_len[included_idx].clip(1e-5)


    traj = waypnts[included_idx] * upper_residual[...,None] + waypnts[included_idx+1] * bottom_residual[...,None]
    traj = np.where(wp_len[included_idx][...,None] < 1e-4, waypnts[included_idx], traj)
    traj[0] = waypnts[0]
    traj[-1] = waypnts[-1]
    
    return traj