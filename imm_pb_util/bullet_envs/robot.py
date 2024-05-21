from pathlib import Path
from typing import Tuple, Dict, List, Set, Optional
import numpy as np
import numpy.typing as npt
from abc import ABC, abstractmethod
from copy import deepcopy
import time
from scipy.spatial.transform import Rotation as sciR

import pybullet_data
from pybullet_utils.bullet_client import BulletClient
from imm.pybullet_util.typing_extra import TranslationT, QuaternionT, EulerT
from imm.pybullet_util.common import ( get_joint_limits, get_joint_positions, get_link_pose, get_relative_transform )
from imm.pybullet_util.vision import CoordinateVisualizer


class BulletRobot(ABC):
    """
    This is an abstract class that holds the 
    common properties of manipulator robots.
    """

    def __init__(self, bc: BulletClient,
                       urdf_file_path: str,
                       base_pos: TranslationT,
                       base_orn: QuaternionT,
                       use_fixed_base: bool,
                       link_index_endeffector_base: int,
                       nonfixed_joint_indices_arm: npt.NDArray,
                       rest_pose: npt.NDArray):
        """Configure common properties

        Args:
            bc (BulletClient): _
            urdf_file_path (str): _
            base_pos (TranslationT): _
            base_orn (QuaternionT): _
            use_fixed_base (bool): _
            link_index_endeffector_base (int): Link index of endeffector base
            num_joints_all (int): Number of all type joints
            nonfixed_joint_indices_arm (npt.NDArray): NON-FIXED joint indices of the manipulator arm.
            rest_pose (npt.NDArray): Default rest pose
        """

        # Pybullet properties
        self.bc : BulletClient = bc
        self.uid: int          = bc.loadURDF(
            fileName        = urdf_file_path,
            basePosition    = base_pos,
            baseOrientation = base_orn,
            useFixedBase    = use_fixed_base,
            flags           = bc.URDF_ENABLE_CACHED_GRAPHICS_SHAPES
        )
        self.base_pos: TranslationT = base_pos  # NOTE getBasePositionAndOrientation will return base inertial pose.
        self.base_orn: QuaternionT  = base_orn

        # URDF properties
        # NOTE: Usage -> idx, value = zip(nonfixed_joint_indices_arm, nonfixed_joint_**)
        self.link_index_endeffector_base: int         = link_index_endeffector_base
        self.nonfixed_joint_indices_arm : npt.NDArray = nonfixed_joint_indices_arm
        self.nonfixed_joint_limits      : npt.NDArray = get_joint_limits(self.bc, self.uid, self.nonfixed_joint_indices_arm)    # Lower, Upper
        self.nonfixed_joint_ranges      : npt.NDArray = self.nonfixed_joint_limits[1] - self.nonfixed_joint_limits[0]
        
        # Pose targets (Access with `rest_pose[nonfixed_joint_indices_arm]`)
        self.rest_pose: npt.NDArray = rest_pose             # Rest pose
        self.last_pose: npt.NDArray = np.copy(rest_pose)    # Last pose is for control input

        # Reset robot with initialization
        for i in self.nonfixed_joint_indices_arm:
            self.bc.resetJointState(self.uid, i, self.last_pose[i])
        BulletRobot.update_arm_control(self)

        # Flags
        self.contact_constraint = None


    def update_arm_control(self, target_arm_joint_positions: Optional[npt.NDArray] = None):
        """
        Update only the control of the robot arm. Not the end-effector.
        If the parameters are not given, it will reinstate the positional control from the last_pose.
        DO NOT DIRECTLY PUT THE IK SOLUTION HERE. 
        IK solution ignores fixed joint indices and contains random finger joint value.

        Args:
            target_arm_joint_positions(Optional[npt.NDArray]): Positions of NON-FIXED arm joints. Holds current target position when None.
        """
        # Update last pose
        if target_arm_joint_positions is not None:
            self.last_pose[self.nonfixed_joint_indices_arm] = target_arm_joint_positions
        # Set control. This must be called at every stepSimulation.
        self.bc.setJointMotorControlArray(bodyUniqueId    = self.uid, 
                                          jointIndices    = self.nonfixed_joint_indices_arm,
                                          controlMode     = self.bc.POSITION_CONTROL,
                                          targetPositions = self.last_pose[self.nonfixed_joint_indices_arm])
        

    def reset(self, target_arm_joint_positions: npt.NDArray):
        """Force reset the arm joint values and controls"""
        self.last_pose[self.nonfixed_joint_indices_arm] = deepcopy(target_arm_joint_positions)
        for i, v in zip(self.nonfixed_joint_indices_arm, target_arm_joint_positions):
            self.bc.resetJointState(self.uid, i, v)
        self.update_arm_control(target_arm_joint_positions)


    def get_arm_positions(self) -> npt.NDArray:
        """Get the positions of NON-FIXED arm joints."""
        positions = get_joint_positions(self.bc, self.uid, self.nonfixed_joint_indices_arm)
        return np.asarray(positions)
    

    def get_endeffector_pose(self) -> Tuple[TranslationT, QuaternionT]:
        """Get the current pose of the end-effector from forward kinematics"""
        pos, orn_q = get_link_pose(self.bc, self.uid, self.link_index_endeffector_base)
        return pos, orn_q


    @abstractmethod
    def activate():
        """Activate gripper"""
        pass


    @abstractmethod
    def release():
        """Release gripper"""
        pass


    @abstractmethod
    def freeze():
        """Freeze the gripper fingers to current value"""
        pass


    @abstractmethod
    def detect_gripper_contact() -> List:
        """Detect gripper contact"""
        pass

    
    def create_grasp_constraint(self, target_uid: int, 
                                      target_link_idx: int = -1): 
        """Create grasp constraint between the end-effector base and target object.
        
        Args:
            target_uid (int): _
            target_lind_idx (int): Defaults to -1.

        Raises:
            RuntimeError: If called twice.
        """
        if self.contact_constraint is None:
            link_rel_pos, link_rel_orn = get_relative_transform(
                bc        = self.bc, 
                body_id_a = self.uid, 
                link_a    = self.link_index_endeffector_base, 
                body_id_b = target_uid,
                link_b    = target_link_idx,
                inertial  = True)
            
            self.contact_constraint = self.bc.createConstraint(
                parentBodyUniqueId     = self.uid,
                parentLinkIndex        = self.link_index_endeffector_base,
                childBodyUniqueId      = target_uid,
                childLinkIndex         = target_link_idx,
                jointType              = self.bc.JOINT_FIXED,
                jointAxis              = (0, 0, 0),
                parentFramePosition    = link_rel_pos,
                parentFrameOrientation = link_rel_orn,
                childFramePosition     = (0, 0, 0),
                childFrameOrientation  = (0, 0, 0, 1))
        else:
            raise RuntimeError("Gripper constraint already exists")

    
    def remove_grasp_constraint(self):
        """Release grapsed rigid object (if any)"""
        if self.contact_constraint is not None:
            self.bc.removeConstraint(self.contact_constraint)
            self.contact_constraint = None




class FrankaPanda(BulletRobot):
    """
    Wrapper class for Franka Panda in PyBullet
    """

    def __init__(self, bc: BulletClient,
                       config: Dict):
        """Configure Franka Panda.

        Args:
            bc (BulletClient): Pybullet client
            config (Dict): Global configuration
        """
        # Config
        robot_params = config["robot_params"]["franka_panda"]
        
        # Path to URDFs
        # urdf_path = Path(pybullet_data.getDataPath())
        custom_urdf_path = config["project_params"]["custom_urdf_path"]
        project_path = Path(__file__).parent.parent
        project_urdf_data_path = project_path / custom_urdf_path
        urdf_path = project_urdf_data_path / "franka_panda_custom" / "panda.urdf"
        
        # Init robot superclass
        super().__init__(
            bc = bc,
            urdf_file_path = str(urdf_path),
            base_pos       = robot_params["base_pos"],
            base_orn       = bc.getQuaternionFromEuler(robot_params["base_orn"]),
            use_fixed_base = True,
            link_index_endeffector_base = robot_params["link_index_endeffector_base"],
            nonfixed_joint_indices_arm  = np.asarray(robot_params["nonfixed_joint_indices_arm"]),
            rest_pose                   = np.asarray(robot_params["rest_pose"]))

        # Gripper
        self.finger_indices: npt.NDArray = np.asarray(robot_params["finger_indices"])
        self.finger_travel : npt.NDArray = np.asarray(robot_params["finger_travel"])
        self.finger_force  : float       = robot_params["finger_force"]
        for i in self.finger_indices:
            self.bc.resetJointState(self.uid, i, self.last_pose[i])
        self.release()

    def reset_finger(self, grip_len):
        for i in self.finger_indices:
            self.bc.resetJointState(self.uid, i, grip_len)
        self.bc.setJointMotorControlArray(
            bodyUniqueId    = self.uid, 
            jointIndices    = self.finger_indices,
            controlMode     = self.bc.POSITION_CONTROL,
            targetPositions = [grip_len,grip_len])
        
    def activate(self):
        """Close the gripper"""
        # Update last pose
        self.last_pose[self.finger_indices] = 0.0
        # Set control.
        self.bc.setJointMotorControlArray(
            bodyUniqueId    = self.uid, 
            jointIndices    = self.finger_indices,
            controlMode     = self.bc.POSITION_CONTROL,
            targetPositions = self.last_pose[self.finger_indices])
        

    def close_gripper(self, duration=1.0, sleep=False):
        """Close the gripper"""
        # Update last pose
        dt = self.bc.getPhysicsEngineParameters()['fixedTimeStep']
        maxforce = 400
        entire_itr = int(duration/dt)
        fix_grip_width = 0
        cnt = 0
        one_step_width = self.finger_travel/2.0/entire_itr
        for i in range(entire_itr+1):
            # Set control.
            self.last_pose[self.finger_indices] = np.maximum(self.finger_travel/2.0*((entire_itr-cnt)/entire_itr), fix_grip_width-6.0*one_step_width)
            self.bc.setJointMotorControlArray(
                bodyUniqueId    = self.uid, 
                jointIndices    = self.finger_indices,
                controlMode     = self.bc.POSITION_CONTROL,
                forces = [maxforce, maxforce],
                targetPositions = self.last_pose[self.finger_indices])
            self.bc.stepSimulation()
            # if fix_grip_width==0:
            tip1_contacts = [cp for cp in self.bc.getContactPoints(bodyA=self.uid) if cp[3] == self.finger_indices[0]]
            tip2_contacts = [cp for cp in self.bc.getContactPoints(bodyA=self.uid) if cp[3] == self.finger_indices[1]]
            if len(tip1_contacts)!=0 and len(tip1_contacts)!=0:
                tip1_contact_force = np.sum([cp[9] for cp in tip1_contacts])
                tip2_contact_force = np.sum([cp[9] for cp in tip2_contacts])
                if tip1_contact_force > 300 and tip2_contact_force > 300:
                    fix_grip_width = self.finger_travel/2.0*((entire_itr-cnt)/entire_itr)
                else:
                    cnt += 1
            else:
                cnt += 1
            if sleep:
                time.sleep(dt)

    def release(self):
        """Open the gripper"""
        # Update last pose
        self.last_pose[self.finger_indices] = self.finger_travel/2.0
        # Set control.
        self.bc.setJointMotorControlArray(
            bodyUniqueId    = self.uid, 
            jointIndices    = self.finger_indices,
            controlMode     = self.bc.POSITION_CONTROL,
            targetPositions = self.last_pose[self.finger_indices])
    

    def freeze(self):
        """Freeze the gripper fingers to current value"""
        # Update the last pose
        joints = get_joint_positions(self.bc, self.uid, self.finger_indices)
        self.last_pose[self.finger_indices] = joints
        # Set control.
        self.bc.setJointMotorControlArray(
            bodyUniqueId    = self.uid, 
            jointIndices    = self.finger_indices,
            controlMode     = self.bc.POSITION_CONTROL,
            targetPositions = self.last_pose[self.finger_indices])

    
    def detect_gripper_contact(self) -> List:
        """Scan penetration contact points in gripper.

        Returns:
            List: PyBullet contact points info
        """
        self.bc.performCollisionDetection()
        cps_all = []
        for i in self.finger_indices:
            cps_new = [
                cp 
                for cp in self.bc.getContactPoints(bodyA=self.uid, linkIndexA=i) 
                if cp[8]<=0 and cp[2]!=self.uid  # Penetration only
            ] 
            cps_all += cps_new
        
        return cps_all


    def transparent(self, alpha=0.4):
        """Set the body color to transparent"""
        for i in range(-1, self.bc.getNumJoints(self.uid)):
            self.bc.changeVisualShape(self.uid, i, rgbaColor=[1,1,1,alpha])
    

    def invisible(self):
        """Set the body color to be invisible"""
        for i in range(-1, self.bc.getNumJoints(self.uid)):
            self.bc.changeVisualShape(self.uid, i, rgbaColor=[1,1,1,0.0])


class PandaGripper:

    def __init__(
        self,
        bc: BulletClient,
        urdf_path: Path
    ):
        # Config
        self.bc = bc
        self.uid = bc.loadURDF(str(urdf_path), useFixedBase=True)
        bc.changeDynamics(self.uid, -1, rollingFriction=1, spinningFriction=1, lateralFriction=1)
        self.visualizer = CoordinateVisualizer(self.bc, 1.0)
        self.finger_indices = np.array([1, 2])
        for i in self.finger_indices:
            self.bc.resetJointState(self.uid, i, 0.04)
        
        self.virtual_joint_index = 0
        self.last_pose = np.array([0.0, 0.4, 0.4])
        self.finger_travel = 0.08

        self.bc.resetJointState(self.uid, 0, -1)
        self.reset_finger(0.04)
        self.reset_z(0.0)


    def set_z(self, z):
        self.bc.setJointMotorControl2(
            bodyUniqueId    = self.uid, 
            jointIndex    = self.virtual_joint_index,
            controlMode     = self.bc.POSITION_CONTROL,
            targetPosition = z)

    def reset_z(self, z):
        self.bc.resetJointState(self.uid, self.virtual_joint_index, z)
        self.set_z(z)


    def set_gripper_pose(self, pos, orn, gripper_frame=False, compensate_orn=True):
        if compensate_orn:
            orn_ = (sciR.from_quat(orn)*sciR.from_euler('z', -np.pi/4.)).as_quat()
        else:
            orn_ = orn
        # Transform the link frame pose to inertial frame pose
        local_inertial_pos, local_inertial_orn \
            = self.bc.getDynamicsInfo(self.uid, -1)[3:5]
        link_pos, link_orn = self.bc.multiplyTransforms(
            pos, orn_,
            local_inertial_pos, 
            local_inertial_orn)
        if gripper_frame:
            link_pos, link_orn = self.bc.multiplyTransforms(
                link_pos, link_orn,
                np.array([0,0,0.1034]), 
                np.array([0.,0,0,1]))
        self.bc.resetBasePositionAndOrientation(self.uid, link_pos, link_orn)
        # self.show_frame()

    def show_frame(self):
        pos, orn = get_link_pose(bc=self.bc, body_id=self.uid, link_id=-1)
        self.visualizer.draw(pos, orn)

    def transparent(self, alpha=0.4):
        """Set the body color to transparent"""
        for i in range(-1, self.bc.getNumJoints(self.uid)):
            self.bc.changeVisualShape(self.uid, i, rgbaColor=[1,1,1,alpha])

    def reset_finger(self, grip_len):
        for i in self.finger_indices:
            self.bc.resetJointState(self.uid, i, grip_len)
        self.bc.setJointMotorControlArray(
            bodyUniqueId    = self.uid, 
            jointIndices    = self.finger_indices,
            controlMode     = self.bc.POSITION_CONTROL,
            targetPositions = [grip_len,grip_len])
        
    def activate(self):
        """Close the gripper"""
        # Update last pose
        # self.last_pose[self.finger_indices] = 0.0
        maxforce = 80
        # Set control.
        # self.bc.setJointMotorControlArray(
        #     bodyUniqueId    = self.uid, 
        #     jointIndices    = self.finger_indices,
        #     controlMode     = self.bc.POSITION_CONTROL,
        #     targetPositions = self.last_pose[self.finger_indices])

        self.bc.setJointMotorControlArray(
            bodyUniqueId    = self.uid, 
            jointIndices    = self.finger_indices,
            controlMode     = self.bc.POSITION_CONTROL,
            targetPositions = self.last_pose[self.finger_indices],
            forces = [maxforce, maxforce])
        

    def release(self):
        """Open the gripper"""
        # Update last pose
        self.last_pose[self.finger_indices] = self.finger_travel/2.0
        # Set control.
        self.bc.setJointMotorControlArray(
            bodyUniqueId    = self.uid, 
            jointIndices    = self.finger_indices,
            controlMode     = self.bc.POSITION_CONTROL,
            targetPositions = self.last_pose[self.finger_indices])
    

    def freeze(self):
        """Freeze the gripper fingers to current value"""
        # Update the last pose
        joints = get_joint_positions(self.bc, self.uid, self.finger_indices)
        self.last_pose[self.finger_indices] = joints
        # Set control.
        self.bc.setJointMotorControlArray(
            bodyUniqueId    = self.uid, 
            jointIndices    = self.finger_indices,
            controlMode     = self.bc.POSITION_CONTROL,
            targetPositions = self.last_pose[self.finger_indices])
        


class PandaGripperVisual:

    def __init__(
        self,
        bc: BulletClient,
        urdf_path: Path
    ):
        # Config
        self.bc = bc
        self.uid = bc.loadURDF(str(urdf_path), useFixedBase=True, flags=bc.URDF_ENABLE_CACHED_GRAPHICS_SHAPES)
        bc.changeDynamics(self.uid, -1, rollingFriction=1, spinningFriction=1, lateralFriction=1)
        self.visualizer = CoordinateVisualizer(self.bc, 1.0)
        self.finger_indices = np.array([0, 1])
        for i in self.finger_indices:
            self.bc.resetJointState(self.uid, i, 0.04)
        
        self.virtual_joint_index = 0
        self.last_pose = np.array([0.0, 0.4, 0.4])
        self.finger_travel = 0.08

        self.bc.resetJointState(self.uid, 0, -1)
        self.reset_finger(0.04)
        self.reset_z(0.0)


    def set_z(self, z):
        self.bc.setJointMotorControl2(
            bodyUniqueId    = self.uid, 
            jointIndex    = self.virtual_joint_index,
            controlMode     = self.bc.POSITION_CONTROL,
            targetPosition = z)

    def reset_z(self, z):
        self.bc.resetJointState(self.uid, self.virtual_joint_index, z)
        self.set_z(z)


    def set_gripper_pose(self, pos, orn, gripper_frame=False, compensate_orn=True):
        if compensate_orn:
            orn_ = (sciR.from_quat(orn)*sciR.from_euler('z', -np.pi/4.)).as_quat()
        else:
            orn_ = orn
        # Transform the link frame pose to inertial frame pose
        local_inertial_pos, local_inertial_orn \
            = self.bc.getDynamicsInfo(self.uid, -1)[3:5]
        link_pos, link_orn = self.bc.multiplyTransforms(
            pos, orn_,
            local_inertial_pos, 
            local_inertial_orn)
        if gripper_frame:
            link_pos, link_orn = self.bc.multiplyTransforms(
                link_pos, link_orn,
                np.array([0,0,0.1034]), 
                np.array([0.,0,0,1]))
        self.bc.resetBasePositionAndOrientation(self.uid, link_pos, link_orn)
        # self.show_frame()

    def show_frame(self):
        pos, orn = get_link_pose(bc=self.bc, body_id=self.uid, link_id=-1)
        self.visualizer.draw(pos, orn)

    def transparent(self, alpha=0.4):
        """Set the body color to transparent"""
        for i in range(-1, self.bc.getNumJoints(self.uid)):
            self.bc.changeVisualShape(self.uid, i, rgbaColor=[1,1,1,alpha])

    def reset_finger(self, grip_len):
        for i in self.finger_indices:
            self.bc.resetJointState(self.uid, i, grip_len)
        self.bc.setJointMotorControlArray(
            bodyUniqueId    = self.uid, 
            jointIndices    = self.finger_indices,
            controlMode     = self.bc.POSITION_CONTROL,
            targetPositions = [grip_len,grip_len])
        
    def activate(self):
        """Close the gripper"""
        # Update last pose
        # self.last_pose[self.finger_indices] = 0.0
        maxforce = 80
        # Set control.
        # self.bc.setJointMotorControlArray(
        #     bodyUniqueId    = self.uid, 
        #     jointIndices    = self.finger_indices,
        #     controlMode     = self.bc.POSITION_CONTROL,
        #     targetPositions = self.last_pose[self.finger_indices])

        self.bc.setJointMotorControlArray(
            bodyUniqueId    = self.uid, 
            jointIndices    = self.finger_indices,
            controlMode     = self.bc.POSITION_CONTROL,
            targetPositions = self.last_pose[self.finger_indices],
            forces = [maxforce, maxforce])
        

    def release(self):
        """Open the gripper"""
        # Update last pose
        self.last_pose[self.finger_indices] = self.finger_travel/2.0
        # Set control.
        self.bc.setJointMotorControlArray(
            bodyUniqueId    = self.uid, 
            jointIndices    = self.finger_indices,
            controlMode     = self.bc.POSITION_CONTROL,
            targetPositions = self.last_pose[self.finger_indices])
    

    def freeze(self):
        """Freeze the gripper fingers to current value"""
        # Update the last pose
        joints = get_joint_positions(self.bc, self.uid, self.finger_indices)
        self.last_pose[self.finger_indices] = joints
        # Set control.
        self.bc.setJointMotorControlArray(
            bodyUniqueId    = self.uid, 
            jointIndices    = self.finger_indices,
            controlMode     = self.bc.POSITION_CONTROL,
            targetPositions = self.last_pose[self.finger_indices])
        