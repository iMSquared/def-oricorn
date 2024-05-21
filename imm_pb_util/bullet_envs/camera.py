from pathlib import Path
from typing import Dict, Tuple, List, Iterable, Optional
import numpy as np
import numpy.typing as npt
from scipy.spatial.transform import Rotation as sciR
from pathlib import Path
from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np
from scipy.spatial.transform import Rotation as R
import math
import open3d as o3d


import pybullet_data
from pybullet_utils.bullet_client import BulletClient

from bullet_envs.robot import BulletRobot
from bullet_envs.manipulation import Manipulation

from imm.pybullet_util.typing_extra import TranslationT, QuaternionT, EulerT
from imm.pybullet_util.vision import (
    compute_gl_view_matrix_from_gl_camera_pose,
    compute_gl_camera_pose_from_gl_view_matrix,
    compute_opencv_extrinsic_matrix_from_gl_camera_pose,
    compute_intrinsic_matrix_from_vertical_fov,
    compute_vfov_from_intrinsics,
    compute_parameters_from_projection_matrix,
    CoordinateVisualizer)



@dataclass(frozen=True)
class CameraExtrinsic:
    camera_pos     : TranslationT
    camera_quat    : QuaternionT
    gl_view_matrix : npt.NDArray
    cv2_extr_matrix: npt.NDArray


@dataclass(frozen=True)
class CameraIntrinsic:
    fov              : float
    width            : int
    height           : int
    aspect           : float
    focusing_distance: float
    intr_matrix      : npt.NDArray
    gl_near_val      : float
    gl_far_val       : float
    gl_proj_matrix   : npt.NDArray

    @property
    def formatted(self) -> npt.NDArray:
        fx = self.intr_matrix[0,0]
        fy = self.intr_matrix[1,1]
        cx = self.intr_matrix[0,2]
        cy = self.intr_matrix[1,2]
        return np.array([self.width, self.height, fx, fy, cx, cy])


class BulletCamera:
    """
    This is an abstract class that holds the 
    common properties of bullet camera.
    """
    def __init__(
            self, 
            bc: BulletClient, 
            extrinsic: CameraExtrinsic,
            intrinsic: CameraIntrinsic, 
    ):
        self.bc = bc
        self.intrinsic = intrinsic
        self.extrinsic = extrinsic
        self.pose_frame_visualizer = CoordinateVisualizer(self.bc, 1.0)


    @staticmethod
    def convert_bullet_properties(
            bc: BulletClient,
            extr_gl_pos: TranslationT,
            extr_gl_orn: QuaternionT, 
            intr_fov: float,
            intr_width: int,
            intr_height: int,
            intr_focusing_distance: float,
            intr_gl_near_val: float,
            intr_gl_far_val: float
    ):
        """Define extrinsic and intrinsics

        Args:
            bc (BulletClient): _
            extr_camera_pos (TranslationT): _
            extr_camera_orn (QuaternionT): _
            intr_fov (float): _
            intr_width (int): _
            intr_height (int): _
            intr_focusing_distance (float): _
            intr_gl_near_val (float): _
            intr_gl_far_val (float): _
        """
        # Compose matrices
        # OpenGL
        gl_view_matrix = compute_gl_view_matrix_from_gl_camera_pose(
            bc  = bc, 
            pos = extr_gl_pos,
            orn = extr_gl_orn)
        gl_proj_matrix = bc.computeProjectionMatrixFOV(
            fov     = intr_fov,
            aspect  = intr_width/intr_height,
            nearVal = intr_gl_near_val,
            farVal  = intr_gl_far_val)
        # OpenCV
        cv2_extr_matrix = compute_opencv_extrinsic_matrix_from_gl_camera_pose(
            bc  = bc,
            pos = extr_gl_pos,
            orn = extr_gl_orn)
        intr_matrix = compute_intrinsic_matrix_from_vertical_fov(
            vfov_degree = intr_fov, 
            w = intr_width, 
            h = intr_height)

        # Set attributes
        extrinsic = CameraExtrinsic(
            camera_pos      = extr_gl_pos,
            camera_quat     = extr_gl_orn,
            gl_view_matrix  = gl_view_matrix,
            cv2_extr_matrix = cv2_extr_matrix)
        intrinsic = CameraIntrinsic(
            fov               = intr_fov,
            width             = intr_width,
            height            = intr_height,
            aspect            = intr_width/intr_height,
            focusing_distance = intr_focusing_distance,
            intr_matrix       = intr_matrix,
            gl_near_val       = intr_gl_near_val,
            gl_far_val        = intr_gl_far_val,
            gl_proj_matrix    = gl_proj_matrix)

        return extrinsic, intrinsic


    @staticmethod
    def convert_gl_properties(
            bc: BulletClient,
            extr_gl_pos: TranslationT,
            extr_gl_orn: QuaternionT, 
            intr_width: int,
            intr_height: int,
            intr_fx: float,
            intr_fy: float,
            intr_cx: float,
            intr_cy: float,
            intr_focusing_distance: float,
            intr_gl_near_val: float,
            intr_gl_far_val: float
    ):
        if type(intr_width) != int or type(intr_height) != int:
            raise TypeError("Image size must be integer.")

        # OpenGL
        gl_view_matrix = compute_gl_view_matrix_from_gl_camera_pose(
            bc  = bc, 
            pos = extr_gl_pos,
            orn = extr_gl_orn
        )
        intr_fov = compute_vfov_from_intrinsics(intr_width, intr_height, intr_fx, intr_fy, intr_cx, intr_cy)
        # gl_proj_matrix = bc.computeProjectionMatrixFOV(
        #     fov     = intr_fov,
        #     aspect  = intr_width/intr_height,
        #     nearVal = intr_gl_near_val,
        #     farVal  = intr_gl_far_val)
        gl_proj_matrix = [
            2*intr_fx/intr_width, 0, 0, 0, 
            0, 2*intr_fy/intr_height, 0, 0, 
            -2*(intr_cx/intr_width)+1, -2*(intr_cy/intr_height)+1, -(intr_gl_far_val+intr_gl_near_val)/(intr_gl_far_val-intr_gl_near_val), -1,
            0, 0, -2*intr_gl_far_val*intr_gl_near_val/(intr_gl_far_val-intr_gl_near_val), 0]
        gl_proj_matrix = [
            2*intr_fx/intr_width, 0, 0, 0, 
            0, 2*intr_fy/intr_height, 0, 0, 
            2*(intr_cx/intr_width)-1, 2*(intr_cy/intr_height)-1, -(intr_gl_far_val+intr_gl_near_val)/(intr_gl_far_val-intr_gl_near_val), -1,
            0, 0, -2*intr_gl_far_val*intr_gl_near_val/(intr_gl_far_val-intr_gl_near_val), 0]
        
        
        
        cv2_extr_matrix = compute_opencv_extrinsic_matrix_from_gl_camera_pose(
            bc  = bc,
            pos = extr_gl_pos,
            orn = extr_gl_orn)
        intr_matrix = np.array([
            [intr_fx, 0,  intr_cx],
            [0,  intr_fy, intr_cy],
            [0,  0,  1]], dtype=np.float32)
    
        # Set attributes
        extrinsic = CameraExtrinsic(
            camera_pos      = extr_gl_pos,
            camera_quat     = extr_gl_orn,
            gl_view_matrix  = gl_view_matrix,
            cv2_extr_matrix = cv2_extr_matrix)
        intrinsic = CameraIntrinsic(
            fov               = intr_fov,
            width             = intr_width,
            height            = intr_height,
            aspect            = intr_width/intr_height,
            focusing_distance = intr_focusing_distance,
            intr_matrix       = intr_matrix,
            gl_near_val       = intr_gl_near_val,
            gl_far_val        = intr_gl_far_val,
            gl_proj_matrix    = gl_proj_matrix)

        return extrinsic, intrinsic


    def show_camera_pose(self, cv2_pose: bool = False):
        """Show camera pose for debug purpose.

        Args:
            cv2_pose (bool): Show cv2 frame instead of GL frame when True.
        """
        if cv2_pose:
            cam_pose = np.linalg.inv(self.extrinsic.cv2_extr_matrix)
        else:
            cam_pose = np.linalg.inv(np.reshape(self.extrinsic.gl_view_matrix, (4,4)).T)
        pos = cam_pose[:3,3]
        orn = R.from_matrix(cam_pose[:3,:3]).as_quat()
        self.pose_frame_visualizer.draw(pos, orn)


    def capture_rgbd_image(self) -> Tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
        """Capture RGB-D image and groundtruth segmentation mask from pybullet

        Returns:
            rgb_array (NDArray): Pixel RGB value [H, W, 3] ranges within [0, 255].
            depth_array (NDArray): Pixel depth value [H, W]
            mask_array (NDArray): Pixel uid value [H, W].
        """
        (w, h, px, px_d, px_id) = self.bc.getCameraImage(
            width            = self.intrinsic.width,
            height           = self.intrinsic.height,
            viewMatrix       = self.extrinsic.gl_view_matrix,
            projectionMatrix = self.intrinsic.gl_proj_matrix,
            renderer         = self.bc.ER_BULLET_HARDWARE_OPENGL)
        
        rgb_array   = np.array(px, dtype=np.uint8)
        rgb_array   = rgb_array[:, :, :3]                 # Remove alpha
        depth_array = np.array(px_d, dtype=float)
        seg_array   = np.array(px_id, dtype=np.int_)

        # For the calculation of the reprojection, see the material below.
        # https://stackoverflow.com/questions/6652253/getting-the-true-z-value-from-the-depth-buffer
        # Calculate z
        far = self.intrinsic.gl_far_val
        near = self.intrinsic.gl_near_val
        z_n = 2.0 * depth_array - 1.0
        depth_array = 2.0 * near * far / (far + near - z_n * (far - near))
                
        return rgb_array, depth_array, seg_array


class SimpleCamera(BulletCamera):

    def __init__(self, i: int, 
                       bc: BulletClient, 
                       config: Dict):
        """Configure simple camera...

        Args:
            i (int): Camera index in configurations.
            bc (BulletClient): Pybullet client
            config (Dict): Global configuration.
        """
        camera_config = config["camera_params"]["cameras"][i]
        self.type = camera_config["type"]
        if self.type != "simple":
            raise TypeError("Camera type and configuration does not match.")
        
        extrinsic, intrinsic = super().convert_bullet_properties(
            bc = bc, 
            extr_gl_pos            = camera_config["extrinsic"]["gl_pos"],
            extr_gl_orn            = bc.getQuaternionFromEuler(camera_config["extrinsic"]["gl_orn"]),
            intr_fov               = camera_config["intrinsic"]["fov"],
            intr_width             = camera_config["intrinsic"]["width"],
            intr_height            = camera_config["intrinsic"]["height"],
            intr_focusing_distance = camera_config["intrinsic"]["focusing_distance"],
            intr_gl_near_val       = camera_config["intrinsic"]["gl_near_val"],
            intr_gl_far_val        = camera_config["intrinsic"]["gl_far_val"])

        super().__init__(bc, extrinsic, intrinsic)

        
class RobotAttachedCamera(BulletCamera):

    def __init__(self, i: int,
                       bc: BulletClient,
                       robot: BulletRobot,
                       config: Dict):
        """_summary_

        Args:
            i (int): Camera index in configurations.
            bc (BulletClient): _description_
            robot (BulletRobot): _description_
            config (Dict): _description_
        """
        camera_config = config["camera_params"]["cameras"][i]
        self.type = camera_config["type"]
        if self.type != "robot_attached":
            raise TypeError("Camera type and configuration does not match.")
        
        # Get initial camera pose
        self.bc = bc
        self.robot = robot
        self.attach_gl_pos = camera_config["attachment"]["attach_gl_pos"]
        self.attach_gl_orn = bc.getQuaternionFromEuler(camera_config["attachment"]["attach_gl_orn"])
        initial_extr_gl_pos, initial_extr_gl_orn = self.__compute_camera_pose()
        extrinsic, intrinsic = super().convert_bullet_properties(
            bc = bc, 
            extr_gl_pos            = initial_extr_gl_pos,
            extr_gl_orn            = initial_extr_gl_orn,
            intr_fov               = camera_config["intrinsic"]["fov"],
            intr_width             = camera_config["intrinsic"]["width"],
            intr_height            = camera_config["intrinsic"]["height"],
            intr_focusing_distance = camera_config["intrinsic"]["focusing_distance"],
            intr_gl_near_val       = camera_config["intrinsic"]["gl_near_val"],
            intr_gl_far_val        = camera_config["intrinsic"]["gl_far_val"])

        super().__init__(bc, extrinsic, intrinsic)
        

    @staticmethod
    def compute_ee_pos_from_camera_pose(
            bc,
            cam_pos: TranslationT, 
            cam_orn: QuaternionT,
            attach_gl_pos: TranslationT,
            attach_gl_orn: QuaternionT,
            convention: str = "gl"
    ) -> Tuple[TranslationT, QuaternionT]:
        """Compute ee pose that is required to get the desired camera pose."""
        rel_ee_pos, rel_ee_orn = bc.invertTransform(attach_gl_pos, attach_gl_orn)
        ee_pos, ee_orn = bc.multiplyTransforms(cam_pos, cam_orn, rel_ee_pos, rel_ee_orn)

        return ee_pos, ee_orn


    def __compute_camera_pose(self) -> Tuple[TranslationT, QuaternionT]:
        """_summary_

        Returns:
            Tuple[TranslationT, QuaternionT]: _
        """
        ee_pos, ee_orn = self.robot.get_endeffector_pose()
        cam_gl_pos, cam_gl_orn = self.bc.multiplyTransforms(ee_pos, ee_orn, self.attach_gl_pos, self.attach_gl_orn)
        # self.debug_cam_frame_uids \
        #     = draw_coordinate(self.bc, cam_pos, cam_orn, brightness=1.0, line_uid_xyz=self.debug_cam_frame_uids)
        
        return cam_gl_pos, cam_gl_orn


    def update_camera_extrinsic(self):
        """_summary_"""
        extr_gl_pos, extr_gl_orn = self.__compute_camera_pose()

        # OpenGL
        gl_view_matrix = compute_gl_view_matrix_from_gl_camera_pose(
            bc  = self.bc, 
            pos = extr_gl_pos,
            orn = extr_gl_orn)
        # OpenCV
        cv2_extr_matrix = compute_opencv_extrinsic_matrix_from_gl_camera_pose(
            bc  = self.bc,
            pos = extr_gl_pos,
            orn = extr_gl_orn)
        # Set attributes
        self.extrinsic  = CameraExtrinsic(
            camera_pos      = extr_gl_pos,
            camera_quat     = extr_gl_orn,
            gl_view_matrix  = gl_view_matrix,
            cv2_extr_matrix = cv2_extr_matrix)

    
    def capture_rgbd_image(self) -> Tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
        """_summary_"""
        self.update_camera_extrinsic()
        return super().capture_rgbd_image()


class DebugCamera(BulletCamera):

    def __init__(self, bc, extrinsic, intrinsic):
        super().__init__(bc, extrinsic, intrinsic)

    @classmethod
    def from_debugger(cls, bc):
        extrinsic, intrinsic = cls.get_properties_from_debugger(bc)
        return cls(bc, extrinsic, intrinsic)

    def update_parameters(self, bc):
        extrinsic, intrinsic = self.get_properties_from_debugger(bc)
        self.extrinsic = extrinsic
        self.intrinsic = intrinsic

    @staticmethod
    def get_properties_from_debugger(bc):
        """Compute properties"""
        intr_width, intr_height, gl_view_matrix, gl_proj_matrix, *args = bc.getDebugVisualizerCamera()
        gl_view_matrix = np.array(gl_view_matrix)
        gl_proj_matrix = np.array(gl_proj_matrix)

        extr_gl_pos, extr_gl_orn = compute_gl_camera_pose_from_gl_view_matrix(bc, gl_view_matrix)
        cv2_extr_matrix = compute_opencv_extrinsic_matrix_from_gl_camera_pose(bc, extr_gl_pos, extr_gl_orn)
        intr_fov, aspect, intr_gl_near_val, intr_gl_far_val = compute_parameters_from_projection_matrix(gl_proj_matrix)
        intr_matrix = compute_intrinsic_matrix_from_vertical_fov(intr_fov, intr_width, intr_height)

        # Undefined.
        intr_focusing_distance = 1.

        # Set attributes
        extrinsic = CameraExtrinsic(
            camera_pos      = extr_gl_pos,
            camera_quat     = extr_gl_orn,
            gl_view_matrix  = gl_view_matrix,
            cv2_extr_matrix = cv2_extr_matrix)
        intrinsic = CameraIntrinsic(
            fov               = intr_fov,
            width             = intr_width,
            height            = intr_height,
            aspect            = intr_width/intr_height,
            focusing_distance = intr_focusing_distance,
            intr_matrix       = intr_matrix,
            gl_near_val       = intr_gl_near_val,
            gl_far_val        = intr_gl_far_val,
            gl_proj_matrix    = gl_proj_matrix)

        return extrinsic, intrinsic
