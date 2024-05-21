from __future__ import annotations
import math
from typing import Optional, Tuple, List
import open3d as o3d
import pybullet as pb
import numpy as np
import numpy.typing as npt
from pybullet_utils.bullet_client import BulletClient
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as ScipyR

from imm.pybullet_util.common import get_link_pose, get_transform_matrix
from imm.pybullet_util.typing_extra import TranslationT, QuaternionT


def unwrap_o3d_pcd(pcd: object) -> npt.NDArray:
    return np.asarray(pcd.points)


def wrap_o3d_pcd(pcd: npt.NDArray) -> object:
    pcd_o3d = o3d.geometry.PointCloud()
    pcd_o3d.points = o3d.utility.Vector3dVector(pcd)
    return pcd_o3d


def transform_points(points: npt.NDArray, pos: TranslationT, orn: QuaternionT) -> npt.NDArray:
    """Transform R^3 points
    
    Args:
        points (npt.NDArray): Point cloud
        pos (TranslationT): Position
        orn (QuaternionT): Rotation
    
    Returns:
        npt.NDArray: Transformed points
    """
    points_h = np.concatenate((points, np.ones((points.shape[0],1))), axis=1)
    T = get_transform_matrix(pos, orn)
    points_h = np.einsum("ij,nj->ni",  T, points_h)
    points = points_h[:,:3]

    return points


class PybulletPCDManager:
    """Too lazy to explain"""
    def __init__(self, bc: BulletClient):
        self.reinit(bc)

    def add(self, pcd: np.ndarray, color=np.array([1,0,0])):
        color = np.asarray(color)
        color = np.repeat(color[None,...], len(pcd), axis=0)
        uid = self.debug_pcd_slots[next(self.get_pcd_idx)]
        self.bc.addUserDebugPoints(pcd, color, pointSize=1, replaceItemUniqueId=uid)

    def remove_all(self):
        for i in range(20):
            uid = self.debug_pcd_slots[next(self.get_pcd_idx)]
            self.bc.addUserDebugPoints([[0,0,0]], [[0,0,0]], pointSize=0.1, replaceItemUniqueId=uid)

    @staticmethod
    def pcd_idx_generator():
        i = 0
        while True:
            yield i
            i += 1
            if i >= 20:
                i = 0
    
    def reinit(self, bc):
        # Swap bc
        self.bc = bc
        # Empty slots
        self.debug_pcd_slots = []
        for i in range(20):
            uid = -1
            while uid == -1:
                uid = self.bc.addUserDebugPoints([[0,0,0]], [[0,0,0]], pointSize=1)
            self.debug_pcd_slots.append(uid)
        # Slot generator
        self.get_pcd_idx = PybulletPCDManager.pcd_idx_generator()



def draw_coordinate(bc: BulletClient, 
                    target_pos: Optional[TranslationT]=None, 
                    target_orn_q: Optional[QuaternionT]=None, 
                    parent_object_unique_id: Optional[int] = None, 
                    parent_link_index: Optional[int] = None, 
                    line_uid_xyz: Optional[Tuple[int, int, int]] = None,
                    brightness: float = 1.0,
                    line_width = 2) -> Tuple[int, int, int]:
    """Draw coordinate frame

    Args:
        bc (BulletClient): PyBullet client
        target_pos (Optional[TranslationT], optional): Position of local frame in global frame
        target_orn_q (Optional[QuaternionT], optional): Orientation of local frame in global frame
        parent_object_unique_id (Optional[int], optional): Inertial frame. Defaults to None.
        parent_link_index (Optional[int], optional): Inertial frame. Defaults to None.
        line_uid (Tuple[int, int, int], optional): Replace uid. Defaults to None.
        brightness (float): Color brightness
        line_width (float): _

    Returns:
        line_uid_xyz (Tuple[int, int, int]): Line uid
    """

    
    origin_pos = [0.0, 0.0, 0.0]
    x_pos = [0.1, 0.0, 0.0]
    y_pos = [0.0, 0.1, 0.0]
    z_pos = [0.0, 0.0, 0.1]
    origin_orn_q = [0.0, 0.0, 0.0, 1.0]


    if parent_object_unique_id is not None:
        if line_uid_xyz is not None:
            line_uid_x = bc.addUserDebugLine(origin_pos, x_pos, [1*brightness, 0, 0], 
                                            lineWidth = line_width, 
                                            parentObjectUniqueId = parent_object_unique_id,
                                            parentLinkIndex = parent_link_index,
                                            replaceItemUniqueId = line_uid_xyz[0])
            line_uid_y = bc.addUserDebugLine(origin_pos, y_pos, [0, 1*brightness, 0], 
                                            lineWidth = line_width, 
                                            parentObjectUniqueId = parent_object_unique_id,
                                            parentLinkIndex = parent_link_index,
                                            replaceItemUniqueId = line_uid_xyz[1])
            line_uid_z = bc.addUserDebugLine(origin_pos, z_pos, [0, 0, 1*brightness], 
                                            lineWidth = line_width, 
                                            parentObjectUniqueId = parent_object_unique_id,
                                            parentLinkIndex = parent_link_index,
                                            replaceItemUniqueId = line_uid_xyz[2])
        else:
            line_uid_x = bc.addUserDebugLine(origin_pos, x_pos, [1*brightness, 0, 0], 
                                            lineWidth = line_width, 
                                            parentObjectUniqueId = parent_object_unique_id,
                                            parentLinkIndex = parent_link_index)
            line_uid_y = bc.addUserDebugLine(origin_pos, y_pos, [0, 1*brightness, 0], 
                                            lineWidth = line_width, 
                                            parentObjectUniqueId = parent_object_unique_id,
                                            parentLinkIndex = parent_link_index)
            line_uid_z = bc.addUserDebugLine(origin_pos, z_pos, [0, 0, 1*brightness], 
                                            lineWidth = line_width, 
                                            parentObjectUniqueId = parent_object_unique_id,
                                            parentLinkIndex = parent_link_index)
    else:
        target_origin_pos, target_origin_orn_q = bc.multiplyTransforms(target_pos, target_orn_q,
                                                                       origin_pos, origin_orn_q) 
        target_x_pos, target_x_orn_q = bc.multiplyTransforms(target_pos, target_orn_q,
                                                             x_pos, origin_orn_q)
        target_y_pos, target_y_orn_q = bc.multiplyTransforms(target_pos, target_orn_q,
                                                             y_pos, origin_orn_q)
        target_z_pos, target_z_orn_q = bc.multiplyTransforms(target_pos, target_orn_q,
                                                             z_pos, origin_orn_q)
        
        if line_uid_xyz is not None:
            line_uid_x = bc.addUserDebugLine(target_origin_pos, target_x_pos, [1*brightness, 0, 0], 
                                            lineWidth = line_width,
                                            replaceItemUniqueId = line_uid_xyz[0])
            line_uid_y = bc.addUserDebugLine(target_origin_pos, target_y_pos, [0, 1*brightness, 0], 
                                            lineWidth = line_width,
                                            replaceItemUniqueId = line_uid_xyz[1])
            line_uid_z = bc.addUserDebugLine(target_origin_pos, target_z_pos, [0, 0, 1*brightness], 
                                            lineWidth = line_width,
                                            replaceItemUniqueId = line_uid_xyz[2])
        else:
            line_uid_x = bc.addUserDebugLine(target_origin_pos, target_x_pos, [1*brightness, 0, 0], 
                                            lineWidth = line_width)
            line_uid_y = bc.addUserDebugLine(target_origin_pos, target_y_pos, [0, 1*brightness, 0], 
                                            lineWidth = line_width)
            line_uid_z = bc.addUserDebugLine(target_origin_pos, target_z_pos, [0, 0, 1*brightness], 
                                            lineWidth = line_width)

    return (line_uid_x, line_uid_y, line_uid_z)



def draw_bounding_box(
        bc: BulletClient,
        bb_corners_canoncial: List[TranslationT],
        bb_pos: TranslationT,
        bb_orn: QuaternionT, 
        color: Tuple[float, float, float] = (0.0, 0.8, 0.8),
        line_width: float = 2.,
        bb_line_uids: Tuple[int|None, ...] = [None for i in range(12)]
) -> Tuple[int, ...]:
    """ Draw bounding box
    
    Args:
        bc (BulletClient): ...
        bb_corners_canoncial (List[TranslationT]): Bounding box corners in canoncial pose
        bb_pos (TranslationT): Position of bb in world.
        bb_orn (QuaternionT): Orientation of bb in world
        color (Tuple[float, float, float]): Defaults to (0.0, 0.8, 0.8)
        line_width (float) bb line width. Defaults to 2.
        bb_line_uids (Tuple[int, ...]): Line uids for debugger update.

    Returns: 
        bb_line_uids (Tuple[int, ...]): Line uids for debugger update.
    """
    bb_corners_h = np.concatenate((bb_corners_canoncial, np.ones((8,1))), axis=1)
    T_link_in_world = get_transform_matrix(bb_pos, bb_orn)
    bb_corners_h = np.einsum("ij,nj->ni",  T_link_in_world, bb_corners_h)
    bb_corners = bb_corners_h[:,:3]

    # Compose Line from corners
    # 1st layer: 0 1 7 2 (CCW)
    # 2nd layer: 3 6 4 5
    lines = [
        [0, 1], [1, 7], [7, 2], [2, 0], 
        [0, 3], [1, 6], [7, 4], [2, 5], 
        [3, 5], [5, 4], [4, 6], [6, 3]]
    for i, line in enumerate(lines):
        idx_a, idx_b = line
        corner_a = bb_corners[idx_a]
        corner_b = bb_corners[idx_b]
        # Init or update
        if bb_line_uids[i] == None:
            bb_line_uids.append(
                    bc.addUserDebugLine(
                        corner_a, corner_b, color, 
                        lineWidth = line_width))
        else:
            bb_line_uids[i] \
                = bc.addUserDebugLine(
                    corner_a, corner_b, color, 
                    lineWidth = line_width,
                    replaceItemUniqueId = bb_line_uids[i])

    return bb_line_uids



class CoordinateVisualizer:

    def __init__(self, bc: BulletClient, brightness: float = 1.0):
        self.bc = bc
        self.brightness = brightness
        self.line_uids: Tuple[int, int, int] = None
    
    def draw(self, target_pos: Optional[TranslationT], 
                   target_orn_q: Optional[QuaternionT]):
        self.line_uids = draw_coordinate(self.bc, target_pos, target_orn_q, 
                                         line_uid_xyz = self.line_uids,
                                         brightness = self.brightness)

    def hide(self):
        for l_uid in self.line_uids:
            self.bc.removeUserDebugItem(l_uid)



class LinkBoundingBoxVisualizer:
    
    def __init__(self, bc: BulletClient,
                       uid: int,
                       link_index: int,
                       bb_center_pos_in_link_frame: TranslationT,
                       bb_center_orn_in_link_frame: QuaternionT,
                       bb_size: TranslationT,
                       bb_corners_in_link_frame: List[TranslationT],
                       color = [0.0, 0.8, 0.8],
                       line_width = 2):
        """This class only visualizes the predefined bounding box of a link."""
        self.bc = bc
        self.uid = uid
        self.link_index = link_index
        self.bb_center_pos_in_link_frame = bb_center_pos_in_link_frame
        self.bb_center_orn_in_link_frame = bb_center_orn_in_link_frame
        self.bb_size = bb_size
        self.bb_corners_in_link_frame = bb_corners_in_link_frame
        self.bb_line_uids = [None for i in range(12)]
        self.color = color
        self.line_width = line_width
        self.coordinate = CoordinateVisualizer(self.bc)

    def draw(self):
        # TODO: replace the content with draw_bounding_box.
        # Transform bb corners to world frame
        bb_corners = np.concatenate((self.bb_corners_in_link_frame, np.ones((8,1))), axis=1)
        link_pos, link_orn = get_link_pose(self.bc, self.uid, self.link_index, inertial=False)
        T_link_in_world = get_transform_matrix(link_pos, link_orn)
        bb_corners = np.einsum("ij,nj->ni",  T_link_in_world, bb_corners)
        bb_corners = bb_corners[:,:3]
        # Compose Line from corners
        # 1st layer: 0 1 7 2 (CCW)
        # 2nd layer: 3 6 4 5
        lines = [[0, 1], [1, 7], [7, 2], [2, 0], 
                 [0, 3], [1, 6], [7, 4], [2, 5], 
                 [3, 5], [5, 4], [4, 6], [6, 3]]
        for i, line in enumerate(lines):
            idx_a, idx_b = line
            corner_a = bb_corners[idx_a]
            corner_b = bb_corners[idx_b]
            # Init or update
            if self.bb_line_uids[i] == None:
                self.bb_line_uids.append(
                    self.bc.addUserDebugLine(
                        corner_a, corner_b, self.color, 
                        lineWidth = self.line_width))
            else:
                self.bb_line_uids[i] \
                    = self.bc.addUserDebugLine(
                        corner_a, corner_b, self.color, 
                        lineWidth = self.line_width,
                        replaceItemUniqueId = self.bb_line_uids[i])

    def draw_coordinate(self):
        self.coordinate.draw(self.bb_center_pos_in_world_frame, self.bb_center_orn_in_world_frame)    

    def hide(self):
        for l_uid in self.bb_line_uids:
            self.bc.removeUserDebugItem(l_uid)
    
    @property
    def bb_center_pos_in_world_frame(self) -> TranslationT:
        link_pos, link_orn = get_link_pose(self.bc, self.uid, self.link_index, inertial=False)
        pos, orn = self.bc.multiplyTransforms(link_pos, link_orn, 
                                              self.bb_center_pos_in_link_frame, 
                                              self.bb_center_orn_in_link_frame)
        return pos

    @property
    def bb_center_orn_in_world_frame(self) -> QuaternionT:
        link_pos, link_orn = get_link_pose(self.bc, self.uid, self.link_index, inertial=False)
        pos, orn = self.bc.multiplyTransforms(link_pos, link_orn, 
                                              self.bb_center_pos_in_link_frame, 
                                              self.bb_center_orn_in_link_frame)
        return orn



def compute_gl_view_matrix_from_gl_camera_pose(bc: BulletClient, pos: TranslationT, orn: QuaternionT):
    """
    Compute PyBullet view matrix from the direct camera pose, 
    with zero aligned with OpenGL style origin.
    
    Args: 
        bc (BulletClient): -
        pos (TranslationT): -
        orn (QuaternionT): -

    Returns:
        transformed_view_matrix (npt.NDArray): -
    """
    # Default pose of camera is identity. 
    # Its inverse is view matrix and looks at -z with +y up.
    default_pose_matrix = np.eye(4)

    # Compose transformation of camera pose
    R = np.asarray(bc.getMatrixFromQuaternion(orn)).reshape((3,3))
    T = np.array([[R[0,0], R[0,1], R[0,2], pos[0]],
                  [R[1,0], R[1,1], R[1,2], pos[1]],
                  [R[2,0], R[2,1], R[2,2], pos[2]],
                  [     0,      0,      0,      1]])

    # Matrix chain
    transformed_camera_pose = T@default_pose_matrix
    transformed_view_matrix = np.linalg.inv(transformed_camera_pose)

    # Make sure to transpose following the GL convention
    return transformed_view_matrix.T.flatten()  


def compute_gl_camera_pose_from_gl_view_matrix(bc: BulletClient, gl_view_matrix: np.ndarray[float]) -> Tuple[TranslationT, QuaternionT]:
    """ Compute open gl camera pose from view matrix.

    Args:
        bc (BulletClient): -
        view_matrix (np.ndarray[float]): Transposed and flattened matrix following gl convention.

    Returns:
        Tuple[TranslationT, QuaternionT]: -
    """
    gl_view_matrix = gl_view_matrix.reshape(4,4).T
    gl_camera_pose = np.linalg.inv(gl_view_matrix)
    
    gl_pos = gl_camera_pose[:3, 3]
    gl_orn = ScipyR.from_matrix(gl_camera_pose[:3,:3]).as_quat()

    return gl_pos, gl_orn


def compute_opencv_extrinsic_matrix_from_gl_camera_pose(bc: BulletClient, pos: TranslationT, orn: QuaternionT):
    """
    Compute extrinsic matrix from the direct camera pose.
    with zero aligned with opencv style (y-down)

    Args:
        bc (BulletClient): _
        pos (TranslationT): _
        orn (QuaternionT): _
    
    Returns:
        transformed_extr_matrix (npt.NDArray): -
    """

    # Default value of extr matrix that looks at -z with +y up.
    default_pose_matrix = np.array([[1,  0,  0,  0],
                                    [0, -1,  0,  0],
                                    [0,  0, -1,  0],
                                    [0,  0,  0,  1]])


    # Compose transformation of camera pose
    R = np.asarray(bc.getMatrixFromQuaternion(orn)).reshape((3,3))
    T = np.array([[R[0,0], R[0,1], R[0,2], pos[0]],
                  [R[1,0], R[1,1], R[1,2], pos[1]],
                  [R[2,0], R[2,1], R[2,2], pos[2]],
                  [     0,      0,      0,      1]])

    # Matrix chain
    transformed_camera_pose = T@default_pose_matrix
    transformed_extr_matrix = np.linalg.inv(transformed_camera_pose)

    return transformed_extr_matrix


def compute_intrinsic_matrix_from_vertical_fov(vfov_degree: float, w: int, h: int) -> npt.NDArray:
    """Derive instrinsic matrix from fov_y

    Only uses the minimum parameters for pybullet.
    hfov, fx, fy, cx, fy are calculated in this function.

    Args:
        vfov_degree (float): Degree
        w (int): _description_
        h (int): _description_
    
    Returns:
        intrinsic_matrix (npt.NDArray): 3x3
    """
    aspect = float(w)/float(h)
    vfov_rad = vfov_degree * math.pi / 180.
    hfov_rad = 2 * math.atan( math.tan(vfov_rad/2.) * aspect)

    fx = (w/2.) / math.tan(hfov_rad/2.)
    fy = (h/2.) / math.tan(vfov_rad/2.)
    cx = w/2.
    cy = h/2.

    intrinsic_matrix = np.array([
        [fx, 0,  cx],
        [0,  fy, cy],
        [0,  0,  1]], dtype=np.float32)
    
    return intrinsic_matrix


def compute_vfov_from_intrinsics(width, height, fx, fy, cx, cy):
    vfov_radians = 2 * math.atan(height / (2 * fy))
    vfov_degree = math.degrees(vfov_radians)
    return vfov_degree


def compute_parameters_from_projection_matrix(proj_matrix: np.ndarray):
    """ Extracts FOV, aspect ratio, near plane, and far plane values from an OpenGL projection matrix.

    Args:
    - proj_matrix: A numpy array representing the 4x4 projection matrix.

    Returns:
    - fov_y_degrees: The field of view in the y-direction in degrees.
    - aspect_ratio: The aspect ratio of the projection (width / height).
    - near_val: The distance to the near clipping plane.
    - far_val: The distance to the far clipping plane.
    """
    # Extracting values from the projection matrix
    P = proj_matrix.reshape(4,4).T
    near_val = (2 * P[3, 2]) / (P[2, 2] + 1)
    far_val = P[3, 2] / (P[2, 2] - 1)
    aspect_ratio = P[1, 1] / P[0, 0]
    fov_y = 2 * np.arctan(1 / P[1, 1])
    
    # Convert FOV from radians to degrees
    fov_y_degrees = np.degrees(fov_y)
    
    return fov_y_degrees, aspect_ratio, near_val, far_val



def visualize_point_cloud(pcds:list, 
                          lower_lim=-0.25, 
                          upper_lim=0.25, 
                          save:bool=False, 
                          save_path:Optional[str]=None):
    '''
    Visualize the numpy point cloud
    '''
    if save:
        plt.switch_backend('Agg') # tkinter keeps crashing... :(
    else:
        plt.switch_backend('TkAgg')

    colors = ["Red", "Blue", "Green", "tab:orange", "magenta", "tab:blue", "tab:purple", "tab:olive"]

    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(projection='3d')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_xlim([lower_lim, upper_lim])
    ax.set_ylim([lower_lim, upper_lim])
    ax.set_zlim([lower_lim, upper_lim])

    # Plot points
    for i, pcd in enumerate(pcds):
        ax.scatter(pcd[:,0], pcd[:,1], pcd[:,2], s=0.2, c=colors[i % len(colors)])

    if not save:
        plt.show()
        # print(4)
    else:
        fig.savefig(save_path)
        plt.close(fig)



def reproject_depth_to_pcd(intrinsic_formatted: npt.NDArray, depth_array: np.ndarray, downsample_voxel: float = 0., convention="cv2") -> np.ndarray:
    """Convert depth image to point clouds.

    Args:
        depth_array (np.ndarray): Depth image
    Returns:
        pointcloud (np.ndarray): Reprojected target point cloud
    """
    # Convert depth image to pcd in pixel unit
    # (x, y, z), y-up, x-right, z-front
    focal_length_x = intrinsic_formatted[2]
    focal_length_y = intrinsic_formatted[3]
    cx = intrinsic_formatted[4]
    cy = intrinsic_formatted[5]
    pcd = np.array([[
                (cy - v, cx - u, depth_array[v, u])
            for u in range(depth_array.shape[1])]
        for v in range(depth_array.shape[0])]).reshape(-1, 3)

    # Getting true depth from OpenGL style perspective matrix
    #   For the calculation of the reprojection, see the material below
    #   https://stackoverflow.com/questions/6652253/getting-the-true-z-value-from-the-depth-buffer
    
    
    # Calculate z
    z = pcd[:,2]
    # Calculate x
    x = z * pcd[:,1] / focal_length_x
    # Calculate y
    y = z * pcd[:,0] / focal_length_y

    # Stack
    pcd = np.stack((x, y, z), axis=1)
    if convention == "gl":
        pcd = pcd * np.array([1,-1,-1])
    elif convention == "cv2":
        pcd = pcd * np.array([-1,-1,1])
    else:
        raise ValueError(f"{convention} not known")

    if downsample_voxel != 0:
        import open3d as o3d
        o3d_pcd = o3d.geometry.PointCloud()
        o3d_pcd.points = o3d.utility.Vector3dVector(pcd)
        o3d_downpcd = o3d_pcd.voxel_down_sample(voxel_size=0.05)
        pcd = np.array(o3d_downpcd.points)

    return pcd

