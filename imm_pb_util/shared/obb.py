# Some dirty tricks to import packages from project root.
import pathlib
import sys
from typing import List, Tuple
import open3d as o3d
import jax
import jax.numpy as jnp
import jax.typing as jnt
import numpy as np
from scipy.spatial.transform import Rotation as sciR


from pybullet_utils.bullet_client import BulletClient
from imm.pybullet_util.typing_extra import TranslationT, QuaternionT, EulerT
from imm.pybullet_util.common import ( get_link_pose, get_visual_poses_and_scales, get_transform_matrix_with_scale )
from imm.pybullet_util.vision import draw_coordinate


PROJECT_ROOT_PATH = str(pathlib.Path(__file__).parent.parent.parent)
sys.path.insert(0, PROJECT_ROOT_PATH)

from util.obb_util import get_mesh_obb

sys.path.remove(PROJECT_ROOT_PATH)


def get_link_obb(bc: BulletClient, 
                     obj_uid: int,
                     obj_link_idx: int = -1) \
                        -> Tuple[TranslationT, QuaternionT, TranslationT, List[TranslationT]]:
    """_summary_

    Longest edge becomes z axis

    Args:
        bc (BulletClient): _
        obj_uid (int): _
        obj_link_index (int): _
    
    Returns:
        bb_center_pos_in_link_frame (): _
        bb_center_orn_in_link_frame (): _
        bb_size (): _
        corners (): _
    """

    # Get visual mesh
    vis_data_list = bc.getVisualShapeData(obj_uid)
    vis_data = [v for v in vis_data_list if v[1]==obj_link_idx][0]
    mesh_asset_file_name    = vis_data[4]
    vis_pos_in_link_frame   = vis_data[5]
    vis_orn_in_link_frame   = vis_data[6]
    vis_scale_in_link_frame = vis_data[3]
    mesh = o3d.io.read_triangle_mesh(mesh_asset_file_name)
    # Transform to link frame.
    T_vis_in_link = get_transform_matrix_with_scale(
        vis_pos_in_link_frame,
        vis_orn_in_link_frame,
        vis_scale_in_link_frame)
    mesh.transform(T_vis_in_link)

    # Get OBB pose in link frame
    center_pos, center_orn, bb_size, corners = get_mesh_obb(mesh, vis=False)
    center_pos = np.array(center_pos)
    center_orn = np.array(center_orn)
    bb_size = np.array(bb_size)
    bb_corners_in_link_frame = np.array(corners)

    # Center pose in link frame
    T_obb_in_link = np.eye(4)
    T_obb_in_link[:3,3]  = center_pos
    T_obb_in_link[:3,:3] = center_orn
    bb_center_pos_in_link_frame = T_obb_in_link[:3,3]
    bb_center_orn_in_link_frame = sciR.from_matrix(T_obb_in_link[:3,:3]).as_quat()

    return bb_center_pos_in_link_frame, bb_center_orn_in_link_frame, bb_size, bb_corners_in_link_frame