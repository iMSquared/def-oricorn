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
from imm.pybullet_util.common import ( get_visual_poses_and_scales, get_transform_matrix, get_transform_matrix_with_scale, get_link_pose )
from imm.pybullet_util.vision import draw_coordinate


PROJECT_ROOT_PATH = str(pathlib.Path(__file__).parent.parent.parent)
sys.path.insert(0, PROJECT_ROOT_PATH)

from util.grasp_util import best_grasp_pose

sys.path.remove(PROJECT_ROOT_PATH)



def grasp_pose_sampler(bc: BulletClient, 
                       obj_uid: int) \
                        -> Tuple[List[Tuple[TranslationT, QuaternionT]], 
                                 List[Tuple[TranslationT, QuaternionT]], 
                                 List[float]]:
    """Sample grasp poses in robot frame.

    Args:
        bc (BulletClient): _
        obj_uid (int): Uid of object to pick

    Returns:
        grasp_world_list (List[Tuple[TranslationT, QuaternionT]): _
        grasp_link_list (List[Tuple[TranslationT, QuaternionT]): _
        grasp_score_list (List[float]): _
    """
    # Get visual mesh
    vis_data = bc.getVisualShapeData(obj_uid)[0]    # FIXME
    mesh_asset_file_name = vis_data[4]
    mesh = o3d.io.read_triangle_mesh(mesh_asset_file_name)


    # Get visual pose in the link_frame
    vis_scale_in_link_frame = vis_data[3]
    vis_pos_in_link_frame = vis_data[5]
    vis_orn_in_link_frame = vis_data[6]
    link_pos, link_orn = get_link_pose(bc, obj_uid, -1, inertial=False) # Link frame.
    vis_pos, vis_orn = bc.multiplyTransforms(link_pos, link_orn,
                                                vis_pos_in_link_frame, vis_orn_in_link_frame)
    vis_scale = vis_scale_in_link_frame
    T_vis_in_link = get_transform_matrix_with_scale(vis_pos, vis_orn, vis_scale)
    mesh.transform(T_vis_in_link)

    # Get link pose in the world frame
    T_link_in_world = get_transform_matrix(link_pos, link_orn)

    # Grasp sampler in link frame
    jkey = jax.random.PRNGKey(0)
    Toe_grasp_sorted_link, score_sorted = best_grasp_pose(jkey, mesh, "mesh", vis=False)
    Toe_grasp_sorted_link = np.array(Toe_grasp_sorted_link)
    score_sorted     = np.array(score_sorted)

    # Matmul along axis=1
    Toe_grasp_sorted_world = np.einsum("ij,njk->nik", T_link_in_world, Toe_grasp_sorted_link)
        

    # To pos and quat
    array_pos_world = Toe_grasp_sorted_world[:,:3,3]
    array_orn_world = Toe_grasp_sorted_world[:,:3,:3]
    array_pos_link = Toe_grasp_sorted_link[:,:3,3]
    array_orn_link = Toe_grasp_sorted_link[:,:3,:3]
    grasp_world_list = []
    grasp_link_list = []
    grasp_score_list = []
    for pos_world, orn_world, pos_link, orn_link, score in \
            zip(array_pos_world, array_orn_world, array_pos_link, array_orn_link, score_sorted):
        _pos_link = pos_link.tolist()
        _orn_link = sciR.from_matrix(orn_link).as_quat()
        _pos_world = pos_world.tolist()
        _orn_world = sciR.from_matrix(orn_world).as_quat()
        grasp_world_list.append((_pos_world, _orn_world))
        grasp_link_list.append((_pos_link, _orn_link))
        grasp_score_list.append(score)

    return grasp_world_list, grasp_link_list, grasp_score_list
