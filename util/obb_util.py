import numpy as np
from typing import Tuple
import jax
import jax.numpy as jnp
import open3d as o3d
import flax
from dataclasses import dataclass
from flax.struct import dataclass as flax_dataclass
from functools import partial

import util.cvx_util as cxutil
import util.transform_util as tutil



# @dataclass(frozen=True)
# class ExplicitOBB:
#     raise NotImplementedError("Not implemented")


@flax_dataclass
class LatentOBB:
    """Assumes the object is always laid down"""
    center: jnp.ndarray # shape = [3]
    dirs: jnp.ndarray   # shape = [3,3]
    dists: jnp.ndarray  # shape = [3]
    
    @property
    def upright_q(self) -> jnp.ndarray:
        rot = self.dirs
        q = tutil.Rm2q(rot)
        return q
    
    @property
    def upright_z_offset(self) -> jnp.ndarray:
        longest = self.dists[2] 
        shortest = self.dists[0]
        return (longest - shortest)/2


def get_obb_from_open3d_mesh(
        mesh: object, 
        num_points: int = 500, 
        vis: bool = False
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Get oriented bounding box from a open3d mesh
    
    Longest edge becomes z axis

    Args:
        mesh (object): o3d mesh
        num_points (int): number of sample points to create bb from.
        vis (bool): debug

    Returns:
        center_pos (jnt.ArrayLike): position of bb center in visual frame. shape (3,)
        center_orn (jnt.ArrayLike): rotation matrix of bb center in visual frame., shape (3, 3)
        bb_size (jnt.ArrayLike): x,y,z size, shape (3,)
        corners (jnt.ArrayLike): corners positions in visual frame, shape (8, 3)
    """

    # Open3D OBB
    pcd = mesh.sample_points_uniformly(num_points)
    oriented_3d_bbox = o3d.geometry.OrientedBoundingBox.create_from_points(pcd.points)
    center_pos = np.asarray(oriented_3d_bbox.get_center())  # (3,)
    corners = np.asarray(oriented_3d_bbox.get_box_points())  # (8, 3)
    vertices = np.concatenate([center_pos[None, :], corners])
    
    # three edges: 1->2 1->3 1->4
    edge_1 = vertices[2]-vertices[1]
    edge_2 = vertices[3]-vertices[1] 
    edge_3 = vertices[4]-vertices[1]

    # compute orthonormal unit vectors
    dist_1 = np.linalg.norm(edge_1)
    dist_2 = np.linalg.norm(edge_2)
    dist_3 = np.linalg.norm(edge_3)
    dir_1 = edge_1 / dist_1
    dir_2 = edge_2 / dist_2
    dir_3 = edge_3 / dist_3

    # ordered direction axis vectors are already a rotation matrix  
    dir_list = jnp.stack([dir_1, dir_2, dir_3])
    dist_list = jnp.stack([dist_1, dist_2, dist_3])
    center_orn = np.linalg.inv(dir_list)
    bb_size = dist_list
    # need to satisfy cross(x, y) = +z.
    # if not, simply flip z.
    if jnp.linalg.det(center_orn) < 0:
        center_orn = center_orn.at[:,2].set(-center_orn[:,2])

    # longest edge to z-up
    long_axis = jnp.argmax(dist_list)
    if long_axis == 0:
        swap = np.array([[0,0,1],
                         [1,0,0],
                         [0,1,0]])
        center_orn = center_orn @ swap
        bb_size = bb_size[np.array([1,2,0])]
    elif long_axis == 1:
        swap = np.array([[0,1,0],
                         [0,0,1],
                         [1,0,0]])
        center_orn = center_orn @ swap
        bb_size = bb_size[np.array([2,0,1])]

    # Debug
    if vis:
        # 1st layer: 1 2 8 3 (CCW)
        # 2nd layer: 4 7 5 6
        lines = [[1, 2], [2, 8], [8, 3], [3, 1], 
                 [1, 4], [2, 7], [8, 5], [3, 6], 
                 [4, 6], [6, 5], [5, 7], [7, 4]]
        colors = [[0, 0, 0] for i in range(len(lines))]

        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(vertices)
        line_set.lines = o3d.utility.Vector2iVector(lines)
        line_set.colors = o3d.utility.Vector3dVector(colors)

        corner_vis = o3d.geometry.PointCloud()
        corner_vis.points = o3d.utility.Vector3dVector(np.asarray([vertices[1], vertices[2], vertices[3], vertices[4]]))

        # Retrieve the normal vectors at the sampled points
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(pcd.points)
        pcd_color = np.array(np.zeros_like(np.asarray(pcd.points)))
        pcd_color[:, 0] = 1
        point_cloud.colors = o3d.utility.Vector3dVector(pcd_color)

        # Debug
        T = np.eye(4)
        T[:3,3] = center_pos
        T[:3,:3] = center_orn
        bbox_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05)
        bbox_frame.transform(T)

        o3d.visualization.draw_geometries([line_set, point_cloud, corner_vis, bbox_frame])


    return center_pos, center_orn, bb_size, corners


def get_obb_from_lying_latent_object(
        models: flax.linen.Module,
        obj: cxutil.LatentObjects,
) -> LatentOBB:
    """Get the OBB of the latent object.
    
    This function assumes the object is always laid down. This function is not jit compatible.
    
    Args:
        models (flax.linen.Module): Models
        obj (cxutil.LatentObjects): Lying object to get OBB, outer_shape=[1,]

    Returns:
        LatentOBB: OBB from latent object. No outer shape is allowed.
    """
    if len(obj.outer_shape) != 1 or obj.outer_shape[0] != 1:
        raise ValueError("Outer shape must be [1,].")

    # Assume the object is laid down... one of the axes of obb is alway z.
    dir_1 = jnp.array([0., 0., 1.])
    dist_1, low_1, high_1 = get_directional_range(models, obj, dir_1)

    # Search the shortest range in xy plane.
    num_theta_interpolation = 100
    thetas = jnp.linspace(0, jnp.pi, num_theta_interpolation)
    for theta in thetas:
        normal_dir = jnp.array([jnp.cos(theta), jnp.sin(theta), 0.])
        dist, low, high = get_directional_range(models, obj, normal_dir)
        if dist < dist_2:
            dist_2 = dist
            dir_2, low_2, high_2 = normal_dir, low, high
    
    # Define the third direction
    dir_3 = jnp.cross(dir_1, dir_2)
    dist_3, low_3, high_3 = get_directional_range(models, obj, dir_3)
    print(f"Principal axes of OBB: {dist_1}, {dist_2}, {dist_3}")

    # Some equation of planes centered at the OBB center
    d_1 = -0.5*((jnp.dot(dir_1, low_1)+jnp.dot(dir_1, high_1)))
    d_2 = -0.5*((jnp.dot(dir_2, low_2)+jnp.dot(dir_2, high_2)))
    d_3 = -0.5*((jnp.dot(dir_3, low_3)+jnp.dot(dir_3, high_3)))
    plane1 = jnp.array([dir_1[0], dir_1[1], dir_1[2], d_1])
    plane2 = jnp.array([dir_2[0], dir_2[1], dir_2[2], d_2])
    plane3 = jnp.array([dir_3[0], dir_3[1], dir_3[2], d_3])
    
    # OBB definition
    center = find_intersecting_point(plane1, plane2, plane3)
    dirs = jnp.stack([dir_1, dir_2, dir_3])
    dists = jnp.stack([dist_1, dist_2, dist_3])
    
    return LatentOBB(center, dirs, dists)



def get_obb_from_lying_latent_object_2(
        models: flax.linen.Module,
        obj: cxutil.LatentObjects,
) -> LatentOBB:
    """Get the OBB of the latent object.
    
    This function assumes the object is always laid down. This function is not jit compatible.
    
    Args:
        models (flax.linen.Module): Models
        obj (cxutil.LatentObjects): Lying object to get OBB, outer_shape=[1,]

    Returns:
        LatentOBB: OBB from latent object. No outer shape is allowed.
    """
    # if len(obj.outer_shape) != 1 or obj.outer_shape[0] != 1:
    #     raise ValueError("Outer shape must be [1,].")

    # Assume the object is laid down... one of the axes of obb is alway z.
    dir_1 = jnp.array([0., 0., 1.])
    dist_1, low_1, high_1 = get_directional_range_2(models, obj, dir_1)

    # Search the shortest range in xy plane.
    num_theta_interpolation = 100
    thetas = jnp.linspace(0, jnp.pi, num_theta_interpolation)
    normal_dirs = jnp.c_[jnp.cos(thetas), jnp.sin(thetas), jnp.zeros_like(thetas)]
    dist, low, high = jax.vmap(partial(get_directional_range_2, models, obj))(normal_dirs)
    min_idx = jnp.argmin(dist)
    dir_2 = normal_dirs[min_idx]
    dist_2 = dist[min_idx]
    low_2 = low[min_idx]
    high_2 = high[min_idx]
    
    # Define the third direction
    dir_3 = jnp.cross(dir_1, dir_2)
    dist_3, low_3, high_3 = get_directional_range_2(models, obj, dir_3)
    dirs = jnp.stack([dir_1, dir_2, dir_3])
    low_pnts = jnp.stack([low_1, low_2, low_3])
    high_pnts = jnp.stack([high_1, high_2, high_3])
    dists = jnp.stack([dist_1, dist_2, dist_3])

    return dirs, low_pnts, high_pnts, dists

def get_directional_range(
        models: flax.linen.Module, 
        obj: cxutil.LatentObjects, 
        normal_dir: jnp.ndarray,
        num_interpolation=100
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]: 
    """Get the occupied range of the object given an arbitrary normal direction for one particle shape.

    Args:
        models (flax.linen.Module): Module
        obj (cxutil.LatentObjects): Object to check. outer_shape must be [1,].
        normal_dir (jnp.ndarray): outer_shape must be none.
        num_interpolation (int, optional): Defaults to 100.

    Returns:
        dir_length (jnp.ndarray): Length in the direction. 0-D array scalar
        low_point (jnp.ndarray): The lowest point of the object. 0-D array scalar
        high_point (jnp.ndarry): The highest point of the object. 0-D array scalar
    """
    if len(obj.outer_shape) != 1 or obj.outer_shape[0] != 1:
        raise ValueError("Outer shape must be [1,].")

    # Use plane predictor for the lowest/highest range estimation by adjusting the plane
    heights_list = jnp.linspace(-0.5, 0.5, num_interpolation)
    heights_point_list = jnp.zeros((num_interpolation,3))
    for i in range(3):
        heights_point_list = heights_point_list.at[...,i].set(normal_dir[i]*heights_list)
    normal_pos = normal_dir
    normal_neg = -normal_dir
    res_pos = models.apply('pln_predictor', obj, heights_point_list, normal_pos) # (1, num)
    res_neg = models.apply('pln_predictor', obj, heights_point_list, normal_neg) # (1, num)
    res_pos = jnp.squeeze(res_pos, axis=0)  # (num)
    res_neg = jnp.squeeze(res_neg, axis=0)  # (num)

    # Swap if upside down
    if res_pos[0]<0:
        tmp = res_pos
        res_pos = res_neg
        res_neg = tmp

    # Get the farthest height from up and down
    low_point = None
    high_point = None
    for i in range(num_interpolation-1):
        if low_point is None and res_pos[i+1]<0:
            low_point = heights_point_list[i]
        if high_point is None and res_neg[i+1]>0:
            high_point = heights_point_list[i+1]
        if low_point is not None and high_point is not None: 
            break

    dir_length = jnp.linalg.norm(high_point-low_point)
    return dir_length, low_point, high_point



def get_directional_range_2(
        models: flax.linen.Module, 
        obj: cxutil.LatentObjects, 
        normal_dir: jnp.ndarray,
        num_interpolation=100
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]: 
    """Get the occupied range of the object given an arbitrary normal direction for one particle shape.

    Args:
        models (flax.linen.Module): Module
        obj (cxutil.LatentObjects): Object to check. outer_shape must be [1,].
        normal_dir (jnp.ndarray): outer_shape must be none.
        num_interpolation (int, optional): Defaults to 100.

    Returns:
        dir_length (jnp.ndarray): Length in the direction. 0-D array scalar
        low_point (jnp.ndarray): The lowest point of the object. 0-D array scalar
        high_point (jnp.ndarry): The highest point of the object. 0-D array scalar
    """

    obj_centered = obj.replace(pos=jnp.zeros_like(obj.pos))

    # Use plane predictor for the lowest/highest range estimation by adjusting the plane
    heights_list = jnp.linspace(0, 0.2, num_interpolation)
    heights_point_list = normal_dir*heights_list[...,None]
    res_pos = models.apply('pln_predictor', obj_centered, heights_point_list, -normal_dir) # (num,)
    res_neg = models.apply('pln_predictor', obj_centered, -heights_point_list, normal_dir) # (num,)
    
    pos_idx = jnp.argmin(jnp.where(res_pos>0, np.arange(num_interpolation), jnp.inf))
    neg_idx = jnp.argmin(jnp.where(res_neg>0, np.arange(num_interpolation), jnp.inf))
    high_point = heights_point_list[pos_idx]
    low_point = -heights_point_list[neg_idx]

    dir_length = jnp.linalg.norm(high_point-low_point)
    return dir_length, obj.pos+low_point, obj.pos+high_point


def find_intersecting_point(plane1: jnp.ndarray, plane2: jnp.ndarray, plane3: jnp.ndarray) -> jnp.ndarray:
    """Find the intersecting point of three planes by solving linear systems of equations.

    Args:
        plane1 (jnp.ndarray): shape=[4,]
        plane2 (jnp.ndarray): shape=[4,]
        plane3 (jnp.ndarray): shape=[4,]

    Returns:
        jnp.ndarray: Intersecting point. shape=[3,]
    """
    coefficients = jnp.array([plane1[:3], plane2[:3], plane3[:3]])
    constants = -jnp.array([plane1[3], plane2[3], plane3[3]])
    intersecting_point = jnp.linalg.solve(coefficients, constants)
    return intersecting_point


def OBB_intersection(obb1_cen, obb1_dir, obb1_ext, obb2_cen, obb2_dir, obb2_ext):
    '''
    obb - centers, directions, extents
    '''

    base_sign = jnp.array([[1,1,1], [1,1,-1], 
                           [1,-1,1], [-1,1,1], 
                           [1,-1,-1], [-1,1,-1], 
                           [-1,-1,1], [-1,-1,-1]], dtype=jnp.float32)
    edge_indices = jnp.array([[0,1], [0,2], [0,3], [1,4], [1,5], [2,4], [2,6], [3,5], [3,6], [4,7], [5,7], [6,7]], dtype=jnp.int32)

    obb1_vtx = obb1_cen[...,None,:] + jnp.sum(obb1_dir[...,None,:,:]*base_sign[...,None,None]*obb1_ext[...,None,None], axis=-2) # (... 8, 3)
    obb2_vtx = obb2_cen[...,None,:] + jnp.sum(obb2_dir[...,None,:,:]*base_sign[...,None,None]*obb2_ext[...,None,None], axis=-2) # (... 8, 3)
    obb1_edge = obb1_vtx[...,edge_indices,:] # (... 12, 2, 3)
    obb2_edge = obb2_vtx[...,edge_indices,:] # (... 12, 2, 3)

    raise NotImplementedError



# def get_upright_inplace_pq(latent_obb: LatentOBB) -> Tuple[jnp.ndarray, jnp.ndarray]:
#     """Get left transformation that rotates the obb upright inplace"""
#     rot = latent_obb.dirs
#     after_rot_pos_inverse = -rot@latent_obb.center 
#     p = after_rot_pos_inverse \
#       + latent_obb.center \
#       + jnp.array([0, 0, latent_obb.dists[2]/2.])
#     q = tutil.Rm2q(rot)

#     return p, q