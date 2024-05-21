from __future__ import annotations

from typing import Sequence, Tuple
import jax.numpy as jnp
import numpy as np
import jax
import einops
from functools import partial

import util.transform_util as tutil
import util.structs as structs

# Typing
import numpy.typing as npt
IntrinsicT = Tuple[int, int, float, float, float, float]


def default_intrinsic(pixel_size:Sequence[int]) -> IntrinsicT:
    """Init default intrinsic from image.

    Args:
        pixel_size (Tuple[int, int]): (height, width).
    
    Returns:
        width (int): Image width
        height (int): Image height
        Fx (float): Focal length of x dimension
        Fy (float): Focal length of y dimension
        Cx (float): Center of x
        Cy (float): Center of y
    """
    return jnp.array([pixel_size[1], pixel_size[0], pixel_size[1], pixel_size[0], 0.5*pixel_size[1], 0.5*pixel_size[0]])


def pixel_ray(pixel_size:Sequence[int], cam_pos:jnp.ndarray, cam_quat:jnp.ndarray, 
                intrinsic:jnp.ndarray, near:float, far:float, coordinate:str='opengl'):
    '''
    bachable
    pixel_size : (2,) i j order
    cam_pos : (... 3) camera position
    cam_quat : (... 4) camera quaternion
    intrinsic : (... 6) camera intrinsic
    
    coordinate : 'opengl' or 'open3d' - oepngl: forward direction is z minus axis / open3d: forward direction is z plus

    return:
    ray start points, ray end points, ray directions
    '''
    cam_zeta = intrinsic[...,2:]
    zeros = jnp.zeros_like(cam_zeta[...,0])
    ones = jnp.ones_like(cam_zeta[...,0])
    # K_mat = jnp.stack([jnp.stack([cam_zeta[...,0], zeros, cam_zeta[...,2]],-1),
    #                     jnp.stack([zeros, cam_zeta[...,1], cam_zeta[...,3]],-1),
    #                     jnp.stack([zeros,zeros,ones],-1)],-2)
    K_mat = intrinsic_to_Kmat(intrinsic)
    # pixel= PVM (colomn-wise)
    # M : points
    # V : inv(cam_SE3)
    # P : Z projection and intrinsic matrix  
    x_grid_idx, y_grid_idx = jnp.meshgrid(jnp.arange(pixel_size[1])[::-1], jnp.arange(pixel_size[0])[::-1])
    pixel_pnts = jnp.concatenate([x_grid_idx[...,None], y_grid_idx[...,None], jnp.ones_like(y_grid_idx[...,None])], axis=-1)
    pixel_pnts = pixel_pnts.astype(jnp.float32)
    K_mat_inv = jnp.linalg.inv(K_mat)
    pixel_pnts = jnp.matmul(K_mat_inv[...,None,None,:,:], pixel_pnts[...,None])[...,0]
    if coordinate == 'opengl':
        # pixel_pnts = pixel_pnts.at[...,-1].set(-pixel_pnts[...,-1])
        pixel_pnts = jnp.c_[pixel_pnts[...,:-1], -pixel_pnts[...,-1:]]
        pixel_pnts = pixel_pnts[...,::-1,:]
    rays_s_canonical = pixel_pnts * near
    rays_e_canonical = pixel_pnts * far

    # cam SE3 transformation
    rays_s = tutil.pq_action(cam_pos[...,None,None,:], cam_quat[...,None,None,:], rays_s_canonical)
    rays_e = tutil.pq_action(cam_pos[...,None,None,:], cam_quat[...,None,None,:], rays_e_canonical)
    ray_dir = rays_e - rays_s
    ray_dir_normalized = tutil.normalize(ray_dir)

    return rays_s, rays_e, ray_dir_normalized


def pbfov_to_intrinsic(
        img_size: Tuple[int, int], 
        fov_deg: float
) -> IntrinsicT:
    """Calculate intrinsic components from fov.

    Assumes balanced focal length Fx=Fy.

    Args:
        img_size (Tuple[int, int]): (height, width).
        fov_degree (float): Vertical FOV in degree (y dimension, height)
    
    Returns:
        width (int): Image width
        height (int): Image height
        Fx (float): Focal length of x dimension
        Fy (float): Focal length of y dimension
        Cx (float): Center of x
        Cy (float): Center of y
    """
    fov_rad = fov_deg * np.pi/180.0
    Fy = img_size[0]*0.5/(np.tan(fov_rad*0.5))
    Fx = Fy
    Cx = img_size[1]*0.5
    Cy = img_size[0]*0.5
    return (img_size[1], img_size[0], Fx, Fy, Cx, Cy)


def intrinsic_to_fov(intrinsic: npt.NDArray):
    img_size_xy = intrinsic[...,:2]
    fovs = np.arctan(intrinsic[...,1]/intrinsic[...,3]*0.5)*2
    return fovs, img_size_xy[...,0] / img_size_xy[...,1]

def intrinsic_to_pb_lrbt(
        intrinsic: Sequence[IntrinsicT]|npt.NDArray, 
        near: float
) -> Tuple[float, float, float, float]:
    """

    Args:
        intrinsic (Sequence[IntrinsicT] | npt.NDArray): Intrinsic either in numpy or tuple...
        near (float): OpenGL near val

    Returns:
        Tuple[float, float, float, float]: ????
    """

    if isinstance(intrinsic, list) or isinstance(intrinsic, tuple):
        intrinsic = np.array(intrinsic)
    pixel_size = intrinsic[...,:2]
    fx = intrinsic[...,2]
    fy = intrinsic[...,3]
    cx = intrinsic[...,4]
    cy = intrinsic[...,5]
    
    halfx_px = pixel_size[...,0]*0.5
    center_px = cx - halfx_px
    right_px = center_px + halfx_px
    left_px = center_px - halfx_px

    halfy_px = pixel_size[...,1]*0.5
    center_py = cy - halfy_px
    bottom_px = center_py - halfy_px
    top_px = center_py + halfy_px

    return left_px/fx*near, right_px/fx*near, bottom_px/fy*near, top_px/fy*near


def intrinsic_to_Kmat(intrinsic):
    '''
    flip y direction - because of our wrong coordinate system...
    '''
    zeros = jnp.zeros_like(intrinsic[...,2])
    # return jnp.stack([jnp.stack([intrinsic[...,2], zeros, intrinsic[...,4]], -1),
    #             jnp.stack([zeros, intrinsic[...,3], intrinsic[...,5]], -1),
    #             jnp.stack([zeros, zeros, jnp.ones_like(intrinsic[...,2])], -1)], -2)
    return jnp.stack([jnp.stack([intrinsic[...,2], zeros, intrinsic[...,4]], -1),
                jnp.stack([zeros, intrinsic[...,3], intrinsic[...,1] - intrinsic[...,5]], -1),
                jnp.stack([zeros, zeros, jnp.ones_like(intrinsic[...,2])], -1)], -2)
    

def global_pnts_to_pixel(intrinsic, cam_posquat, pnts, expand=False):
    '''
    expand==False
        intrinsic, cam_posquat, pnts should have same dim

    expand==True
        intrinsic, cam_posquat : (... NR ...)
        pnts : (... NS 3)
        return - (... NR NS 2), out
    '''
    if not isinstance(cam_posquat, tuple):
        cam_posquat = (cam_posquat[...,:3], cam_posquat[...,3:])
    
    if expand:
        intrinsic, cam_posquat = jax.tree_map(lambda x: x[...,None,:], (intrinsic, cam_posquat))
        pnts = pnts[...,None,:,:]

    pixel_size = intrinsic[...,:2]
    pnt_img_pj = tutil.pq_action(*tutil.pq_inv(*cam_posquat), pnts) # (... NS NR 3)
    kmat = intrinsic_to_Kmat(intrinsic)
    px_coord_xy = jnp.einsum('...ij,...j', kmat[...,:2,:2], pnt_img_pj[...,:2]/(-pnt_img_pj[...,-1:])) + kmat[...,:2,2]
    out_pnts = jnp.logical_or(jnp.any(px_coord_xy<0, -1), jnp.any(px_coord_xy>=pixel_size, -1))
    px_coord_xy = jnp.clip(px_coord_xy, 0.001, pixel_size-0.001)
    px_coord_ij = jnp.stack([pixel_size[...,1]-px_coord_xy[...,1], px_coord_xy[...,0]], -1)
    return px_coord_ij, out_pnts
    # px_coord = px_coord.astype(jnp.float32)
    # px_coord = jnp.stack([-px_coord[...,1], px_coord[...,0]] , -1)# xy to ij

def cam_info_to_render_params(cam_info):
    cam_posquat, intrinsic = cam_info

    return dict(
        intrinsic=intrinsic,
        pixel_size=jnp.c_[intrinsic[...,1:2], intrinsic[...,0:1]].astype(jnp.int32),
        camera_pos=cam_posquat[...,:3],
        camera_quat=cam_posquat[...,3:],
    )


def pcd_from_depth(depth, intrinsic, pixel_size, coordinate='opengl', visualize=False):
    if depth.shape[-1] != 1:
        depth = depth[...,None]

    xgrid, ygrid = np.meshgrid(np.arange(pixel_size[1]), np.arange(pixel_size[0]), indexing='xy')
    xygrid = np.stack([xgrid, ygrid], axis=-1)
    xy = (xygrid - intrinsic[...,None,None,4:6]) * depth / intrinsic[...,None,None,2:4]

    xyz = jnp.concatenate([xy, depth], axis=-1)

    if coordinate=='opengl':
       xyz =  jnp.stack([xyz[...,0], -xyz[...,1], -xyz[...,2]], axis=-1)

    if visualize:
        import open3d as o3d
        pcd_o3d = o3d.geometry.PointCloud()
        pcd_o3d.points = o3d.utility.Vector3dVector(xyz.reshape(-1,3))
        mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=0.1, origin=[0, 0, 0])
        o3d.visualization.draw_geometries([pcd_o3d, mesh_frame])

    return xyz

def resize_intrinsic(intrinsic, origin_pixel_size, output_pixel_size):
    intrinsic = jnp.array(intrinsic)
    # output_pixel_size_xy = (output_pixel_size[1], output_pixel_size[0])
    idx1 = 0
    idx2 = 1
    scale = origin_pixel_size[idx1]/output_pixel_size[idx1]
    croped_size = np.array(output_pixel_size) * scale
    if origin_pixel_size[1] - croped_size[1] < 0:
        idx1 = 1
        idx2 = 0
        scale = origin_pixel_size[idx1]/output_pixel_size[idx1]
        croped_size = np.array(output_pixel_size) * scale
    dif = int((origin_pixel_size[idx2] - croped_size[idx2])*0.5)

    cam_intr_resized = intrinsic.at[:,-1-idx2].set(intrinsic[:,-1-idx2] - dif)
    cam_intr_resized = cam_intr_resized*output_pixel_size[idx1]/origin_pixel_size[idx1]
    cam_intr_resized = cam_intr_resized.at[:,:2].set(np.array([output_pixel_size[1], output_pixel_size[0]]))
    return cam_intr_resized

def resize_img(img, output_pixel_size, method='linear', outer_extend=True):
    img = jnp.array(img)
    if img.shape[-3:-1] == output_pixel_size:
        return img
    if img.shape[-1] > 10 and outer_extend:
        img = img[...,None]
    if img.ndim==3:
        img = img[None]
    origin_pixel_size = img.shape[-3:-1]
    origin_dtype = img.dtype
    idx1 = 0
    idx2 = 1
    scale = origin_pixel_size[idx1]/output_pixel_size[idx1]
    croped_size = np.array(output_pixel_size) * scale
    if origin_pixel_size[1] - croped_size[1] < 0:
        idx1 = 1
        idx2 = 0
        scale = origin_pixel_size[idx1]/output_pixel_size[idx1]
        croped_size = np.array(output_pixel_size) * scale
    dif = int((origin_pixel_size[idx2] - croped_size[idx2])*0.5)

    if idx1 == 0:
        cropped = img[...,:,dif:img.shape[-2]-dif, :]
    else:
        cropped = img[...,dif:img.shape[-3]-dif, :, :]
    resize_output_shape = (*cropped.shape[:-3], *output_pixel_size, cropped.shape[-1])
    img_rsized = jax.image.resize(cropped, resize_output_shape, method=method)
    return img_rsized.astype(origin_dtype)


def visualize_pcd(rgb_list_, depth_list_, cam_posquat_, intrinsic_, return_elements=False, pcd_o3d=None, tag_size=None, area=None):
    pixel_size_ = (intrinsic_[0,1], intrinsic_[0,0])
    intrinsic_matrix = intrinsic_to_Kmat(intrinsic_)
    pcd = pcd_from_depth(depth_list_, intrinsic_, pixel_size_, visualize=False)

    pcd_flat = pcd.reshape(cam_posquat_.shape[0],-1,3)

    pcd_tf = tutil.pq_action(cam_posquat_[...,None,:3], cam_posquat_[...,None,3:], pcd_flat)

    import open3d as o3d

    cam_vis_list = []
    cam_pq_mesh_list = []
    for i in range(cam_posquat_.shape[0]):
        x, y = intrinsic_[i,:2].astype(np.int32)
        Tm = tutil.pq2H(cam_posquat_[i,:3], cam_posquat_[i,3:])
        mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=0.1, origin=[0, 0, 0])
        mesh_frame.transform(np.array(Tm))
        cam_pq_mesh_list.append(mesh_frame)
        Tm = np.linalg.inv(Tm)
        cam_vis_list.append(o3d.geometry.LineSet.create_camera_visualization(view_width_px=x, view_height_px=y, 
                                                                            intrinsic=intrinsic_matrix[i], extrinsic=Tm, scale=0.1))

    pcd_tf = pcd_tf.reshape(-1,3)
    rgb_list_reshape = rgb_list_.reshape(-1,3)/255.
    if area is not None:
        valid_pcd_mask = np.logical_and(np.all(pcd_tf>area[0],-1), np.all(pcd_tf<area[1],-1))
        pcd_tf = pcd_tf[valid_pcd_mask]
        rgb_list_reshape = rgb_list_reshape[valid_pcd_mask]

    if pcd_o3d is None:
        pcd_o3d = o3d.geometry.PointCloud()
    pcd_o3d.points = o3d.utility.Vector3dVector(pcd_tf)
    pcd_o3d.colors = o3d.utility.Vector3dVector(rgb_list_reshape)
    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=0.1, origin=[0, 0, 0])
    
    tag_pnts = []
    if tag_size is not None:
        test_pnts = np.array([[0.,0.,0.],
                        [tag_size/2., tag_size/2., 0.],
                        [-tag_size/2., tag_size/2., 0.],
                        [tag_size/2., -tag_size/2., 0.],
                        [-tag_size/2., -tag_size/2., 0.],
                        [tag_size/2., 0., 0.],
                        [-tag_size/2., 0., 0.],
                        [0., tag_size/2., 0.],
                        [0., -tag_size/2., 0.],])
        for test_pnt_ in test_pnts:
            pnts_o3d = o3d.geometry.TriangleMesh.create_sphere(radius=0.010)
            pnts_o3d.compute_vertex_normals()
            pnts_o3d.paint_uniform_color(np.array([1.0,0,0]))
            pnts_o3d.translate(test_pnt_)
            tag_pnts.append(pnts_o3d)

    ## o3d point projection test
    # o3d_intrinsic = o3d.camera.PinholeCameraIntrinsic()
    # o3d_intrinsic.set_intrinsics(int(intrinsic_[0,0]), int(intrinsic_[0,1]), intrinsic_[0,2], intrinsic_[0,3], intrinsic_[0,4], intrinsic_[0,5])
    # o3d_depth = o3d.geometry.Image((depth_list_[0]*1000).astype(np.uint16))
    # pcd_from_o3d = o3d.geometry.PointCloud.create_from_depth_image(o3d_depth, o3d_intrinsic, depth_scale=1000.0, depth_trunc=1000.0, stride=1)

    if not return_elements:
        o3d.visualization.draw_geometries([*cam_vis_list, *cam_pq_mesh_list, pcd_o3d, mesh_frame, *tag_pnts])
    else:
        return [*cam_vis_list, *cam_pq_mesh_list, pcd_o3d, mesh_frame, *tag_pnts], pcd_tf, rgb_list_reshape

def partial_pcd_from_depth(depth, intrinsic, pixel_size, coordinate='opengl', visualize=False):
    if depth.shape[-1] != 1:
        depth = depth[...,None]

    # depth (#cam, w, h, 1)

    xgrid, ygrid = np.meshgrid(np.arange(pixel_size[1]), np.arange(pixel_size[0]), indexing='xy')
    xygrid = np.stack([xgrid, ygrid], axis=-1)
    xy = (xygrid - intrinsic[...,None,None,4:6]) * depth / intrinsic[...,None,None,2:4]
    # xy (#cam, w, h, #particle)

    xyz = jnp.concatenate([xy, depth], axis=-1)
    if coordinate=='opengl':
       xyz =  jnp.stack([xyz[...,0], -xyz[...,1], -xyz[...,2]], axis=-1)

    if visualize:
        import open3d as o3d
        pcd_o3d = o3d.geometry.PointCloud()
        pcd_o3d.points = o3d.utility.Vector3dVector(xyz.reshape(-1,3))
        mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=0.1, origin=[0, 0, 0])
        o3d.visualization.draw_geometries([pcd_o3d, mesh_frame])

    return xyz

def gen_partial_pcd(depth_list_, seg_boolean, cam_posquat_, intrinsic_, o3dvis=None):
    pixel_size_ = (intrinsic_[0,1], intrinsic_[0,0])
    intrinsic_matrix = intrinsic_to_Kmat(intrinsic_)
    pcd = partial_pcd_from_depth(depth_list_, intrinsic_, pixel_size_, visualize=False)

    # pcd (#cam, w, h, 3 (xyz))
    # seg_boolean (#cam, w, h, #particle)
    total_partial_pcd = []
    for particle_num in range(seg_boolean.shape[-1]):

        seg_boolean_mask = seg_boolean[..., particle_num].reshape(-1) # (#cam, w, h) -> (-1)

        pcd_flat = pcd.reshape(cam_posquat_.shape[0],-1,3)

        pcd_tf = tutil.pq_action(cam_posquat_[...,None,:3], cam_posquat_[...,None,3:], pcd_flat)
        pcd_tf = pcd_tf.reshape(-1,3)
        partial_pcd = []
        for i in range(pcd_tf.shape[0]):
            if seg_boolean_mask[i]: partial_pcd.append(pcd_tf[i, :])

        partial_pcd = np.array(partial_pcd)
        total_partial_pcd.append(partial_pcd)

        import open3d as o3d

        cam_vis_list = []
        cam_pq_mesh_list = []
        for i in range(cam_posquat_.shape[0]):
            x, y = intrinsic_[i,:2].astype(np.int32)
            Tm = tutil.pq2H(cam_posquat_[i,:3], cam_posquat_[i,3:])
            mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
                size=0.1, origin=[0, 0, 0])
            mesh_frame.transform(np.array(Tm))
            cam_pq_mesh_list.append(mesh_frame)
            Tm = np.linalg.inv(Tm)
            cam_vis_list.append(o3d.geometry.LineSet.create_camera_visualization(view_width_px=x, view_height_px=y, 
                                                                                intrinsic=intrinsic_matrix[i], extrinsic=Tm, scale=0.1))

        pcd_o3d = o3d.geometry.PointCloud()
        pcd_o3d.points = o3d.utility.Vector3dVector(partial_pcd)
        mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=0.1, origin=[0, 0, 0])
        if o3dvis is None:
            o3d.visualization.draw_geometries([*cam_vis_list, *cam_pq_mesh_list, pcd_o3d, mesh_frame])
        else:
            for o in [*cam_vis_list, *cam_pq_mesh_list, pcd_o3d, mesh_frame]:
                o3dvis.update_geometry(o)
            o3dvis.poll_events()
            o3dvis.update_renderer()
    
    return total_partial_pcd



def take_img_features(img_ft, intrinsic, cam_posquat, qpnts):
    '''
    input
    img_ft : (... NI NJ NF)
    intrinsic : (..., 7)
    cam posquat : (..., 3), (..., 4)
    qpnts : (..., 3)
    '''
    img_ft_size = jnp.array(img_ft.shape[-3:-1])
    input_img_size_ = intrinsic[...,:2]
    px_coord_ctn, out_pnts_indicator = global_pnts_to_pixel(intrinsic, cam_posquat, qpnts) # (... 2)
    px_coord_ctn = px_coord_ctn/input_img_size_ * jnp.array(img_ft_size).astype(jnp.float32)

    px_coord = (px_coord_ctn-0.5).astype(jnp.int32) # centering # (... 2)

    # apply bilinear interpolation
    min_pnt = px_coord.astype(jnp.float32) + 0.5
    bound_pnt = min_pnt[...,None,:] + jnp.array([[0,0], [0,1], [1,0], [1,1]], dtype=jnp.float32)
    bound_pnt = bound_pnt.clip(0.5, img_ft_size-0.5)
    px_coord_residual = jnp.abs(px_coord_ctn[...,None,:].clip(0.5, img_ft_size.astype(jnp.float32)-0.5) - bound_pnt)[...,(3,2,1,0),:]
    px_coord_ratio = jnp.prod(px_coord_residual+1e-2, axis=-1)
    px_coord_ratio = px_coord_ratio/jnp.sum(px_coord_ratio, axis=-1, keepdims=True).clip(1e-4)
    px_coord_ratio = jax.lax.stop_gradient(px_coord_ratio)
    # assert jnp.sum(jnp.sum(px_coord_ratio, axis=-1) < 1-1e-5) == 0

    px_coord = px_coord.clip(0, img_ft_size-1).astype(jnp.int32)
    px_coord_bound = px_coord[...,None,:] + jnp.array([[0,0], [0,1], [1,0], [1,1]], dtype=jnp.int32)
    px_coord_bound = px_coord_bound.clip(0, img_ft_size-1)

    px_flat_idx = px_coord_bound[...,1] + px_coord_bound[...,0] * img_ft_size[...,1] # (... 4)

    img_fts_flat = einops.rearrange(img_ft, '... i j k -> ... (i j) k')
    selected_img_fts = jnp.take_along_axis(img_fts_flat, px_flat_idx[...,None], axis=-2) # (... 4 NF)
    img_ft_res = jnp.sum(px_coord_ratio[...,None] * selected_img_fts, axis=-2)
    img_ft_res = jnp.concatenate([out_pnts_indicator[...,None].astype(jnp.float32), img_ft_res], axis=-1)

    return img_ft_res

def resize_rgb_and_cam_info(rgb, cam_info, output_pixel_size):
    '''
    input : (NB, ...) or (NB, NC ...)
    '''
    if rgb.ndim == 4:
        rgb, cam_info = jax.tree_map(lambda x: x[None], (rgb, cam_info))

    intrinsic_rs = jax.vmap(partial(resize_intrinsic, origin_pixel_size=rgb.shape[-3:-1], output_pixel_size=output_pixel_size))(cam_info[1])
    cam_info_init_rs = (cam_info[0], intrinsic_rs)
    rgb_rs = jax.vmap(partial(resize_img, output_pixel_size=output_pixel_size))(rgb)

    return rgb_rs, cam_info_init_rs

def default_cond_feat(pixel_size)->structs.ImgFeatures:
    intrinsic = default_intrinsic(pixel_size)
    cam_pos = jnp.array([0,0,-1])
    cam_quat = tutil.aa2q(jnp.array([0,0,0.]))
    cam_posquat = jnp.concatenate([cam_pos, cam_quat], -1)
    return structs.ImgFeatures(intrinsic[None], cam_posquat[None], None)
    