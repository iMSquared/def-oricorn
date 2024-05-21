from __future__ import annotations
import jax
import jax.numpy as jnp
import einops
import numpy as np
import numpy.typing as npt
from typing import Tuple
from dataclasses import replace
from flax.struct import dataclass as flax_dataclass
from pybullet_utils.bullet_client import BulletClient
from functools import partial

import util.transform_util as tutil
import util.GJK as GJK
import util.ev_util.rotm_util as rmutil

def vex_obj_parsing(
        filename: str, 
        max_dec_size: int, 
        max_vertices: int,
        scale:float=1.0, 
        visualize: bool = False
) -> Tuple[npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray]:

    """Parse mesh from file to numpy

    Args:
        filename (str): Path to the file.
        max_dec_size (int): Max number of convex decompositions defined in the mesh.
        max_vertices (int): Max number of vertices for each convex defined in the mesh.
        visualize (bool, optional): Open debug visualizer. Defaults to False.

    Returns:
        vertices (npt.NDArry): Vertices. Padded with ?
        faces (npt.NDArry): Faces. Padded with ?
        vtx_no (npt.NDArry): Number of vertices in the convex
        face_no (npt.NDArry): Number of faces in the convex
    """
    vertices = 1000000*np.ones((max_dec_size, max_vertices, 3))
    if max_vertices==16:
        max_faces = 28
    elif max_vertices==32:
        max_faces = 60
    else:
        max_faces = max_vertices*2-4
    faces = -np.ones((max_dec_size, max_faces, 3), dtype=np.int32)
    vtx_no = np.zeros(max_dec_size, dtype=np.int32)
    face_no = np.zeros(max_dec_size, dtype=np.int32)
    f = open(filename)
    nd = -1
    nv = 0
    nf = 0
    for line in f:
        if line[:2] == "o ":
            vtx_no[nd] = nv
            face_no[nd] = nf
            nv = 0
            nf = 0
            nd += 1
        if line[:2] == "v ":
            if nd < 0:
                nd = 0
            index1 = line.find(" ") + 1
            index2 = line.find(" ", index1 + 1)
            index3 = line.find(" ", index2 + 1)
            vertex = (float(line[index1:index2])*scale, float(line[index2:index3])*scale, float(line[index3:])*scale)
            vertices[nd, nv] = vertex
            nv += 1
        if line[:2] == "f ":
            index1 = line.find(" ") + 1
            index2 = line.find(" ", index1 + 1)
            index3 = line.find(" ", index2 + 1)
            face_idx = (int(line[index1:index2].split("/")[0])-1, int(line[index2:index3].split("/")[0])-1, int(line[index3:].split("/")[0])-1)
            faces[nd, nf] = face_idx
            nf += 1
            
    vtx_no[nd] = nv
    face_no[nd] = nf
    f.close()

    # ## aabb compensations
    # center = cal_center_from_obj_file(filename)
    # vertices -= center

    if visualize:
        import open3d as o3d
        geom_list = []
        for i in range(vertices.shape[0]):
            pcd_ = o3d.geometry.PointCloud()
            pcd_.points = o3d.utility.Vector3dVector(vertices[i,:vtx_no[i]])
            pcd_.paint_uniform_color(np.random.uniform(0,1,size=(3,)))
            geom_list.append(pcd_)

        o3d.visualization.draw_geometries(geom_list)
    return vertices, faces, vtx_no, face_no


@flax_dataclass
class CvxObjects:
    pos: jnp.ndarray=jnp.zeros(3,)  # (... 3)
    color: jnp.ndarray=None         # (... 3)
    dc_rel_centers:jnp.ndarray=None
    rel_vtx: jnp.ndarray=None       # (... ND NV 3)
    fc: jnp.ndarray=None            # (... ND NF 3)
    rel_pcd: jnp.ndarray=None       # (... NP 3)
    pcd_dc_idx: jnp.ndarray=None    # (... NP 3)

    def __getitem__(self, idx: jnp.ndarray|int):
        """Convenient indexing for dataclass"""
        return jax.tree_map(lambda x: x[idx], self) 

    def init_obj_info(self, obj_info) -> "CvxObjects":
        if isinstance(obj_info, dict):
            pq = obj_info['obj_posquats'].astype(jnp.float32)
            fcs = obj_info['obj_cvx_faces_padded'].astype(jnp.int32)
            vtx = obj_info['obj_cvx_verts_padded'].astype(jnp.float32)
        elif isinstance(obj_info, tuple) or isinstance(obj_info, list):
            pq, vtx, fcs = obj_info
        # pq, vtx, fcs = obj_info
        return self.init_vtx(vtx, fcs).apply_pq_vtx(pq[...,:3], pq[...,3:])

    def init_vtx(self, vtx, fcs) -> "CvxObjects":
        vtx = jnp.array(vtx)
        fcs = jnp.array(fcs)
        c = get_volumetric_centroid(vtx, fcs)[...,None,None,:]
        dc_centers = get_volumetric_centroid_dc(vtx, fcs)
        self = replace(self, rel_vtx=vtx-c).validate_rel_vtx()
        self = replace(self, fc=fcs)
        self = replace(self, pos=jnp.squeeze(c, (-3,-2)))
        self = replace(self, dc_rel_centers=dc_centers-c.squeeze(-2))
        return self
        
    def init_pcd(self, pcd, dc_idx) -> "CvxObjects":
        assert self.pos is not None
        self = replace(self, rel_pcd=jnp.array(pcd[...,:3]).astype(jnp.float32)-self.pos[...,None,:])
        self = replace(self, pcd_dc_idx=jnp.array(dc_idx).astype(jnp.int32))
        return self


    def random_color(self, jkey) -> "CvxObjects":
        return replace(self, color=jax.random.uniform(jkey, self.outer_shape + (4,)).at[...,-1].set(1.))
    
    def broadcast(self, outer_shape) -> "CvxObjects":
        def broadcast_(elements:jnp.ndarray, inner_axis):
            if elements is not None:
                return jnp.broadcast_to(elements, outer_shape + elements.shape[inner_axis:])
            else:
                return None
        self=replace(self, rel_vtx=broadcast_(self.rel_vtx, -3))
        self=replace(self, fc=broadcast_(self.fc, -3))
        self=replace(self, pos=broadcast_(self.pos, -1))
        self=replace(self, rel_pcd=broadcast_(self.rel_pcd, -2))
        self=replace(self, pcd_dc_idx=broadcast_(self.pcd_dc_idx, -1))
        self=replace(self, color=broadcast_(self.color, -1))
        self=replace(self, dc_rel_centers=broadcast_(self.dc_rel_centers, -2))
        return self
    
    def apply_pq_vtx(self, pos, quat) -> "CvxObjects":
        pos_ = tutil.pq_action(pos, quat, self.pos)
        self = replace(self, pos=pos_)
        self = self.rotate_rel_vtx(quat)
        return self
    
    def apply_pq_vtxpcd(self, pos, quat) -> "CvxObjects":
        pos_ = tutil.pq_action(pos, quat, self.pos)
        self = replace(self, pos=pos_)
        self = self.rotate_rel_vtxpcd(quat)
        return self
    
    def translate(self, pos) -> "CvxObjects":
        return replace(self, pos=self.pos+pos)

    def fill_vtxfc(self, nv, nf) -> "CvxObjects":
        if self.rel_vtx is None:
            vtx = 1e6*jnp.ones(self.z.shape[:-1] + (nv,3))
            self=replace(self, rel_vtx=vtx)
        if self.fc is None:
            fc = -1*jnp.ones(self.z.shape[:-1] + (nf,3))
            self=replace(self, fc=fc)
        return self


    def rotate_rel_vtxpcd(self, quat) -> "CvxObjects":
        return self.rotate_rel_vtx(quat).rotate_rel_pcd(quat)

    def rotate_rel_vtx(self, quat) -> "CvxObjects":
        if self.rel_vtx is not None:
            return replace(self, rel_vtx= tutil.qaction(quat[...,None,None,:], self.rel_vtx)).validate_rel_vtx().rotate_dc_rel_centers(quat)
        else:
            return self

    def validate_rel_vtx(self) -> "CvxObjects":
        return replace(self, rel_vtx=jnp.where(self.vtx_valid_mask, self.rel_vtx, 1e6))

    def rotate_rel_pcd(self, quat) -> "CvxObjects":
        if self.rel_pcd is not None:
            return replace(self, rel_pcd=tutil.qaction(quat[...,None,:], self.rel_pcd))
        else:
            return self
    
    def rotate_dc_rel_centers(self, quat) -> "CvxObjects":
        if self.dc_rel_centers is not None:
            return replace(self, dc_rel_centers= tutil.qaction(quat[...,None,:], self.dc_rel_centers))
        else:
            return self

    def padding_or_none(self, objB, padding) -> "CvxObjects":
        def padding_or_none(a:jnp.ndarray, b:jnp.ndarray):
            if a is None and b is None:
                return None, None
            if a is not None and b is not None:
                return a, b
            if not padding:
                return None, None
            if a is None and b is not None:
                a = jnp.zeros(self.outer_shape + b.shape[len(objB.outer_shape):])
            if a is not None and b is None:
                b = jnp.zeros(objB.outer_shape + a.shape[len(self.outer_shape):])
            return a, b
        return jax.tree_map(lambda a, b: padding_or_none(a,b), self, objB)
    
    def concat(self, objectsB, axis, padding=False) -> "CvxObjects":
        '''
        axis : axis within outer shape
        '''
        if axis < 0:
            axis = len(self.outer_shape) + 1 + axis
        def padding_or_none(a:jnp.ndarray, b:jnp.ndarray):
            if a is None and b is None:
                return None
            if a is not None and b is not None:
                return jnp.concatenate((a, b), axis=axis)
            if not padding:
                return None
            if a is None and b is not None:
                a = jnp.zeros(self.outer_shape + b.shape[len(objectsB.outer_shape):])
            if a is not None and b is None:
                b = jnp.zeros(objectsB.outer_shape + a.shape[len(self.outer_shape):])
            return jnp.concatenate((a, b), axis=axis)
        return jax.tree_map(lambda *x: padding_or_none(*x), self, objectsB)


    def stack(self, objectsB, axis, padding=False) -> "CvxObjects":
        '''
        axis : axis within outer shape
        '''
        if axis < 0:
            axis = len(self.outer_shape) + 1 + axis
        def padding_or_none(a:jnp.ndarray, b:jnp.ndarray):
            if a is None and b is None:
                return None
            if a is not None and b is not None:
                return jnp.stack((a, b), axis=axis)
            if not padding:
                return None
            if a is None and b is not None:
                a = jnp.zeros(self.outer_shape + b.shape[len(objectsB.outer_shape):])
            if a is not None and b is None:
                b = jnp.zeros(objectsB.outer_shape + a.shape[len(self.outer_shape):])
            return jnp.stack((a, b), axis=axis)
        return jax.tree_map(lambda *x: padding_or_none(*x), self, objectsB)

    def extend_outer_shape(self, axis) -> "CvxObjects":
        if axis < 0:
            axis = len(self.outer_shape) + 1 + axis
        return jax.tree_map(lambda x: jnp.expand_dims(x, axis), self)
    
    def squeeze_outer_shape(self, axis) -> "CvxObjects":
        if axis < 0:
            axis = len(self.outer_shape) + 1 + axis
        return jax.tree_map(lambda x: jnp.squeeze(x, axis), self)

    def extend_and_repeat_outer_shape(self, r, axis) -> "CvxObjects":
        if axis < 0:
            axis = len(self.outer_shape) + 1 + axis
        self = self.extend_outer_shape(axis)
        return self.repeat_outer_shape(r, axis)

    def repeat_outer_shape(self, r, axis) -> "CvxObjects":
        if axis < 0:
            axis = len(self.outer_shape) + 1 + axis
        return jax.tree_map(lambda x: jnp.broadcast_to(x, x.shape[:axis] +(r,) + x.shape[axis+1:]), self)

    def pair_split(self, axis):
        objA = jax.tree_map(lambda x : jnp.split(x, 2, axis=axis)[0], self)
        objB = jax.tree_map(lambda x : jnp.split(x, 2, axis=axis)[1], self)
        return objA, objB
    
    def take_along_outer_axis(self, indices, axis) -> "CvxObjects":
        def align_indices_ndim(x, indices_):
            for _ in range(x.ndim - indices_.ndim):
                indices_ = indices_[...,None]
            return indices_
        return jax.tree_util.tree_map(lambda x: jnp.take_along_axis(x, align_indices_ndim(x, indices), axis), self)

    def reshape_outer_shape(self, new_shape):
        outer_ndim = len(self.outer_shape)
        return jax.tree_util.tree_map(lambda x: x.reshape(new_shape + x.shape[outer_ndim:]), self)

    def register_pcd(self, jkey, npoint) -> "CvxObjects":
        rel_spnts, dc_idx = sampling_from_surface_convex_dec(jkey, self.rel_vtx, self.fc, ns=npoint)
        return self.replace(rel_pcd=rel_spnts, pcd_dc_idx=dc_idx)

    def set_z_with_models(self, jkey, models, keep_gt_info=False, keep_pcd=True, dc_center_no=0) -> "LatentObjects":
        self = self.register_pcd(jkey, models.args.npoint)
        latent_obj:LatentObjects = models.apply('shape_encoder', self, jkey, det=True)
        if dc_center_no == 0 or dc_center_no is None or latent_obj.dc_rel_centers.shape[-2] == dc_center_no:
            pass
        else:
            _, jkey = jax.random.split(jkey)
            sample_idx = jax.random.permutation(jkey, latent_obj.dc_rel_centers.shape[-2])[:dc_center_no]
            latent_obj = latent_obj.replace(dc_rel_centers=latent_obj.dc_rel_centers[...,sample_idx,:])
        if keep_gt_info:
            return latent_obj.replace(
                # pos            = self.pos,
                color          = self.color,
                # dc_rel_centers = self.dc_rel_centers,
                rel_vtx        = self.rel_vtx,
                fc             = self.fc,
                rel_pcd        = self.rel_pcd,
                pcd_dc_idx     = self.pcd_dc_idx,
            )
        elif keep_pcd:
            return latent_obj.replace(
                rel_pcd        = self.rel_pcd,
            )
        else:
            return latent_obj

    @property
    def outer_shape(self):
        return self.rel_vtx.shape[:-3]

    @property
    def nd(self):
        return self.rel_vtx.shape[-3]

    @property
    def vtx_tf(self):
        return self.pos[...,None,None,:] + self.rel_vtx
    
    @property
    def pcd_tf(self):
        return self.pos[...,None,:] + self.rel_pcd
    
    @property
    def dc_centers_tf(self):
        return self.pos[...,None,:] + self.dc_rel_centers
    
    @property
    def vtx_centroid(self):
        return get_center(self.vtx_tf, self.fc, True)
    @property
    def dc_valid_mask(self):
        return self.fc_valid_mask[...,0,:].squeeze(-1)
    
    @property
    def vtx_valid_mask(self):
        return get_valid_vtx_mask(self.rel_vtx, True)
    
    @property
    def fc_valid_mask(self):
        return get_valid_fcs_mask(self.fc, True)
    
    @property
    def obj_valid_mask(self):
        return self.fc[...,0,0,0]>=0
    
    @property
    def fcvtx(self):
        return get_fcsvtx(self.vtx_tf, self.fc)
    
    @property
    def normal(self):
        return get_normals(self.vtx_tf, self.fc)



@flax_dataclass
class LatentObjects(CvxObjects):
    z:jnp.ndarray=None

    def drop_gt_info(self, color=False) -> "LatentObjects":
        self = replace(self, rel_vtx=None)
        self = replace(self, fc=None)
        # self = replace(self, rel_pcd=None)
        self = replace(self, pcd_dc_idx=None)
        if color:
            self = replace(self, color=None)
        return self
    
    def set_z(self, z) -> "LatentObjects":
        self = replace(self, z=z)
        return self

    def init_h(self, h, latent_shape) -> "LatentObjects":
        nd, nf, nz = latent_shape
        z = einops.rearrange(h[...,:nd*nf*nz], '... (i j k) -> ... i j k', i=nd, k=nz)
        pos = h[...,-3:]*0.3
        dc_rel_centers_tf = h[...,nd*nf*nz:-3]*0.3
        dc_rel_centers_tf = einops.rearrange(dc_rel_centers_tf, '... (i j) -> ... i j', j=3)
        return self.replace(z=z, dc_rel_centers=dc_rel_centers_tf - pos[...,None,:], pos=pos)
    
    def set_h(self, h) -> "LatentObjects":
        z = einops.rearrange(h[...,:self.nd*self.z_flat.shape[-1]], '... (i j k) -> ... i j k', i=self.nd, k=self.nz)
        pos = h[...,-3:]*0.3
        dc_rel_centers_tf = h[...,self.nd*self.z_flat.shape[-1]:-3]*0.3
        dc_rel_centers_tf = einops.rearrange(dc_rel_centers_tf, '... (i j) -> ... i j', j=3)
        return self.replace(z=z, dc_rel_centers=dc_rel_centers_tf - pos[...,None,:], pos=pos)

    def apply_pq_z(self, pos, quat, rot_configs) -> "LatentObjects":
        self = self.apply_pq_vtx(pos, quat)
        self = self.rotate_z(quat, rot_configs)
        return self

    def rotate_z(self, quat, rot_configs) -> "LatentObjects":
        '''
        rotate z while keeping center point
        '''
        self = replace(self, z=rmutil.apply_rot(self.z, tutil.q2R(quat), rot_configs, -2, 2))
        self = self.rotate_dc_rel_centers(quat)
        return self


    def broadcast(self, outer_shape) -> "LatentObjects":
        self = super().broadcast(outer_shape)
        def broadcast_(elements:jnp.ndarray, inner_axis):
            if elements is not None:
                return jnp.broadcast_to(elements, outer_shape + elements.shape[inner_axis:])
            else:
                return None
        self=replace(self, z=broadcast_(self.z, -3))
        return self
    
    def get_df(self, pos_weight=1) -> jnp.ndarray:
        return jnp.concatenate([self.z_flat, pos_weight*self.dc_centers_tf], -1) # assume one ND
    
    def register_pcd_from_latent(self, models, npoints, jkey, pcd_sample_func=None):
        origin_outer_shape = self.outer_shape
        self_rs = self.reshape_outer_shape((-1,))
        if pcd_sample_func is None:
            pcd = get_pcd_from_latent_w_voxel(jkey, self_rs, npoints, models)
        else:
            pcd = pcd_sample_func(jkey, self_rs)
        self_rs = self_rs.replace(rel_pcd=pcd - self_rs.pos[...,None,:])
        return self_rs.reshape_outer_shape(origin_outer_shape)

    @property
    def h(self)->jnp.ndarray:
        return jnp.concatenate([einops.rearrange(self.z_flat, '... i j -> ... (i j)'), einops.rearrange(self.dc_centers_tf, '... i j -> ... (i j)')/0.3, self.pos/0.3], -1)

    @property
    def z_flat(self)->jnp.ndarray:
        return einops.rearrange(self.z, '... f d -> ... (f d)')
    
    @property
    def nz(self):
        return self.z.shape[-1]
    
    @property
    def nf(self):
        return self.z.shape[-2]
    
    @property
    def nd(self):
        if self.z is not None:
            return self.z.shape[-3]
        else:
            return self.rel_vtx.shape[-3]

    @property
    def outer_shape(self):
        if self.z is not None and self.rel_vtx is not None:
            assert self.z.shape[:-3] == self.rel_vtx.shape[:-3]
            return self.z.shape[:-3]
        elif self.z is not None:
            return self.z.shape[:-3]
        else:
            return self.rel_vtx.shape[:-3]
    
    @property
    def latent_shape(self):
        return (self.nd, self.nf, self.nz)



def extract_vtx_elements_with_faces(vtxeles, cvxfaces):
    '''
    input
    vtxeles : (... ND, NV, NE)
    cvxfaces : (... ND, NF, 3)
    return : (... ND, NF, 3, NE)
    '''
    return jnp.take_along_axis(vtxeles[...,None,:], cvxfaces[...,None], axis=-3) # (..., ND, NF, fc-3, NE)

def get_normals(cvxvtx, cvxfcs):
    # fcsvtx = jnp.take_along_axis(cvxvtx[...,None,:], cvxfcs[...,None], axis=-3) # (..., ND, NF, fc-3, vt-3)
    fcsvtx = get_fcsvtx(cvxvtx, cvxfcs)
    centers = get_center(cvxvtx, keepdims=True)

    va = fcsvtx[...,2,:] - fcsvtx[...,0,:] # (..., ND, NF, vt-3)
    vb = fcsvtx[...,1,:] - fcsvtx[...,0,:] # (..., ND, NF, vt-3)
    normal = jnp.cross(vb, va) # (..., ND, NF, 3)
    rev_mask = jnp.einsum('...k,...k',fcsvtx[...,0,:]-centers, normal) < 0 # (..., ND, NF)
    normal = jnp.where(rev_mask[...,None], -normal, normal) # toward space
    return tutil.normalize(normal)

def get_valid_vtx_mask(cvxvtx, keepdims=False):
    return jnp.all(jnp.abs(cvxvtx) < 1e3, -1, keepdims=keepdims)

def get_valid_fcs_mask(cvxfcs, keepdims=False):
    if keepdims:
        return cvxfcs[...,0:1]>=0
    else:
        return cvxfcs[...,0]>=0

def get_center(cvxvtx, keepdims=False):
    valild_vtx = jnp.all(jnp.abs(cvxvtx) < 1e3, -1, keepdims=True)
    centers = jnp.sum(valild_vtx * cvxvtx, axis=-2, keepdims=keepdims) / jnp.sum(valild_vtx, axis=-2, keepdims=keepdims).clip(1e-10) # (..., ND, 1, 3)
    return centers

def get_fcsvtx(cvxvtx, cvxfcs):
    cvxfcs = get_face_idx_per_dec(cvxvtx, cvxfcs) # (...,)
    valid_fc_mask = cvxfcs[...,0] >= 0
    fcsvtx = jnp.take_along_axis(cvxvtx[...,None,:], cvxfcs[...,None], axis=-3) # (..., ND, NF, fc-3, vt-3)
    fcsvtx = jnp.where(valid_fc_mask[...,None,None], fcsvtx, 1e6)
    return fcsvtx

def get_areas(cvxvtx, cvxfcs):
    cvxfcs = get_face_idx_per_dec(cvxvtx, cvxfcs) # (...,)
    valid_fc_mask = cvxfcs[...,0] >= 0
    fcsvtx = jnp.take_along_axis(cvxvtx[...,None,:], cvxfcs[...,None], axis=-3) # (..., ND, NF, fc-3, vt-3)

    va = fcsvtx[...,2,:] - fcsvtx[...,0,:] # (..., ND, NF, vt-3)
    vb = fcsvtx[...,1,:] - fcsvtx[...,0,:] # (..., ND, NF, vt-3)
    areas = jnp.linalg.norm(jnp.cross(va, vb), axis=-1)*0.5 # (..., ND, NF)
    areas = jnp.where(valid_fc_mask, areas, 0)

    return areas

def get_center_with_area(cvxvtx, cvxfcs, keepdims=False, area_out=False):
    cvxfcs = get_face_idx_per_dec(cvxvtx, cvxfcs) # (...,)
    valid_fc_mask = cvxfcs[...,0] >= 0
    fcsvtx = jnp.take_along_axis(cvxvtx[...,None,:], cvxfcs[...,None], axis=-3) # (..., ND, NF, fc-3, vt-3)
    fc_center = jnp.mean(fcsvtx, -2) # (..., ND, NF, 3)

    va = fcsvtx[...,2,:] - fcsvtx[...,0,:] # (..., ND, NF, vt-3)
    vb = fcsvtx[...,1,:] - fcsvtx[...,0,:] # (..., ND, NF, vt-3)
    areas = jnp.linalg.norm(jnp.cross(va, vb), axis=-1) # (..., ND, NF)
    areas = jnp.where(valid_fc_mask, areas, 0)
    weights = (areas/(1e-6+jnp.sum(areas, axis=-1,keepdims=True)))
    weights = jnp.where(valid_fc_mask, weights, 0)[...,None]
    center = jnp.sum(fc_center * weights, axis=-2, keepdims=keepdims) # (..., ND, 3)
    if area_out:
        return center, areas
    else:
        return center


def get_face_idx_per_dec(cvxvtx, cvxfcs):
    if cvxvtx.shape[-3] == 1:
        return cvxfcs
    # assert jnp.max(cvxfcs) > cvxvtx.shape[-2]
    valild_vtx = jnp.all(jnp.abs(cvxvtx) < 1e3, -1, keepdims=True)
    fc_idx = jnp.sum(valild_vtx, axis=-2, keepdims=True)
    fc_idx = jnp.cumsum(fc_idx, -3)
    fc_idx = cvxfcs-jnp.concatenate([jnp.zeros_like(fc_idx[...,0:1,:,:]), fc_idx[...,:-1,:,:]], -3)
    return jnp.maximum(fc_idx, -1)

def occ_query(cvxvtx, cvxfcs, query_pnts):
    '''
    inputs
    cvxvtx : (..., ND, NV, 3)
    cvxfcs : (..., ND, NF, 3)
    query_pnts : (..., NQ, 3)

    return 
    occ_res : (..., NQ) true if occupied
    '''
    valid_faces = cvxfcs[...,0:1] >=0
    valid_dec = valid_faces[...,:1,0]
    cvxfcs = get_face_idx_per_dec(cvxvtx, cvxfcs)

    fcsvtx = jnp.take_along_axis(cvxvtx[...,None,:], cvxfcs[...,None], axis=-3) # (..., ND, NF, fc-3, vt-3)
    centers = get_center(cvxvtx, keepdims=True)

    va = fcsvtx[...,2,:] - fcsvtx[...,0,:] # (..., ND, NF, vt-3)
    vb = fcsvtx[...,1,:] - fcsvtx[...,0,:] # (..., ND, NF, vt-3)
    normal = jnp.cross(vb, va) # (..., ND, NF, 3)
    rev_mask = jnp.einsum('...k,...k',fcsvtx[...,0,:]-centers, normal) < 0 # (..., ND, NF)
    normal = jnp.where(rev_mask[...,None], -normal, normal) # toward space

    inside_mask = jnp.einsum('...fqt,...ft->...fq',query_pnts[...,None,None,:,:]-fcsvtx[...,0:1,:], normal)<0 # (..., ND, NF, NQ)
    occ_per_cvx = jnp.all(jnp.logical_or(jnp.logical_not(valid_faces), inside_mask), axis=-2) #(..., ND, NQ)
    occ_res = jnp.any(jnp.where(valid_dec, occ_per_cvx, False), axis=-2)

    return occ_res



def min_dist(cvxvtx, cvxfcs, query_pnts):
    '''
    inputs
    cvxvtx : (..., ND, NV, 3)
    cvxfcs : (..., ND, NF, 3)
    query_pnts : (..., NQ, 3)

    return 
    occ_res : (..., NQ) true if occupied
    '''
    valid_faces = cvxfcs[...,0:1] >=0
    valid_vtx = jnp.all(jnp.abs(cvxvtx)<1e5, axis=-1)
    valid_dec = valid_faces[...,:1,0]
    cvxfcs = get_face_idx_per_dec(cvxvtx, cvxfcs)

    cvxvtx_inf = jnp.where(valid_vtx[...,None], cvxvtx, jnp.inf)
    fcsvtx = jnp.take_along_axis(cvxvtx_inf[...,None,:], cvxfcs[...,None], axis=-3) # (..., ND, NF, fc-3, vt-3)

    fcsvtx_qp = fcsvtx[...,None,:,:,:,:]-query_pnts[...,None,None,None,:]
    # centers = get_center(cvxvtx, keepdims=True)

    min_vec, _, _ = GJK.min_distance_to_simplex(fcsvtx_qp, rec=False)

    return occ_res



def sampling_from_surface(jkey, cvxvtx, cvxfcs, ns, normal_out=False, fc_idx_out=False):
    '''
    inputs
    cvxvtx : (..., ND, NV, 3)
    cvxfcs : (..., ND, NF, 3)
    ns : int

    return
    sample points : (..., )
    '''
    cvxfcs = get_face_idx_per_dec(cvxvtx, cvxfcs)
    valid_fc_mask = cvxfcs[...,0] >= 0
    fcsvtx = jnp.take_along_axis(cvxvtx[...,None,:], cvxfcs[...,None], axis=-3) # (..., ND, NF, fc-3, vt-3)

    va = fcsvtx[...,2,:] - fcsvtx[...,0,:] # (..., ND, NF, vt-3)
    vb = fcsvtx[...,1,:] - fcsvtx[...,0,:] # (..., ND, NF, vt-3)
    areas = jnp.linalg.norm(jnp.cross(va, vb), axis=-1) # (..., ND, NF)
    areas = jnp.where(valid_fc_mask, areas, 0)

    # merge ND NF for area calculation
    spnt = einops.rearrange(fcsvtx[...,0,:], '... i j k -> ... (i j) k')
    va = einops.rearrange(va, '... i j k -> ... (i j) k')
    vb = einops.rearrange(vb, '... i j k -> ... (i j) k')
    areas = einops.rearrange(areas, '... i j -> ... (i j)')

    # alpha beta sample
    ab = jax.random.uniform(jkey, shape=areas.shape[:-1]+(ns,2)) # (..., NS,2)
    ab = jnp.where(jnp.sum(ab,-1, keepdims=True)>=1, 1-ab, ab) # (..., NS,2) # uniform projection to make it inside triangle

    log_sm = jnp.where(areas<=1e-8, -1e5, jnp.log(areas/jnp.sum(areas, axis=-1, keepdims=True))) # (..., NS)

    fidx = jax.random.categorical(jkey, logits=log_sm[...,None,:], axis=-1, shape=log_sm.shape[:-1]+(ns,)) # (..., NS)
    vas = jnp.take_along_axis(va, fidx[...,None], -2)
    vbs = jnp.take_along_axis(vb, fidx[...,None], -2)
    start_pnt = jnp.take_along_axis(spnt, fidx[...,None], -2)

    pnts = vas * ab[...,:1] + vbs * ab[...,1:2] + start_pnt # (..., NS, 3)

    if normal_out:
        normal = jnp.cross(vbs, vas) # (..., NS)
        normal = tutil.normalize(normal)
        # rev_mask = jnp.einsum('...k,...k',start_pnt - get_center(cvxvtx, keepdims=True), normal) < 0 # (..., ND, NF)
        # normal = jnp.where(rev_mask[...,None], -normal, normal) # toward space
        if fc_idx_out:
            return pnts, normal, fidx
        else:
            return pnts, normal
    if fc_idx_out:
        return pnts, fidx
    else:
        return pnts


def sampling_from_surface_convex_dec(jkey, vts, fcs, ns, normal_out=False):

    # pnts, fidx = sampling_from_surface(jkey, vts, fcs, ns=3*ns, normal_out=normal_out, fc_idx_out=True)
    sp_res = sampling_from_surface(jkey, vts, fcs, ns=3*ns, normal_out=normal_out, fc_idx_out=True)
    dec_idx = sp_res[-1]//fcs.shape[-2]
    _, jkey = jax.random.split(jkey)
    qpnts = sp_res[0] + jax.random.normal(jkey, shape=sp_res[0].shape) * 0.0005
    occ_res = occ_query(vts, fcs, qpnts)

    def filter_with_where(x):
        origin_shape = x.shape
        x = x.reshape((-1,)+x.shape[-1:])
        idx_ = jax.vmap(lambda x: jnp.where(jnp.logical_not(x), size=ns))(x)[0]
        idx_ = idx_.reshape(origin_shape[:-1] + (ns,))
        return idx_
    pnt_idx = filter_with_where(occ_res)
    spnts = jnp.take_along_axis(qpnts, pnt_idx[...,None], axis=-2)
    dec_idx_pick = jnp.take_along_axis(dec_idx, pnt_idx, axis=-1)
    if normal_out:
        return spnts, jnp.take_along_axis(sp_res[1], pnt_idx[...,None], axis=-2), dec_idx_pick
    else:
        return spnts, dec_idx_pick

def get_AABB(cvxvtx):
    valid_vtx_mask = get_valid_vtx_mask(cvxvtx, keepdims=True)
    aabb_max = jnp.max(valid_vtx_mask*cvxvtx-1e6*(1-valid_vtx_mask), axis=(-2,-3)) # (..., 3)
    aabb_min = jnp.min(valid_vtx_mask*cvxvtx+1e6*(1-valid_vtx_mask), axis=(-2,-3)) # (..., 3)
    return aabb_min, aabb_max

def sampling_from_AABB(jkey, cvxvtx, ns, margin=0.):
    '''
    inputs
    cvxvtx : (..., ND, NV, 3)
    ns : int

    return
    sample points : (..., NS, 3)
    '''
    valid_vtx_mask = get_valid_vtx_mask(cvxvtx, keepdims=True)
    aabb_max = jnp.max(valid_vtx_mask*cvxvtx-1e6*(1-valid_vtx_mask), axis=(-2,-3))+margin # (..., 3)
    aabb_min = jnp.min(valid_vtx_mask*cvxvtx+1e6*(1-valid_vtx_mask), axis=(-2,-3))-margin # (..., 3)
    pnts = jax.random.uniform(jkey, shape=cvxvtx.shape[:-3]+(ns,3), minval=aabb_min[...,None,:], maxval=aabb_max[...,None,:])

    return pnts
    

def ray_casting(spnt, ray_dir, cvxvtx, cvxfcs, min_depth=None, max_depth=1.0, normalize_dir=True):
    '''
    spnt : (... NQ, 3)
    ray_dir : (... NQ, 3)
    cvxvtx : (... ND NV 3)
    cvxfcs : (... ND NF 3)
    
    return : (... NQ ...)
    '''
    if normalize_dir:
        ray_dir = tutil.normalize(ray_dir)
    cvxfcs = get_face_idx_per_dec(cvxvtx, cvxfcs)
    fcsvtx = jnp.take_along_axis(cvxvtx[...,None,:], cvxfcs[...,None], axis=-3) # (..., ND, NF, fc-3, vt-3)
    valid_cvs_mask = cvxfcs[...,0:1] >= 0 # (... ND NF 1)

    va = fcsvtx[...,2,:] - fcsvtx[...,0,:] # (..., ND, NF, vt-3)
    vb = fcsvtx[...,1,:] - fcsvtx[...,0,:] # (..., ND, NF, vt-3)
    # normal = get_normals(cvxvtx, cvxfcs) # (..., ND, NF, 3)
    normal = jnp.cross(vb, va) # (..., ND, NF, 3)
    rev_mask = jnp.einsum('...k,...k',fcsvtx[...,0,:] - get_center(cvxvtx, keepdims=True), normal) < 0 # (..., ND, NF)
    normal = jnp.where(rev_mask[...,None], -normal, normal) # toward space
    d = jnp.einsum('...i,...i',fcsvtx[...,0,:], normal) # (..., ND, NF)
    
    #bc (..., NQ, ND, NF, 3)
    spnt = spnt[...,None,None,:]
    ray_dir = ray_dir[...,None,None,:]
    normal = normal[...,None,:,:,:]
    d = d[...,None,:,:,None]

    den = jnp.einsum('...i,...i',normal,ray_dir)[...,None]
    perp_mask = jnp.where(jnp.abs(den) > 1e-8, 1, 0) # perp # (... NQ ND NF 1)
    valid_mask = jnp.logical_and(valid_cvs_mask[...,None,:,:,:], perp_mask) # (... NQ ND NF 1)
    temp1 = (d-jnp.einsum('...i,...i',normal,spnt)[...,None])
    temp1 = jnp.where(valid_mask, temp1, 0)
    den = jnp.where(valid_mask,den,1)
    t = temp1/den # (..., NQ, ND, NF, 1) # normal * (spnt+ray_dir*t) = d
    t = jnp.where(valid_mask, t, 0).clip(-1e2,1e2)
    intpnts = spnt + t*ray_dir # (..., NQ, ND, NF, 3)
    vq = intpnts - fcsvtx[...,None,:,:,0,:]
    vq = jnp.where(valid_mask, vq, 0)
    va = va[...,None,:,:,:]
    vb = vb[...,None,:,:,:]

    vab = jnp.stack([va,vb], axis=-1)
    pinv = jnp.linalg.pinv(vab)
    pinv = jnp.where(valid_mask[...,None], pinv, 0)
    alphabeta = jnp.einsum('...ij,...j->...i', pinv, vq)
    valid_mask = jnp.logical_and(valid_mask[...,0], jnp.all(jnp.logical_and(alphabeta>=0, alphabeta<=1), -1))
    valid_mask = jnp.logical_and(jnp.sum(alphabeta, -1) <= 1, valid_mask)[...,None] # (..., NQ, ND, NF 1)
    
    if min_depth is not None:
        t = jnp.where(t < min_depth, 1e6, t)

    depth_key = jnp.where(valid_mask, t, 1e6) # (..., NQ, ND, NF, 1)
    idx = jnp.argmin(depth_key, axis=-2) # (..., NQ, ND, 1)

    depth_pick = jnp.squeeze(jnp.take_along_axis(depth_key, idx[...,None,:], axis=-2), -2)
    depth_pick = jnp.clip(depth_pick, a_max=max_depth)
    
    normal_pick = jnp.squeeze(jnp.take_along_axis(normal, idx[...,None,:], axis=-2), -2)
    normal_pick = tutil.normalize(normal_pick)

    if cvxvtx.shape[-3] != 1:
        didx = jnp.argmin(depth_pick, -2)
        depth_pick = jnp.take_along_axis(depth_pick, didx[...,None,:], axis=-2)
        normal_pick = jnp.take_along_axis(normal_pick, didx[...,None,:], axis=-2)

    depth_pick = jnp.squeeze(depth_pick, axis=-2)
    normal_pick = jnp.squeeze(normal_pick, axis=-2)
    return depth_pick, normal_pick


def o3d_visualizer(cvxvtx, cvxfcs, lines=False):
    import open3d as o3d

    valild_vtx = jnp.all(jnp.abs(cvxvtx) < 1e3, -1)
    valid_faces = cvxfcs[...,0] >=0

    vtx = cvxvtx[np.where(valild_vtx)]
    fcs = cvxfcs[np.where(valid_faces)]

    # mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(vtx), o3d.utility.Vector3iVector(fcs))
    mesh = o3d.geometry.TriangleMesh(vertices=o3d.utility.Vector3dVector(vtx), triangles=o3d.utility.Vector3iVector(fcs))
    mesh.compute_vertex_normals()
    # o3d.visualization.draw_geometries([mesh])
    if lines:
        return o3d.geometry.LineSet.create_from_triangle_mesh(mesh)
    return mesh


def generate_proxy_object(jkey, pair_cvx_object:CvxObjects, depth_scale=0.007, pos_noise_scale=0.020, rot_noise_scale_deg=5.0):
    '''
    input
    pair_cvx_object : (... 2, ND, NV, 3)
    pair_pos : (... 2 3)
    pair quat : (... 2 4)

    return
    (labels_uniform, pos_surface, quat_surface, labels_surface, jkey)
    labels_uniform : (... 1) - current collision results
    pos_surface : (... 2 3) - contact position with noise
    quat surface : (... 2 4) - contact quat with noise
    labels_surface : (... 1) - contact label with noise
    '''
    nd = pair_cvx_object.nd
    if nd==1:
        max_ncd = 1
    elif nd >= 20:
        max_ncd = 200
    else:
        max_ncd = 3
    outer_shape = pair_cvx_object.outer_shape[:-1]
    vtx = pair_cvx_object.vtx_tf
    pair_pos = pair_cvx_object.pos
    _, labels_uniform, min_dir = GJK.GJK_cvx_dec(vtx[...,0,:,:,:], vtx[...,1,:,:,:], None, itr_limit=10, max_ncd=max_ncd, collision_only=False)
    dir_normalized = tutil.normalize(min_dir)
    pen_depth = jax.random.uniform(jkey, shape=outer_shape + (1,), dtype=jnp.float32, minval=0, maxval=depth_scale)
    _, jkey = jax.random.split(jkey)
    posB_add = pair_pos[...,1,:] + min_dir + pen_depth*dir_normalized + jax.random.normal(jkey, shape=outer_shape+(3,)) * pos_noise_scale
    _, jkey = jax.random.split(jkey)
    pos_surface = jnp.stack([pair_pos[...,0,:], posB_add], axis=-2)
    # quatB_add = tutil.qmulti(pair_quat[...,1,:], tutil.qExp(jax.random.normal(jkey,shape=outer_shape+(3,)) * np.pi/180.0*rot_noise_scale_deg))
    # quat_surface = jnp.stack([pair_quat[...,0,:], quatB_add], axis=-2)
    quatB_add = tutil.qExp(jax.random.normal(jkey,shape=outer_shape+(3,)) * np.pi/180.0*rot_noise_scale_deg)
    quat_surface = jnp.stack([quatB_add*0+jnp.array([0,0,0,1.]), quatB_add], axis=-2)
    pair_cvx_object_surface = pair_cvx_object.replace(pos=pos_surface).rotate_rel_vtx(quat_surface)
    # pair_cvx_object = pair_cvx_object.apply_pq_vtx(pos_surface, quat_surface)
    vtx = pair_cvx_object_surface.vtx_tf
    _, jkey = jax.random.split(jkey)
    _, labels_surface, _ = GJK.GJK_cvx_dec(vtx[...,0,:,:,:], vtx[...,1,:,:,:], None, 
                                            itr_limit=10, max_ncd=max_ncd, collision_only=False)
    
    # assert jnp.all(jnp.abs(pair_cvx_object_surface.pos) < 10000)
    
    return labels_uniform, pair_cvx_object_surface, labels_surface, jkey


def match_outer_shape(inputs, x):
    assert x.ndim >= inputs.ndim
    inputs_ = inputs
    for _ in range(x.ndim - inputs.ndim):
        inputs_ = jnp.expand_dims(inputs_, axis=-1)
    return inputs_

def gen_col_dataset(jkey, nb, cvxvtx_list, cvxcfs_list, scale_min=0.4, scale_max=1.8, priority_func=None, visualize=False)->Tuple[CvxObjects, jnp.ndarray, jnp.ndarray]:
    '''
    priority_func - should recieve cvxvtx (... 2 ND NV 3), label (... 1)
    '''
    if priority_func is not None:
        gnb = int(nb*1.5)
    else:
        gnb = nb
    cvx_obj_idx = jax.random.randint(jkey, shape=(gnb, 2), minval=0, maxval=cvxvtx_list.shape[0])
    _, jkey = jax.random.split(jkey)
    cvx_obj = cvxvtx_list[cvx_obj_idx]
    # mix symetry
    rmask = jax.random.uniform(jkey, cvx_obj[...,:1,:1,:1].shape) > 0.5 # (... 2 1 1 1)
    _,jkey = jax.random.split(jkey)
    cvx_obj = jnp.where(rmask, cvx_obj, cvx_obj.at[...,0].set(-cvx_obj[...,0]))
    rmask = jax.random.uniform(jkey, cvx_obj[...,:1,:1,:1].shape) > 0.5
    _,jkey = jax.random.split(jkey)
    cvx_obj = jnp.where(rmask, cvx_obj, cvx_obj.at[...,1].set(-cvx_obj[...,1]))
    rmask = jax.random.uniform(jkey, cvx_obj[...,:1,:1,:1].shape) > 0.5
    _,jkey = jax.random.split(jkey)
    cvx_obj = jnp.where(rmask, cvx_obj, cvx_obj.at[...,2].set(-cvx_obj[...,2]))
    
    cvxcfs = cvxcfs_list[cvx_obj_idx]
    
    # rand_scale = jax.random.uniform(jkey, shape=(gnb,2,1), minval=0.65, maxval=1.7)
    # rand_scale = jax.random.uniform(jkey, shape=(gnb,2,1), minval=0.7, maxval=1.6)
    rand_scale = jax.random.uniform(jkey, shape=(gnb,2,1), minval=scale_min, maxval=scale_max)
    _, jkey = jax.random.split(jkey)
    rand_scale *= jax.random.uniform(jkey, shape=(gnb,2,3), minval=0.85, maxval=1.2)
    _, jkey = jax.random.split(jkey)
    pos1 = jax.random.uniform(jkey, shape=(gnb//2,2,3), minval=-0.5, maxval=0.5)
    _, jkey = jax.random.split(jkey)
    pos2 = jax.random.uniform(jkey, shape=(gnb-gnb//2,2,3), minval=-0.1, maxval=0.1)
    pos = jnp.concatenate([pos1, pos2], axis=0)
    _, jkey = jax.random.split(jkey)
    quat = tutil.qrand((gnb,2),jkey)
    _, jkey = jax.random.split(jkey)

    cvx_obj_scale = cvx_obj*rand_scale[...,None,None,:]
    cvx_obj_pair = CvxObjects().init_vtx(cvx_obj_scale, cvxcfs).replace(pos=pos).rotate_rel_vtx(quat)

    # assert jnp.all(jnp.abs(cvx_obj_pair.pos) < 10000)

    labels_uniform, cvx_obj_pair_surface, labels_surface, jkey = \
                        generate_proxy_object(jkey, cvx_obj_pair)

    # assert jnp.all(jnp.abs(cvx_obj_pair_surface.pos) < 10000)

    # top-K select
    if priority_func is not None:
        # # priority_func - should recieve cvxvtx (... 2 ND NV 3), label (... 1)
        cvx_pairs_ext = jax.tree_map(lambda *x: jnp.r_[x[0],x[1]], cvx_obj_pair, cvx_obj_pair_surface)
        labels = jnp.r_[labels_uniform, labels_surface]
        batch_loss = priority_func(cvx_pairs_ext, labels)
        idx = jnp.argsort(-batch_loss)[:nb]
        cvx_pairs_res, labels = jax.tree_util.tree_map(lambda x: x[idx], (cvx_pairs_ext, labels))
    else:
        # random select
        rand_mask = jax.random.randint(jkey, shape=(nb,1,1), minval=0, maxval=20)<19
        _, jkey = jax.random.split(jkey)
        labels = jnp.where(rand_mask[...,0,0], labels_surface, labels_uniform)
        cvx_pairs_res:CvxObjects = jax.tree_map(lambda x, y: jnp.where(match_outer_shape(rand_mask, x), x, y), cvx_obj_pair_surface, cvx_obj_pair)

    if visualize:
        for i in range(cvx_pairs_res.outer_shape[0]):
            import open3d as o3d
            print(labels[i])
            mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
            visobj1 = o3d_visualizer(cvx_pairs_res.vtx_tf[i,0], cvx_pairs_res.fc[i,0], lines=True)
            if labels[i] < 0:
                visobj1.paint_uniform_color(np.array([0.9, 0.1, 0.2]))
            else:
                visobj1.paint_uniform_color(np.array([0.2, 0.1, 0.9]))
            visobj2 = o3d_visualizer(cvx_pairs_res.vtx_tf[i,1], cvx_pairs_res.fc[i,1], lines=True)
            visobj2.paint_uniform_color(np.array([0.1, 0.9, 0.2]))

            o3d.visualization.draw_geometries([visobj1, visobj2, mesh_frame])

    # assert jnp.all(jnp.abs(cvx_pairs_res.pos) < 10000)

    return cvx_pairs_res, labels, jkey


def gen_occ_dataset(jkey, ns, objects:CvxObjects, noise_scale=0.010, visualize=False)->Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    # sampling from surface
    cvxvtx, cvxfcs = objects.vtx_tf, objects.fc
    outer_shape = objects.outer_shape
    qpnts = sampling_from_surface(jkey, cvxvtx, cvxfcs, (ns//20)*17)
    _, jkey = jax.random.split(jkey)
    qpnts += jax.random.normal(jkey, qpnts.shape) * noise_scale

    # sampling from uniform
    # qpnts_uniform = sampling_from_AABB(jkey, cvxvtx, ns//5, margin=0.50)
    qpnts_uniform = objects.pos[...,None,:] + jax.random.uniform(jkey, shape=outer_shape+((ns//20)*3, 3), minval=-1.0,maxval=1.0)
    qpnts = jnp.concatenate([qpnts, qpnts_uniform], axis=-2)

    occ_res = occ_query(cvxvtx, cvxfcs, qpnts)
    occ_res = jnp.logical_not(occ_res).astype(jnp.float32)

    if visualize:
        import open3d as o3d
        mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
        for i in range(objects.outer_shape[0]):
            for j in range(objects.outer_shape[1]):
                if objects.vtx_tf.ndim == 5:
                    occ_res_vis, qpnts_vis, objects_vis = jax.tree_map(lambda x : x[i,j], (occ_res, qpnts, objects))
                else:
                    if j==1:
                        break
                    occ_res_vis, qpnts_vis, objects_vis = jax.tree_map(lambda x : x[i], (occ_res, qpnts, objects))
                o3d_line = o3d_visualizer(objects_vis.vtx_tf, objects_vis.fc, lines=True)
                ppnts = qpnts_vis[np.where(occ_res_vis==1)]
                npnts = qpnts_vis[np.where(occ_res_vis==0)]
                pcd1 = o3d.geometry.PointCloud()
                pcd1.points = o3d.utility.Vector3dVector(ppnts)
                pcd1.colors = o3d.utility.Vector3dVector(np.ones_like(ppnts) * np.array([1,0,0]))
                pcd2 = o3d.geometry.PointCloud()
                pcd2.points = o3d.utility.Vector3dVector(npnts)
                pcd2.colors = o3d.utility.Vector3dVector(np.ones_like(npnts) * np.array([0,0,1]))

                o3d.visualization.draw_geometries([o3d_line, pcd1, pcd2, mesh_frame])

    return qpnts, occ_res, jkey

def gen_pln_dataset(jkey, ns, objects:CvxObjects, noise_scale=0.015):
    '''
    objects : 
    '''

    ns_ = 2*ns

    vtx_dcflat = einops.rearrange(objects.vtx_tf, '... i j k -> ... (i j) k')[...,None,:,:] # (... 1 NDV 3)
    vm_dcflat = einops.rearrange(objects.vtx_valid_mask, '... i j k -> ... (i j) k')[...,None,:,0] # (... 1 NDV 1)

    # origin min pln
    pln_v_n = jax.random.normal(jkey, shape=objects.outer_shape + (ns_, 3)) # (... NS 3)
    _, jkey = jax.random.split(jkey)
    pln_v_n = tutil.normalize(pln_v_n)
    idx = jnp.argmin(jnp.where(vm_dcflat, jnp.einsum('...i,...i', vtx_dcflat, pln_v_n[...,None,:]), 1e6), -1) # (... NS)
    pln_v_p = jnp.squeeze(jnp.take_along_axis(vtx_dcflat, idx[...,None,None], axis=-2), axis=-2)

    # test
    # pln_v_p += jax.random.normal(jkey, shape=objects.outer_shape + (ns_, 3)) * noise_scale
    # col_label = jnp.all(jnp.where(vm_dcflat, jnp.einsum('...i,...i', vtx_dcflat-pln_v_p[...,None,:], pln_v_n[...,None,:]), 1e6) >= 0, axis=-1) # (... NQ)

    # face pln
    # areas = get_areas(objects.rel_vtx, objects.fc)[...,None,:] # (... ND, 1, NF)
    # ridx = jax.random.categorical(jkey, jnp.log(areas.clip(1e-10)), shape=areas.shape[:-2] + (ns_,)) # (... ND)
    # _, jkey = jax.random.split(jkey)
    # pick_triangles = jnp.take_along_axis(objects.fcvtx, ridx[...,None,None], axis=-3) # (... ND 3 3)
    # pick_normals = jnp.take_along_axis(objects.normal, ridx[...,None],axis=-2) # (... ND NQ 3)
    # assert objects.nd == 1
    # pick_triangles = jnp.squeeze(pick_triangles, axis=-4) # assume nd == 1
    # pln_fc_n = jnp.squeeze(-pick_normals, axis=-3)
    # pln_fc_p = jnp.mean(pick_triangles, -2)

    # face pln ND!=0
    areas = get_areas(objects.rel_vtx, objects.fc)
    areas = einops.rearrange(areas, '... i j -> ... (i j)')[..., None, :] # (... 1, ND*NF)
    areas = areas*1e5
    ridx = jax.random.categorical(jkey, jnp.log(areas), shape=areas.shape[:-2] + (ns_,)) # (... NQ)
    _, jkey = jax.random.split(jkey)
    fcvtx_flat = einops.rearrange(objects.fcvtx, '... i j k l -> ... (i j) k l')
    normal_flat = einops.rearrange(objects.normal, '... i j k -> ... (i j) k')
    pick_triangles = jnp.take_along_axis(fcvtx_flat, ridx[...,None,None], axis=-3) # (... NQ 3 3)
    pick_normals = jnp.take_along_axis(normal_flat, ridx[...,None],axis=-2) # (... NQ 3)
    # assert objects.nd == 1
    # pick_triangles = jnp.squeeze(pick_triangles, axis=-4) # assume nd == 1
    # pln_fc_n = jnp.squeeze(-pick_normals, axis=-3)
    pln_fc_p = jnp.mean(pick_triangles, -2) # pick center point
    pln_fc_n = -pick_normals

    mask = jax.random.uniform(jkey, shape=pln_fc_n[...,:1].shape) < 0.3
    pln_n = jnp.where(mask, pln_fc_n, pln_v_n)
    _, jkey = jax.random.split(jkey)
    pln_p = jnp.where(mask, pln_fc_p, pln_v_p) # (... NQ 3)
    pln_p += jax.random.normal(jkey, shape=objects.outer_shape + (ns_, 3)) * noise_scale
    _, jkey = jax.random.split(jkey)
    pln_n += jax.random.normal(jkey, shape=objects.outer_shape + (ns_, 3)) * 0.05
    _, jkey = jax.random.split(jkey)
    pln_n = tutil.normalize(pln_n)

    # labeling
    col_label = jnp.all(jnp.where(vm_dcflat, jnp.einsum('...i,...i', vtx_dcflat-pln_p[...,None,:], pln_n[...,None,:]), 1e6) > 0, axis=-1) # (... NQ) # False : collision

    # balancing
    prob = jnp.where(col_label, jnp.arange(col_label.shape[-1]) * jnp.sum(col_label), jnp.arange(col_label.shape[-1]) * jnp.sum(1-col_label))
    idx = jnp.argsort(prob, -1)[...,:ns]
    pln_p = jnp.take_along_axis(pln_p, idx[...,None], -2)
    pln_n = jnp.take_along_axis(pln_n, idx[...,None], -2)
    col_label = jnp.take_along_axis(col_label, idx, -1)

    # # visualization
    # pos = objects.pos[...,None,None,:]
    # vtx_pnt = einops.rearrange(objects.vtx_tf, '... i j k -> ... 1 (i j) k')
    # vtx_msk = einops.rearrange(objects.vtx_valid_mask, '... i j k -> ... 1 (i j) k')
    # pln_p_ext = pln_p[...,None,:]
    # pln_n_ext = pln_n[...,None,:]
    # pln_n_p1 = jnp.cross(jnp.array([1,0,0]), pln_n_ext)
    # pln_n_p1 = tutil.normalize(pln_n_p1)
    # pln_n_p2 = pln_n_ext
    # p1 = jax.tree_map(lambda x : jnp.einsum('...j,...j',x, pln_n_p1), (vtx_pnt, pln_p_ext, pln_n_ext, pos))
    # p2 = jax.tree_map(lambda x : jnp.einsum('...j,...j',x, pln_n_p2), (vtx_pnt, pln_p_ext, pln_n_ext, pos))
    # vtx_pnt_pj, pln_p_pj, pln_n_pj, pos_pj = jax.tree_map(lambda x1, x2: jnp.stack([x1, x2], axis=-1), p1, p2)
    # pln_np_pj = jnp.c_[pln_n_pj[...,1:], -pln_n_pj[...,0:1]]
    # pln_line1 = pln_p_pj + 0.1*pln_np_pj
    # pln_line2 = pln_p_pj - 0.1*pln_np_pj
    # pln_line = jnp.concatenate([pln_line1, pln_line2], axis=-2)
    # import matplotlib.pyplot as plt
    # plt.figure()
    # for i in range(5):
    #     for j in range(5):
    #         mask_ = vtx_msk[i,0,j,:,0]
    #         cur_vtx_ = vtx_pnt_pj[i,0,j,:]
    #         cur_vtx_ = cur_vtx_[np.where(mask_)]
    #         plt.subplot(5,5,5*i+j+1)
    #         plt.scatter(cur_vtx_[:,0], cur_vtx_[:,1])
    #         plt.scatter(pos_pj[i,0,j,0,0], pos_pj[i,0,j,0,1])
    #         plt.plot(pln_line[i,0,j,:,0], pln_line[i,0,j,:,1])
    #         plt.gca().set_title(str(i)+str(j)+'_'+str(col_label[i,0,j]))
    # plt.show()
    # # visualization

    return pln_p, pln_n, col_label, jkey

def gen_ray_dataset(jkey, ns:int, objects:CvxObjects):
    ns_ = ns*2
    objects_ = objects.translate(-objects.pos)
    surface_pnts = sampling_from_surface(jkey, objects_.vtx_tf, objects_.fc, ns_) # (... NS 3)
    _, jkey = jax.random.split(jkey)
    rpnts = surface_pnts + jax.random.normal(jkey, surface_pnts.shape) * 0.020
    _, jkey = jax.random.split(jkey)
    rdir = jax.random.normal(jkey, surface_pnts.shape)
    _, jkey = jax.random.split(jkey)
    rdir = tutil.normalize(rdir)
    jk1, jk2, jkey = jax.random.split(jkey, 3)
    rpnts = jnp.where(jax.random.uniform(jk1, shape=objects_.outer_shape + (1,1))<0.95, rpnts, jax.random.uniform(jk2, shape=surface_pnts.shape, minval=-1.5, maxval=1.5))

    rpnts = rpnts-jnp.einsum('...i,...i',rpnts,rdir)[...,None]*rdir
    depth, normals = ray_casting(rpnts, rdir, objects_.vtx_tf, objects_.fc, min_depth=-10, max_depth=10, normalize_dir=False)

    # balancing
    sign = jnp.abs(depth)[...,0] < 9
    prob = jnp.where(sign, jnp.arange(sign.shape[-1]) * jnp.sum(sign), jnp.arange(sign.shape[-1]) * jnp.sum(1-sign))
    idx = jnp.argsort(prob, -1)[...,:ns]
    rpnts = jnp.take_along_axis(rpnts, idx[...,None], -2)
    rdir = jnp.take_along_axis(rdir, idx[...,None], -2)
    depth = jnp.take_along_axis(depth, idx[...,None], -2)
    normals = jnp.take_along_axis(normals, idx[...,None], -2)
    rpnts += objects.pos[...,None,:]

    return rpnts, rdir, depth.clip(-10.0,10.0), normals, jkey


# @jax.jit
def occ_ds_given_spnts(jkey, vtx, fcs, spnts, multiplier=2):
    np = spnts.shape[-2]
    aabb = jnp.max(spnts, axis=-2, keepdims=True) - jnp.min(spnts, axis=-2, keepdims=True)
    obj_len = jnp.mean(aabb, axis=-1, keepdims=True)
    spnts1 = spnts[...,:np//10,:]
    spnts2 = spnts[...,np//10:,:]
    spnts2, spnts3 = spnts2[...,:spnts2.shape[-2]//2,:], spnts2[...,spnts2.shape[-2]//2:,:]
    qpnts1 = spnts1[None] + jax.random.normal(jkey, shape=(multiplier,)+spnts1.shape)*0.1
    _, jkey = jax.random.split(jkey)
    qpnts2 = spnts2[None] + jax.random.normal(jkey, shape=(multiplier,)+spnts2.shape)*obj_len*0.07
    _, jkey = jax.random.split(jkey)
    qpnts3 = spnts3[None] + jax.random.normal(jkey, shape=(multiplier,)+spnts3.shape)*obj_len*0.005
    _, jkey = jax.random.split(jkey)
    qpnts = jnp.concatenate([qpnts1, qpnts2, qpnts3], axis=-2)
    qpnts = einops.rearrange(qpnts, 'i ... k j -> ... (i k) j')
    occ_res = occ_query(vtx, fcs, qpnts)
    return qpnts, occ_res


def hull_to_mesh(hull):
    '''
    input
        hull
    
    return
        vertices (NV, ND)
        vertice_simplices (NF, 3)
        vertices_normal (NV, ND)
        triangle_normal (NF, ND)
    '''
    from scipy.spatial import ConvexHull
    idx_pt_vc = -np.ones(hull.points.shape[0], dtype=np.int32)
    idx_pt_vc[hull.vertices] = np.arange(hull.vertices.shape[0])

    vertices = hull.points[hull.vertices]
    if vertices.shape[-1]==2:
        return (vertices, )

    vertices = hull.points[hull.vertices].astype(np.float32)
    vertice_simplices = idx_pt_vc[hull.simplices]

    # index matching & calculate normals
    # face indice should be matched with normal direction
    center = np.mean(vertices, axis=0)
    vertices_normal = np.zeros_like(vertices)
    triangle_normal = np.zeros(vertice_simplices.shape, dtype=np.float32)
    for i in range(vertice_simplices.shape[0]):
        v012 = vertices[vertice_simplices[i]]
        vc0 = v012[0] - center
        face_normal = np.cross(v012[1] - v012[0], v012[2] - v012[0])
        area = np.linalg.norm(face_normal)
        face_normal = face_normal / np.linalg.norm(face_normal)
        if np.dot(face_normal, vc0) < 0:
            vertice_simplices[i] = np.array([vertice_simplices[i][0], vertice_simplices[i][2], vertice_simplices[i][1]])
            face_normal = -face_normal
        triangle_normal[i] = face_normal
        vertices_normal[vertice_simplices[i]] += face_normal*area
    vertices_normal /= np.linalg.norm(vertices_normal, axis=-1, keepdims=True)

    return vertices, vertice_simplices, vertices_normal, triangle_normal

    
def generate_convex_shape(rng:np.random.Generator, nvertices:int=5, dim:int=3):
    '''
    generate convex hull to mesh

    return
        vertices (NV, ND)
        vertice_simplices (NF, 3)
        vertices_normal (NV, ND)
        triangle_normal (NF, ND) 
    '''
    from scipy.spatial import ConvexHull
    points = rng.uniform(-1.,1.,size=(nvertices, dim))
    hull = ConvexHull(points, incremental=True)

    while len(hull.vertices) < nvertices:
        hull.add_points(rng.uniform(-1.,1.,size=(1, dim)))
    
    return hull_to_mesh(hull)

def tetrahedron_volume(tets:jnp.ndarray):
    a,b,c,d = jnp.split(tets, 4, -2)
    return (jnp.abs(jnp.einsum('...j,...j', a-d, jnp.cross(b-d, c-d))) / 6.)[...,0]

def convex_hull_volume_Delaunay(pts):
    from scipy.spatial import Delaunay
    dt = Delaunay(pts)
    tets = dt.points[dt.simplices]
    return np.sum(tetrahedron_volume(tets))

def get_volumetric_centroid(cvxvtx, cvxfcs):
    # assert cvxvtx.shape[-3] == 1
    fcsvtx = get_fcsvtx(cvxvtx, cvxfcs)
    interial = get_center(cvxvtx, keepdims=True)
    tets = jnp.concatenate([fcsvtx, fcsvtx[...,0:1,:]*0 + interial[...,None,:]], -2)
    volumes = tetrahedron_volume(tets)
    entire_volume = jnp.sum(volumes, axis=(-1,-2), keepdims=True)

    ctrd = jnp.mean(tets, -2)
    # cent = jnp.sum(ctrd * volumes[...,None], -2) / entire_volume[...,None].clip(1e-10)
    cent = jnp.squeeze(jnp.sum(ctrd * volumes[...,None], axis=(-2, -3), keepdims=True), -2) / entire_volume.clip(1e-10)

    cent = jnp.squeeze(cent, axis=-2)

    return cent


def get_volumetric_centroid_dc(cvxvtx, cvxfcs):
    fcsvtx = get_fcsvtx(cvxvtx, cvxfcs)
    interial = get_center(cvxvtx, keepdims=True)
    tets = jnp.concatenate([fcsvtx, fcsvtx[...,0:1,:]*0 + interial[...,None,:]], -2)
    volumes = tetrahedron_volume(tets)
    entire_volume = jnp.sum(volumes, axis=-1, keepdims=True)

    ctrd = jnp.mean(tets, -2)
    # cent = jnp.sum(ctrd * volumes[...,None], -2) / entire_volume[...,None].clip(1e-10)
    cent = jnp.sum(ctrd * volumes[...,None], axis=-2) / entire_volume.clip(1e-10)

    # cent = jnp.squeeze(cent, axis=-2)

    return cent


def get_volumes(cvxvtx, cvxfcs, method=1):
    if method == 1:
        fcsvtx = get_fcsvtx(cvxvtx, cvxfcs)
        interial = get_center(cvxvtx, keepdims=True)
        tets = jnp.concatenate([fcsvtx, fcsvtx[...,0:1,:]*0 + interial[...,None,:]], -2)
        volumes = tetrahedron_volume(tets)
        entire_volume = jnp.sum(volumes, -1)

    elif method == 2:
        fc_normals = get_normals(cvxvtx, cvxfcs)
        fcsvtx = get_fcsvtx(cvxvtx, cvxfcs)
        fcsvtx_proj = fcsvtx.at[...,-1].set(0)
        prj_areas = jnp.linalg.norm(jnp.cross(fcsvtx_proj[...,2,:] - fcsvtx_proj[...,0,:], fcsvtx_proj[...,1,:] - fcsvtx_proj[...,0,:]), axis=-1)*0.5
        pil_vol = prj_areas * jnp.mean(fcsvtx[...,-1], axis=-1) * jnp.sign(fc_normals[...,-1])
        entire_volume = jnp.sum(pil_vol, -1)
    else:
        raise ValueError
    
    return entire_volume

def create_box(ext, nd, max_vertices)->CvxObjects:

    if max_vertices==16:
        max_faces = 28
    elif max_vertices==32:
        max_faces = 60
    else:
        max_faces = max_vertices*2-4

    vertices = jnp.array([
        [-1.,-1,-1], [-1,1,-1], [1,-1,-1], [1,1,-1], #0-3
               [-1,-1,1], [-1,1,1], [1,-1,1], [1,1,1], #4-7
               [1,0,0], [-1,0,0], [0,1,0], [0,-1,0], [0,0,1], [0,0,-1] #8-13
               ])
    vertices = vertices*ext/2.0

    rec1 = [1,0,2,3,13]
    rec2 = [5,4,6,7,12]
    rec3 = [0,1,5,4,9]
    rec4 = [2,3,7,6,8]
    rec5 = [0,2,6,4,11]
    rec6 = [1,3,7,5,10]

    recs = jnp.array([rec1, rec2, rec3, rec4, rec5, rec6])
    idx = jnp.array([[0,1,4], [1,2,4], [2,3,4], [3,0,4]])
    fcs = recs[:,idx]
    fcs = fcs.reshape(-1,3)

    vtx_pd = (1000000*jnp.ones((nd, max_vertices,3), dtype=jnp.float32)).at[0,:vertices.shape[0]].set(vertices)
    fcs_pd = (-1*jnp.ones((nd, max_faces,3), dtype=jnp.int32)).at[0,:fcs.shape[0]].set(fcs)

    return CvxObjects().init_vtx(vtx_pd, fcs_pd)

def create_shelf(
        num_dec: int, 
        num_verts: int,
        height = 0.15, 
        width = 0.3, 
        depth = 0.2, 
        thinkness = 0.03, 
        include_bottom = True
) -> CvxObjects:
    '''
    height - z direction
    width - y direction
    depth - x direction
    bottom aligned with z=0
    inner volume is aligned with height x width x depth
    '''
    box_obj1 = create_box(jnp.array([depth,width+2*thinkness,thinkness]), num_dec, num_verts)
    box_obj1 = box_obj1.apply_pq_vtx(jnp.array([0,0,height+thinkness*0.5]), jnp.array([0,0,0,1]))
    box_obj2 = create_box(jnp.array([depth,thinkness,height]), num_dec, num_verts)
    box_obj2 = box_obj2.apply_pq_vtx(jnp.array([0,0.5*(width+thinkness),0.5*height]), jnp.array([0,0,0,1]))
    box_obj3 = create_box(jnp.array([depth,thinkness,height]), num_dec, num_verts)
    box_obj3 = box_obj3.apply_pq_vtx(jnp.array([0,-0.5*(width+thinkness),0.5*height]), jnp.array([0,0,0,1]))
    if include_bottom:
        box_obj4 = create_box(jnp.array([depth,width+2*thinkness,thinkness]), num_dec, num_verts)
        box_obj4 = box_obj4.apply_pq_vtx(jnp.array([0,0,-thinkness*0.5]), jnp.array([0,0,0,1]))
        shelf_obj:CvxObjects = jax.tree_map(lambda *x: jnp.stack(x,0), box_obj1, box_obj2, box_obj3, box_obj4)
    else:
        shelf_obj:CvxObjects = jax.tree_map(lambda *x: jnp.stack(x,0), box_obj1, box_obj2, box_obj3)

    shelf_obj = shelf_obj.replace(color=jnp.array([0.0,0.5,1.0]))
    shelf_obj = shelf_obj.broadcast(shelf_obj.outer_shape)
    return shelf_obj


def create_shelf_dc(
        num_dec: int, 
        num_verts: int,
        height = 0.15, 
        width = 0.3, 
        depth = 0.2, 
        thinkness = 0.03, 
        include_bottom = True
) -> CvxObjects:
    '''
    height - z direction
    width - y direction
    depth - x direction
    bottom aligned with z=0
    inner volume is aligned with height x width x depth
    '''
    box_obj1 = create_box(jnp.array([depth,width+2*thinkness,thinkness]), 1, num_verts)
    box_obj1 = box_obj1.apply_pq_vtx(jnp.array([0,0,height+thinkness*0.5]), jnp.array([0,0,0,1]))
    box_obj2 = create_box(jnp.array([depth,thinkness,height]), 1, num_verts)
    box_obj2 = box_obj2.apply_pq_vtx(jnp.array([0,0.5*(width+thinkness),0.5*height]), jnp.array([0,0,0,1]))
    box_obj3 = create_box(jnp.array([depth,thinkness,height]), 1, num_verts)
    box_obj3 = box_obj3.apply_pq_vtx(jnp.array([0,-0.5*(width+thinkness),0.5*height]), jnp.array([0,0,0,1]))
    gap = jnp.sum(box_obj1.vtx_valid_mask)
    if include_bottom:
        box_obj4 = create_box(jnp.array([depth,width+2*thinkness,thinkness]), 1, num_verts)
        box_obj4 = box_obj4.apply_pq_vtx(jnp.array([0,0,-thinkness*0.5]), jnp.array([0,0,0,1]))
        boxes = [box_obj1, box_obj2, box_obj3, box_obj4]
    else:
        boxes = [box_obj1, box_obj2, box_obj3]
    concat_vtx = jnp.concatenate([bx.vtx_tf for bx in boxes], 0)
    fcs = [bx.fc for bx in boxes]
    fcs = [jnp.where(fcs[i]>=0, fcs[i]+gap*i, -1) for i in range(len(fcs))]
    concat_fc = jnp.concatenate(fcs, 0)
    concat_vtx = (1e6*jnp.ones((num_dec, num_verts, 3))).at[:len(fcs)].set(concat_vtx)
    concat_fc = (-1*jnp.ones((num_dec, 2*num_verts-4, 3), dtype=jnp.int32)).at[:len(fcs)].set(concat_fc)
    shelf_obj = CvxObjects().init_vtx(concat_vtx, concat_fc)

    shelf_obj = shelf_obj.replace(color=jnp.array([0.0,0.5,1.0]))
    shelf_obj = shelf_obj.broadcast(shelf_obj.outer_shape)
    return shelf_obj


def create_tray(
        num_dec: int, 
        num_verts: int,
        height=0.08, 
        width=0.12, 
        thickness=0.01, 
        distance=0.055, 
        include_bottom=False
) -> CvxObjects:
    '''
    height - z direction of tray wall
    width - y direction
    thickness - tray wall thickness
    distance - distance to each wall from origin
    bottom aligned with z=0
    '''
    box_obj1 = create_box(jnp.array([width, thickness, height]), num_dec, num_verts)
    box_obj1 = box_obj1.apply_pq_vtx(jnp.array([0, distance, 0.5*height]), jnp.array([0,0,0,1]))
    box_obj2 = create_box(jnp.array([width, thickness, height]), num_dec, num_verts)
    box_obj2 = box_obj2.apply_pq_vtx(jnp.array([0, -distance, 0.5*height]), jnp.array([0,0,0,1]))
    box_obj3 = create_box(jnp.array([thickness, width, height]), num_dec, num_verts)
    box_obj3 = box_obj3.apply_pq_vtx(jnp.array([distance, 0, 0.5*height]), jnp.array([0,0,0,1]))
    box_obj4 = create_box(jnp.array([thickness, width, height]), num_dec, num_verts)
    box_obj4 = box_obj4.apply_pq_vtx(jnp.array([-distance, 0, 0.5*height]), jnp.array([0,0,0,1]))
    if include_bottom:
        box_obj5 = create_box(jnp.array([distance*2,distance*2, thickness]), num_dec, num_verts)
        box_obj5 = box_obj5.apply_pq_vtx(jnp.array([0, 0, -thickness]), jnp.array([0,0,0,1]))
        shelf_obj:CvxObjects = jax.tree_map(lambda *x: jnp.stack(x,0), box_obj1, box_obj2, box_obj3, box_obj4, box_obj5)
    else:
        shelf_obj:CvxObjects = jax.tree_map(lambda *x: jnp.stack(x,0), box_obj1, box_obj2, box_obj3, box_obj4)

    shelf_obj = shelf_obj.replace(color=jnp.array([0.,0.5,1.]))
    shelf_obj = shelf_obj.broadcast(shelf_obj.outer_shape)
    return shelf_obj


import optax
def get_pcd_from_latent_w_voxel(jkey, query_objects:LatentObjects, num_points:int, models, visualize=False):
    '''
    query_objects: outer_shape - (nb, ...) or (...)
    num_points: number of output points
    
    return surface points (nb, num_points, 3)
    '''

    # preprocessing size
    if len(query_objects.outer_shape) == 0:
        query_objects = jax.tree_map(lambda x: x[None], query_objects)
    
    # define hyper parameters
    initial_density = 100
    initial_sample_num_pnts = initial_density**3
    intermediate_sample_num_pnts = 20000
    initial_bound_half_len = 0.25
    occ_boundary_value = -0.5
    occ_logit_threshold_for_surface = 8.0
    coarse_to_fine_itr_no = 2
    occ_logit_threshold_for_surface_l2 = [0.4, 0.1]
    assert coarse_to_fine_itr_no == len(occ_logit_threshold_for_surface_l2)

    nb =query_objects.outer_shape[0]

    dec = partial(models.apply, 'occ_predictor')

    # generate initial query points from grid
    x = np.linspace(-initial_bound_half_len, initial_bound_half_len, initial_density+1)
    y = np.linspace(-initial_bound_half_len, initial_bound_half_len, initial_density+1)
    z = np.linspace(-initial_bound_half_len, initial_bound_half_len, initial_density+1)
    xv, yv, zv = np.meshgrid(x, y, z)
    grid = np.stack([xv, yv, zv]).astype(np.float32).reshape(3, -1).transpose()[None]
    grid = jnp.array(grid)
    query_points_l1 = grid + query_objects.pos[...,None,:]

    # evaluate and gather valid points for level 1
    _, jkey = jax.random.split(jkey)
    occ_res = dec(query_objects, query_points_l1, jkey)
    within_mask_l1 = jnp.abs(occ_res-occ_boundary_value)<occ_logit_threshold_for_surface
    if visualize:
        print(jnp.sum(within_mask_l1))
    _, jkey = jax.random.split(jkey)
    valid_pnts_l1 = jax.vmap(lambda jk, a, p: jax.random.choice(jk, a, shape=(initial_sample_num_pnts,), p=p))(jax.random.split(jkey, nb), query_points_l1, within_mask_l1.astype(jnp.float32))
    _, jkey = jax.random.split(jkey)

    # level 2 querying
    bound_half_len_l2 = initial_bound_half_len/initial_density*2
    valid_pnts = valid_pnts_l1
    for i in range(coarse_to_fine_itr_no):
        query_points_l2 = jax.random.uniform(jkey, shape=query_objects.outer_shape + (initial_sample_num_pnts,3), minval=-bound_half_len_l2, maxval=bound_half_len_l2)
        _, jkey = jax.random.split(jkey)
        query_points_l2 = valid_pnts + query_points_l2
        occ_res_l2 = dec(query_objects, query_points_l2, jkey)
        within_mask_l2 = jnp.abs(occ_res_l2-occ_boundary_value)<occ_logit_threshold_for_surface_l2[i]
        if visualize:
            print(jnp.sum(within_mask_l2))
        valid_pnts_l2 = jax.vmap(lambda jk, a, p: jax.random.choice(jk, a, shape=(intermediate_sample_num_pnts,), p=p))(jax.random.split(jkey, nb), query_points_l2, within_mask_l2.astype(jnp.float32))
        
        # make density uniform
        resolution = 0.005 # resolution to calculate density in meter
        unique_check = (valid_pnts_l2/resolution).astype(jnp.int32)
        unique_count = jnp.sum(jnp.all(unique_check[...,None,:] == unique_check[...,None,:,:], axis=-1), axis=-1)
        assert intermediate_sample_num_pnts >= num_points
        if i==coarse_to_fine_itr_no-1:
            valid_pnts_l2 = jax.vmap(lambda jk, a, p: jax.random.choice(jk, a, shape=(num_points,), p=p))(jax.random.split(jkey, nb), valid_pnts_l2, 1/unique_count.astype(jnp.float32))
            break
        valid_pnts_l2 = jax.vmap(lambda jk, a, p: jax.random.choice(jk, a, shape=(initial_sample_num_pnts,), p=p))(jax.random.split(jkey, nb), valid_pnts_l2, 1/unique_count.astype(jnp.float32))
        _, jkey = jax.random.split(jkey)
        valid_pnts = valid_pnts_l2

    # perform optimization
    def occupancy_loss(x):
        occ = models.apply('occ_predictor', query_objects, x) # (#pnt, 1)
        return jnp.sum(jnp.abs(occ-occ_boundary_value))

    grad_func = jax.value_and_grad(occupancy_loss)
    optimizer = optax.adam(2e-5)
    surface_pnts = valid_pnts_l2
    opt_state = optimizer.init(surface_pnts)
    for _ in range(10):
        loss, grad = grad_func(surface_pnts)
        updates, opt_state = optimizer.update(grad, opt_state, surface_pnts)
        surface_pnts = optax.apply_updates(surface_pnts, updates)
        if visualize:
            print(loss)
    
    # surface_normals = jax.grad(lambda x: jnp.sum(models.apply('occ_predictor', query_objects, x)))(surface_pnts)

    if visualize:
        # visualization
        import open3d as o3d
        for i in range(nb):
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(np.array(query_points_l1[within_mask_l1]))
            o3d.visualization.draw_geometries([pcd])

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(np.array(query_points_l2[within_mask_l2]))
            o3d.visualization.draw_geometries([pcd])

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(np.array(valid_pnts_l2[i]))
            o3d.visualization.draw_geometries([pcd])

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(np.array(surface_pnts)[i])
            o3d.visualization.draw_geometries([pcd])

    return surface_pnts