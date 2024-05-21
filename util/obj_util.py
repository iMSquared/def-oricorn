from __future__ import annotations
import pybullet as p
import numpy as np
import jax.numpy as jnp
import os


import util.render_util as rutil
import util.io_util as ioutil
import util.cvx_util as cxutil


class ObjCls(object):
    def __init__(
            self, 
            obj_finename=None, 
            scale=np.array([1.,1.,1.]), 
            primitive_params=None, 
            pbcolobj=None, 
            pbvisobj=None, 
            o3dobj=None, 
            pcdobj=None, 
            shift_com=False
    ):
        self.obj_filename = obj_finename
        self.primitive_params = primitive_params # 0: sphere, 1: box, 2: cylinder ....
        self.pbcolobj = pbcolobj
        self.pbvisobj = pbvisobj
        self.o3dobj = o3dobj
        self.pcdobj = pcdobj
        self.shift_com = shift_com

        if self.pbvisobj is not None or self.pbcolobj is not None:

            build_dir_name = os.path.basename(os.path.dirname(pbcolobj))
            build_metadata = ioutil.BuildMetadata.from_str(build_dir_name)
            vertices, faces, vtx_no, face_no = cxutil.vex_obj_parsing(
                pbcolobj, build_metadata.max_dec_num, build_metadata.max_vertices, scale=scale)
            self.canonical_vtx_fc = (vertices, faces)
            self.cvx_pd = (vertices, faces, vtx_no, face_no)
            
            self.create_pbobj(scale)


    def create_pbobj(self, scale):
        if self.pbcolobj is not None:
            cshape = p.createCollisionShape(p.GEOM_MESH, fileName=self.pbcolobj, meshScale=[scale,scale,scale])
        else:
            cshape = -1

        # if self.pbvisobj is not None and self.pbvisobj == self.pbcolobj:
        #     vtx_mask = np.all(np.abs(self.cvx_pd[0])<1e3, axis=-1)
        #     vtx = self.cvx_pd[0][np.where(vtx_mask)]
        #     aabb = np.concatenate([jnp.min(vtx, -2), jnp.max(vtx, -2)], -1)
        #     fcs_mask = self.cvx_pd[1][...,0]>=0
        #     fcs = self.cvx_pd[1][np.where(fcs_mask)].reshape(-1)
        #     uvs = rutil.texcoord_parameterization(vtx)
        #     vshape = p.createVisualShape(p.GEOM_MESH, vertices=list(vtx), indices=list(fcs), uvs=list(uvs))
        if self.pbvisobj is not None:
            vshape = p.createVisualShape(p.GEOM_MESH, fileName=self.pbvisobj, meshScale=[scale,scale,scale])
        else:
            vshape = -1
        baseInertialFramePosition = np.zeros(3)
        # if self.shift_com:
        #     ri = np.random.randint(0,3)
        #     sign = 1 if np.random.uniform()<0.5 else -1
        #     aabb_len = (aabb[...,3:] - aabb[...,:3])
        #     aabb_cen = 0.5*(aabb[...,3:] + aabb[...,:3])
        #     if ri == 0:
        #         axis_idx = np.argmax(aabb_len)
        #         baseInertialFramePosition[axis_idx] = sign*1.5 * 0.5 * aabb_len[axis_idx] + aabb_cen[axis_idx]
        #     elif ri == 1:
        #         axis_idx = np.argsort(aabb_len)[-2]
        #         baseInertialFramePosition[axis_idx] = sign*1.5 * 0.5 * aabb_len[axis_idx] + aabb_cen[axis_idx]
        self.cvx_pd = (self.cvx_pd[0]-baseInertialFramePosition, *self.cvx_pd[1:]) # need to incorporate offset in the vertices
        self.baseInertialFramePosition = baseInertialFramePosition
        self.pbobj = p.createMultiBody(baseMass=0.3, baseInertialFramePosition=baseInertialFramePosition, baseCollisionShapeIndex=cshape, baseVisualShapeIndex=vshape)


def cal_center_from_obj_file(fileName):
    vtx = []
    f = open(fileName)
    for line in f:
        if line[:2] == "v ":
            index1 = line.find(" ") + 1
            index2 = line.find(" ", index1 + 1)
            index3 = line.find(" ", index2 + 1)
            vertex = (float(line[index1:index2]), float(line[index2:index3]), float(line[index3:-1]))
            vtx.append(vertex)
    f.close()
    return (np.min(vtx, axis=0) + np.max(vtx, axis=0)) * 0.5