# %%
# import libraries
import ray
import numpy as np
import fcl
import util.transform_util as tutil
from scipy.spatial.transform import Rotation as sciR
import einops
import glob
import jax
import jax.numpy as jnp
import open3d as o3d
import pybullet as p
import copy
import pickle
from datetime import datetime
import os

class ObjLoad(object):
    def __init__(self, fileName, scale=None, o3d_obj=False):
        vertices = []
        faces = []
        f = open(fileName)
        for line in f:
            if line[:2] == "v ":
                index1 = line.find(" ") + 1
                index2 = line.find(" ", index1 + 1)
                index3 = line.find(" ", index2 + 1)
                if scale is None:
                    vertex = (float(line[index1:index2]), float(line[index2:index3]), float(line[index3:-1]))
                else:
                    vertex = (scale[0]*float(line[index1:index2]), scale[1]*float(line[index2:index3]), scale[2]*float(line[index3:-1]))
                vertices.append(vertex)

            elif line[0] == "f":
                string = line.replace("//", "/")
                ##
                i = string.find(" ") + 1
                face = []
                for item in range(string.count(" ")):
                    if string.find(" ", i) == -1:
                        face.append(int(string[i:-1].split("/")[0])-1)
                        break
                    face.append(int(string[i:string.find(" ", i)].split("/")[0])-1)
                    i = string.find(" ", i) + 1
                ##
                faces.append(tuple(face))

        vertices = np.array(vertices).astype(float)
        faces = np.array(faces).astype(int)
        
        f.close()

        ## vertices compensation
        # vertices_offset = (np.min(vertices,0) + np.max(vertices,0)) * 0.5
        # vertices = vertices - vertices_offset
        # self.vertices = vertices

        # generate collision shapes
        model = fcl.BVHModel()
        model.beginModel(len(vertices), len(faces))
        model.addSubModel(vertices, faces)
        model.endModel()

        # model = fcl.Box(0.02000000, 0.08000000, 0.0400000)

        col_object = fcl.CollisionObject(model, fcl.Transform())

        # # o3d mesh
        if o3d_obj:
            mesho3d = o3d.io.read_triangle_mesh(fileName)
            # aabb_o3d = mesho3d.get_axis_aligned_bounding_box()
            # offset = 0.5*(aabb_o3d.get_max_bound() + aabb_o3d.get_min_bound())
            # mesho3d.translate(-offset)
            self.mesho3d = mesho3d
            mesho3d = o3d.t.geometry.TriangleMesh.from_legacy(mesho3d)
            sceneo3d = o3d.t.geometry.RaycastingScene()
            sceneo3d.add_triangles(mesho3d)
            self.sceneo3d = sceneo3d
        else:
            self.sceneo3d = None

        self.col_object = col_object
        self.aabb = np.concatenate([np.min(vertices,0), np.max(vertices,0)], axis=0)

def fcl_query(o_cls_list, pos, quat, visualize=False):
    drequest = fcl.DistanceRequest(enable_nearest_points=True)
    crequest = fcl.CollisionRequest(enable_contact=True)

    o_list = [oc.col_object for oc in o_cls_list]

    for o, ps, q in zip(o_list, pos, quat):
        o.setTransform(fcl.Transform(sciR.from_quat(q).as_matrix(), ps))
    
    cresult = fcl.CollisionResult()
    cret = fcl.collide(*o_list, crequest, cresult)
    if cret != 0:
        sd = -cresult.contacts[0].penetration_depth
        dir = cresult.contacts[0].normal
    else:
        dresult = fcl.DistanceResult()
        sd = fcl.distance(*o_list, drequest, dresult)
        if sd == -1:
            sd = 0
        else:
            # need to check inside or outside
            a_maxlen = np.max(o_cls_list[0].aabb[3:]-o_cls_list[0].aabb[:3])
            a_minlen = np.min(o_cls_list[0].aabb[3:]-o_cls_list[0].aabb[:3])
            b_maxlen = np.max(o_cls_list[1].aabb[3:]-o_cls_list[1].aabb[:3])
            b_minlen = np.min(o_cls_list[1].aabb[3:]-o_cls_list[1].aabb[:3])
            # np.all(np.max(o_cls_list[0].aabb[3:]-o_cls_list[0].aabb[:3]) > np.max(o_cls_list[1].aabb[3:]-o_cls_list[1].aabb[:3]), axis=0)
            if o_cls_list[0].sceneo3d is not None and a_minlen > b_maxlen:
                # check 2 insdie 1
                pnt2 = o_cls_list[1].vertices[0]
                pnt2_tf, _ = p.multiplyTransforms(pos[1], quat[1], pnt2, np.array([0,0,0,1]))
                qpnt2, _ = p.multiplyTransforms(*p.invertTransform(pos[0], quat[0]), pnt2_tf, np.array([0,0,0,1]))
                ores1 = o_cls_list[0].sceneo3d.compute_occupancy(o3d.core.Tensor([qpnt2], dtype=o3d.core.Dtype.Float32))
                if ores1==1:
                    sd *= -1
            elif o_cls_list[1].sceneo3d is not None and a_maxlen < b_minlen:
                # check 1 insdie 2
                pnt1 = o_cls_list[0].vertices[0]
                pnt1_tf, _ = p.multiplyTransforms(pos[0], quat[0], pnt1, np.array([0,0,0,1]))
                qpnt1, _ = p.multiplyTransforms(*p.invertTransform(pos[1], quat[1]), pnt1_tf, np.array([0,0,0,1]))
                ores2 = o_cls_list[1].sceneo3d.compute_occupancy(o3d.core.Tensor([qpnt1], dtype=o3d.core.Dtype.Float32))
                if ores2==1:
                    sd *= -1

        dir = dresult.nearest_points[1] - dresult.nearest_points[0]

    if visualize and sd < 0:
        ## visualization
        o1mesh = copy.deepcopy(o_cls_list[0].mesho3d)
        o1mesh.compute_vertex_normals()
        o1mesh.transform(tutil.pq2H(pos[0], quat[0]))
        ls1 = o3d.geometry.LineSet.create_from_triangle_mesh(o1mesh)

        o2mesh = copy.deepcopy(o_cls_list[1].mesho3d)
        o2mesh.compute_vertex_normals()
        o2mesh.transform(tutil.pq2H(pos[1], quat[1]))
        ls2 = o3d.geometry.LineSet.create_from_triangle_mesh(o2mesh)
        ls2.paint_uniform_color(np.array([0.7,0.2,0.3]))

        if cret == 0:
            p1 = o3d.geometry.TriangleMesh.create_sphere(radius=0.001)
            p1.compute_vertex_normals()
            p1.translate(dresult.nearest_points[0])

            p2 = o3d.geometry.TriangleMesh.create_sphere(radius=0.001)
            p2.compute_vertex_normals()
            p2.translate(dresult.nearest_points[1])
            p2.paint_uniform_color(np.array([0.1,0.2,0.8]))

            # # attach mesh 1
            # ls3 = copy.deepcopy(ls1)
            # ls3.translate(dir)

            o3d.visualization.draw_geometries([ls1, o2mesh, p1, p2])
        else:
            o3d.visualization.draw_geometries([ls1, o2mesh])

    return sd, dir

def batch_query(o_list, pos, quat):
    ans = np.zeros([pos.shape[0],1])
    dir = np.zeros([pos.shape[0],3])
    for i in range(pos.shape[0]):
        ans[i], dir[i] = fcl_query(o_list, pos[i], quat[i])
    dir = dir / (np.linalg.norm(dir, axis=-1, keepdims=True) + 1e-7)
    return ans, dir

# %%
# def visualize_with_o3d(x, y, idx_pair, value_type='all', max_sample_no=3, mesh_dir_list=MESH_DIR_LIST, primitive_name=[]):

#     ## visualize 1 frame
#     pos_relative, quat_relative = tutil.pq_multi(*tutil.pq_inv(x[0][:,0], x[1][:,0]), x[0][:,1], x[1][:,1])
#     o1mesh = o3d.io.read_triangle_mesh(mesh_dir_list[idx_pair[0]])
#     o1mesh.compute_vertex_normals()
#     o1mesh.translate(-o1mesh.get_axis_aligned_bounding_box().get_center())
#     ls1 = o3d.geometry.LineSet.create_from_triangle_mesh(o1mesh)

#     if mesh_dir_list[idx_pair[1]] in primitive_name:
#         ray_len = primitive_name[mesh_dir_list[idx_pair[1]]][-1]
#         o2mesh = o3d.geometry.TriangleMesh.create_cylinder(radius=0.002, height=ray_len)
#         o2mesh.compute_vertex_normals()
#     else:
#         o2mesh = o3d.io.read_triangle_mesh(mesh_dir_list[idx_pair[1]])
#         o2mesh.compute_vertex_normals()
#         o2mesh.translate(-o2mesh.get_axis_aligned_bounding_box().get_center())
#         ls2 = o3d.geometry.LineSet.create_from_triangle_mesh(o2mesh)

#     con_value = np.clip(y * 1/0.004, -1, 1)[...,0]
#     raymesh_list = []
#     # random_idx = np.random.randint(0, y.shape[0], size=(10,))
#     if value_type == 'all':
#         random_idx = np.where(np.ones_like(con_value))[0]
#     elif value_type == 'negative':
#         random_idx = np.where(con_value < 0)[0]
#     elif value_type == 'positive':
#         random_idx = np.where(con_value > 0)[0]
#     elif value_type == 'equal':
#         random_idx = np.where(con_value == 0)[0]

#     random_idx = np.random.choice(random_idx, max_sample_no)
#     for i in range(len(random_idx)):
#         idx_ = random_idx[i]
#         rm_ = copy.deepcopy(o2mesh)
#         rm_.transform(tutil.pq2H(pos_relative[idx_], quat_relative[idx_]))
#         rm_.paint_uniform_color(np.array([-con_value[idx_]*0.5 + 0.5,0.0, con_value[idx_]*0.5 + 0.5]))
#         raymesh_list.append(rm_)

#     meshframe = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
#     o3d.visualization.draw_geometries([o1mesh, *raymesh_list])

# def pybullet_vis(obj_filename, pos_test, quat_test):
#     mb_list = []
#     for i, fn in enumerate(obj_filename):

#         if fn in PRIMITIVE_NAME:
#             pparams = PRIMITIVE_PARAMS[fn]
#             if pparams[0] == 0:
#                 pass
#             elif pparams[0] == 1:
#                 visualShapeId = p.createVisualShape(shapeType=p.GEOM_CYLINDER,
#                                             rgbaColor=[1, 1, 1, 1],
#                                             radius=pparams[1],
#                                             length=pparams[2],
#                                             )

#         else:
#             mesh_o3d =  o3d.io.read_triangle_mesh(fn)
#             aabb_o3d = mesh_o3d.get_axis_aligned_bounding_box()
#             offset = 0.5*(aabb_o3d.get_max_bound() + aabb_o3d.get_min_bound())
#             visualShapeId = p.createVisualShape(shapeType=p.GEOM_MESH,
#                                             fileName=fn,
#                                             rgbaColor=[1, 1, 1, 1],
#                                             visualFramePosition=-offset,
#                                             )
#         mb_list.append(p.createMultiBody(baseMass=0,
#                         baseVisualShapeIndex=visualShapeId,
#                         basePosition=pos_test[i],
#                         baseOrientation=quat_test[i]))
#     return mb_list
