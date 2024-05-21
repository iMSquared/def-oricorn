# %%
# import libraries
from typing import Sequence
import jax.numpy as jnp
import numpy as np
from scipy.spatial import ConvexHull
# import open3d as o3d
import einops
import jax

import time

import util.transform_util as tutil

def hull_to_mesh(hull:ConvexHull):
    '''
    input
        hull
    
    return
        vertices (NV, ND)
        vertice_simplices (NF, 3)
        vertices_normal (NV, ND)
        triangle_normal (NF, ND)
    '''

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
    points = rng.uniform(-0.1,0.1,size=(nvertices, dim))
    hull = ConvexHull(points, incremental=True)

    while len(hull.vertices) < nvertices:
        hull.add_points(np.random.uniform(-0.1,0.1,size=(1, dim)))
    
    return hull_to_mesh(hull)




# %%
def minkowski_diff(vA:np.ndarray, pqA:np.ndarray, vB:np.ndarray, pqB:np.ndarray):
    '''
    calculate minkowski difference between two vertices

    args:
        vA (NB 3) : Vertices of object A
        pqA ((NB 3), (NB 4)) : Pos and quaternion of object A
        vB (NB 3) : Vertices of object B
        pqB ((NB 3), (NB 4)) : Pos and quaternion of object B

    return:
        vertices (NV, ND)
        vertice_simplices (NF, 3)
        vertices_normal (NV, ND)
        triangle_normal (NF, ND)
    '''
    
    vsAp = tutil.pq_action(*pqA, vA)
    vsBp = tutil.pq_action(*pqB, vB)
    amb = (vsAp[...,None,:] - vsBp[...,None,:,:])
    amb = einops.rearrange(amb, '... i j k -> ... (i j) k')
    chull = ConvexHull(amb)
    return hull_to_mesh(chull)


# %%
# o3d visualize
def create_o3d_vis(mesh_info:Sequence[np.ndarray]):
    import open3d as o3d
    '''
    mesh_info
        vertices (NV, ND)
        vertice_simplices (NF, 3)
        vertices_normal (NV, ND)
        triangle_normal (NF, ND)
    '''
    sAo3d = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(mesh_info[0]), o3d.utility.Vector3iVector(mesh_info[1]))
    sAo3d.vertex_normals = o3d.utility.Vector3dVector(mesh_info[2])
    sAo3d.triangle_normals = o3d.utility.Vector3dVector(mesh_info[3])
    return sAo3d

def create_o3d_lineset(shape:Sequence[np.ndarray]):
    import open3d as o3d

    '''
    shape
        vertices (NV, ND)
        vertice_simplices (NF, 3)
        vertices_normal (NV, ND)
        triangle_normal (NF, ND)
    '''
    pnts = shape[0]
    lineset = np.concatenate([shape[1][...,:2], shape[1][...,1:], shape[1][...,(0,2)]], axis=0)
    return o3d.geometry.LineSet(points=o3d.utility.Vector3dVector(pnts), lines=o3d.utility.Vector2iVector(lineset))

# mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
#     size=0.1, origin=[0, 0, 0])


def support(vsA, d, idx=False):
    invalid = jnp.any(jnp.abs(vsA) > 1e4, axis=-1)
    vsAdotd = jnp.where(invalid, -1e6, jnp.sum(vsA * d[...,None,:], axis=-1))
    am_idx = jnp.argmax(vsAdotd, axis=-1)[...,None]
    if idx:
        return jnp.squeeze(jnp.take_along_axis(vsA, am_idx[...,None], axis=-2), axis=-2), am_idx
    else:
        return jnp.squeeze(jnp.take_along_axis(vsA, am_idx[...,None], axis=-2), axis=-2)


def min_distance_to_simplex(Ws:jnp.ndarray, rec:bool=False, max_vertices_no:int=4):
    '''
    args:
        Ws (... NW ND) : vertices in simplex

    return 
        min_vec (... ND)
        updated_Ws (... NW ND)
        is_interior (... ) : zero point is interior
    '''
    inf_vale = 1e5
    dim = Ws.shape[-2]
    if dim == 1:
        # point
        return Ws[...,0,:], Ws, False
    if dim == 2:
        # edge
        a, b = Ws[...,0,:], Ws[...,1,:]
        ab = b-a
        oa = a
        abnorm = jnp.linalg.norm(ab, axis=-1, keepdims=True)
        abnormalize = ab/(abnorm + 1e-6)
        oa_dot_ab = jnp.einsum('...j,...j',oa, abnormalize)
        supports = jnp.clip(oa_dot_ab[...,None], -abnorm, 0)
        min_vec = a - supports * abnormalize
        return min_vec, Ws, False
    if dim == 3:
        if rec:
            minvec_edge = inf_vale
        else:
            # min distance to edges
            minvec_edges, _, _ = min_distance_to_simplex(jnp.stack([Ws[...,(0,1),:], Ws[...,(1,2),:], Ws[...,(0,2),:]], axis=-3))
            min_idx = jnp.argmin(jnp.linalg.norm(minvec_edges, axis=-1), axis=-1)
            minvec_edge = jnp.take_along_axis(minvec_edges, min_idx[...,None,None], axis=-2)[...,0,:]
        # min distance to a face
        a, b, c = Ws[...,0,:], Ws[...,1,:], Ws[...,2,:]
        ab, ac, ao = b-a, c-a, -a
        sol = jnp.einsum('...ij,...j->...i', jnp.linalg.pinv(jnp.stack([ab,ac], axis=-1)), ao)
        is_interior = jnp.logical_and(jnp.all(sol > 0, axis=-1), jnp.sum(sol, axis=-1) <= 1.0)
        if max_vertices_no == 3:
            # Ws update
            remain_vtx_idx = jnp.array([[0,1],[1,2],[0,2]], dtype=jnp.int32)[min_idx]
            Ws_updated = jnp.take_along_axis(Ws, remain_vtx_idx[...,None], axis=-2)
            return minvec_edge, Ws_updated, is_interior
        else:
            minvec_face = a + jnp.einsum('...ij,...j->...i', jnp.stack([ab,ac], axis=-1), sol)
            return jnp.where(is_interior[...,None], minvec_face, minvec_edge), Ws, False
    if dim == 4:
        assert max_vertices_no == 4
        minvec_faces, _, _ = min_distance_to_simplex(jnp.stack([Ws[...,(0,1,2),:], Ws[...,(0,1,3),:], Ws[...,(0,2,3),:], Ws[...,(1,2,3),:]], axis=-3), rec=True)
        edges = jnp.stack([Ws[...,(0,1),:], Ws[...,(0,2),:], Ws[...,(0,3),:], 
                            Ws[...,(1,2),:], Ws[...,(1,3),:], Ws[...,(2,3),:]], axis=-3)
        minvec_edges, _, _ = min_distance_to_simplex(edges, rec=True)
        minvec_fe = jnp.concatenate([minvec_faces,minvec_edges], axis=-2)
        min_idx_fe = jnp.argmin(jnp.linalg.norm(minvec_fe, axis=-1), axis=-1)
        minvec_face = jnp.take_along_axis(minvec_fe, min_idx_fe[...,None,None], axis=-2)[...,0,:]
        # assert jnp.all(jnp.abs(minvec_face) != inf_vale)
        min_idx_face = jnp.array([0,1,2,3, 0,0,1,0,1,2], dtype=jnp.int32)[min_idx_fe]

        a, b, c, d = Ws[...,0,:], Ws[...,1,:], Ws[...,2,:], Ws[...,3,:]
        ab, ac, ad, ao = b-a, c-a, d-a, -a
        sol = jnp.einsum('...ij,...j->...i', jnp.linalg.inv(jnp.stack([ab,ac,ad], axis=-1)), ao)
        is_interior = jnp.logical_and(jnp.all(sol > 0, axis=-1), jnp.sum(sol, axis=-1) <= 1.0)

        remain_vtx_idx = jnp.array([[0,1,2],[0,1,3],[0,2,3],[1,2,3]], dtype=jnp.int32)[min_idx_face]
        Ws_updated = jnp.take_along_axis(Ws, remain_vtx_idx[...,None], axis=-2)

        return minvec_face, Ws_updated, is_interior
    else:
        raise ValueError

def GJK_3d(vsA:jnp.ndarray, vsB:jnp.ndarray, pqA:Sequence[jnp.ndarray]=None, pqB:Sequence[jnp.ndarray]=None, 
            itr_limit:int=12, collision_only:bool=True):
    '''
    args:
        vsA (... NV 3)
        vsB (... NV 3)
        pqA ((... 3), (... 4)) : position and quaternion (x y z w) of object A
        pqB ((... 3), (... 4)) : position and quaternion (x y z w) of object A
        itr_limit : GJK iteration limit
        collision_only : used only for collision checking, not for min distance. For min distance, this value should be False
    
    return:
        collision_res (... ) if not collision_only
        else
            collision result (... )
            minimum vector from A to B (... 3)
            argmin index in A (... )
            argmin index in B (... )
    '''

    invalid_vsA = jnp.all(jnp.abs(vsA)>1e4, (-1,-2))
    invalid_vsB = jnp.all(jnp.abs(vsB)>1e4, (-1,-2))
    invalid_vtx = jnp.logical_or(invalid_vsA, invalid_vsB)

    if pqA is None or pqB is None:
        vsA_tf = vsA
        vsB_tf = vsB
    else:
        pqA, pqB = jax.tree_util.tree_map(lambda x: x[...,None,:], (pqA, pqB))
        vsA_tf = tutil.pq_action(*pqA, vsA)
        vsB_tf = tutil.pq_action(*pqB, vsB)

    def mink_dif(d, idx=False):
        if idx:
            spA = support(vsA_tf, d, idx=idx) 
            spB = support(vsB_tf, -d, idx=idx)
            return spA[0] - spB[0], spA[1], spB[1]
        else:
            return support(vsA_tf, d, idx=idx) - support(vsB_tf, -d, idx=idx)
    
    # initialization
    v = jnp.ones((3,), dtype=jnp.float32)
    w = mink_dif(-v)

    # assert jnp.all(jnp.abs(w) < 10000)

    found_infeasible_v = jnp.einsum('...i,...i', w, -v) < 0
    Ws = w[...,None,:]

    # end condition : find_infeasible_v, inside_simplex
    inside_simplex = False
    for i in range(itr_limit):
        # get support vector
        Ws_prior = Ws
        v, Ws_remain, is_interior = min_distance_to_simplex(Ws)
        inside_simplex = jnp.logical_or(inside_simplex, is_interior)

        # update Ws
        if not collision_only and i == itr_limit-1:
            w, Aidx, Bidx = mink_dif(-v, idx=True)
        else:
            w = mink_dif(-v)
        found_infeasible_v = jnp.logical_or(found_infeasible_v, jnp.einsum('...i,...i', w, -v) < 0)
        Ws = jnp.concatenate([Ws_remain, w[...,None,:]], axis=-2)

        # finish
        if i > 3:
            Ws = jnp.where(inside_simplex[...,None,None], Ws_prior, Ws)
        
    if collision_only:
        return jnp.logical_and(inside_simplex, jnp.logical_not(found_infeasible_v))
    
    else:
        # return jnp.logical_and(inside_simplex, jnp.logical_not(found_infeasible_v)), v, Aidx, Bidx
        return jnp.logical_and(inside_simplex, jnp.logical_not(found_infeasible_v)), v


def GJK_3d_new(vsA:jnp.ndarray, vsB:jnp.ndarray, pqA:Sequence[jnp.ndarray]=None, pqB:Sequence[jnp.ndarray]=None, 
            itr_limit:int=12, collision_only:bool=True):
    '''
    args:
        vsA (... NV 3)
        vsB (... NV 3)
        pqA ((... 3), (... 4)) : position and quaternion (x y z w) of object A
        pqB ((... 3), (... 4)) : position and quaternion (x y z w) of object A
        itr_limit : GJK iteration limit
        collision_only : used only for collision checking, not for min distance. For min distance, this value should be False
    
    return:
        collision_res (... ) if not collision_only
        else
            collision result (... )
            minimum vector from A to B (... 3)
            argmin index in A (... )
            argmin index in B (... )
    '''
    if pqA is None or pqB is None:
        vsA_tf = vsA
        vsB_tf = vsB
    else:
        pqA, pqB = jax.tree_util.tree_map(lambda x: x[...,None,:], (pqA, pqB))
        vsA_tf = tutil.pq_action(*pqA, vsA)
        vsB_tf = tutil.pq_action(*pqB, vsB)

    def mink_dif(d, idx=False):
        if idx:
            spA = support(vsA_tf, d, idx=idx) 
            spB = support(vsB_tf, -d, idx=idx)
            return spA[0] - spB[0], spA[1], spB[1]
        else:
            return support(vsA_tf, d, idx=idx) - support(vsB_tf, -d, idx=idx)
    
    # initialization
    v = jnp.ones((3,), dtype=jnp.float32)
    w = mink_dif(-v)
    found_infeasible_v = jnp.einsum('...i,...i', w, -v) < 0
    Ws = w[...,None,:]

    # end condition : find_infeasible_v, inside_simplex
    inside_simplex = False
    def _f(i, carry, preprocessing=False):
        Ws, v, inside_simplex, found_infeasible_v = carry
        # get support vector
        Ws_prior = Ws
        v, Ws_remain, is_interior = min_distance_to_simplex(Ws)
        inside_simplex = jnp.logical_or(inside_simplex, is_interior)

        # update Ws
        w = mink_dif(-v)
        found_infeasible_v = jnp.logical_or(found_infeasible_v, jnp.einsum('...i,...i', w, -v) < 0)
        Ws = jnp.concatenate([Ws_remain, w[...,None,:]], axis=-2)

        if not preprocessing:
            # finish
            Ws = jnp.where(inside_simplex[...,None,None], Ws_prior, Ws)
        return Ws, v, inside_simplex, found_infeasible_v

    for i in range(4):
        Ws, v, inside_simplex, found_infeasible_v = _f(i, (Ws, v, inside_simplex, found_infeasible_v), preprocessing=True)
    Ws, v, inside_simplex, found_infeasible_v = jax.lax.fori_loop(0, itr_limit-4, _f, (Ws, v, inside_simplex, found_infeasible_v))
    
    if collision_only:
        return jnp.logical_and(inside_simplex, jnp.logical_not(found_infeasible_v))
    else:
        # return jnp.logical_and(inside_simplex, jnp.logical_not(found_infeasible_v)), v, Aidx, Bidx
        return jnp.logical_and(inside_simplex, jnp.logical_not(found_infeasible_v)), v

def GJK_2d(vsA:jnp.ndarray, vsB:jnp.ndarray, pqA:Sequence[jnp.ndarray]=None, pqB:Sequence[jnp.ndarray]=None,
             itr_limit:int=12, collision_only:bool=True):
    '''
    args:
        vsA (... NV 3)
        vsB (... NV 3)
        pqA ((... 3), (... 4)) : position and quaternion (x y z w) of object A
        pqB ((... 3), (... 4)) : position and quaternion (x y z w) of object A
        itr_limit : GJK iteration limit
        collision_only : used only for collision checking, not for min distance. For min distance, this value should be False
    
    return:
        collision_res (... ) if not collision_only
        else
            collision result (... )
            minimum vector from A to B (... 3)
            argmin index in A (... )
            argmin index in B (... )
    '''

    if pqA is None or pqB is None:
        vsA_tf = vsA
        vsB_tf = vsB
    else:
        pqA, pqB = jax.tree_util.tree_map(lambda x: x[...,None,:], (pqA, pqB))
        vsA_tf = tutil.pq_action(*pqA, vsA)
        vsB_tf = tutil.pq_action(*pqB, vsB)

    def mink_dif(d, idx=False):
        if idx:
            spA = support(vsA_tf, d, idx=idx) 
            spB = support(vsB_tf, -d, idx=idx)
            return spA[0] - spB[0], spA[1], spB[1]
        else:
            return support(vsA_tf, d, idx=idx) - support(vsB_tf, -d, idx=idx)
    
    # initialization
    v = jnp.ones((2,), dtype=jnp.float32)
    w = mink_dif(-v)
    found_infeasible_v = jnp.einsum('...i,...i', w, -v) < 0
    Ws = w[...,None,:]

    # end condition : find_infeasible_v, inside_simplex
    # TODO: change this loop to jax.lax.for_i_loop
    inside_simplex = False
    for i in range(itr_limit):
        # get support vector
        Ws_prior = Ws
        v, Ws_remain, is_interior = min_distance_to_simplex(Ws, max_vertices_no=3)
        inside_simplex = jnp.logical_or(inside_simplex, is_interior)

        # update Ws
        if not collision_only and i == itr_limit-1:
            w, Aidx, Bidx = mink_dif(-v, idx=True)
        else:
            w = mink_dif(-v)
        found_infeasible_v = jnp.logical_or(found_infeasible_v, jnp.einsum('...i,...i', w, -v) < 0)
        Ws = jnp.concatenate([Ws_remain, w[...,None,:]], axis=-2)

        # finish
        if i > 2:
            Ws = jnp.where(inside_simplex[...,None,None], Ws_prior, Ws)
        
    if collision_only:
        return jnp.logical_and(inside_simplex, jnp.logical_not(found_infeasible_v))
    
    else:
        return jnp.logical_and(inside_simplex, jnp.logical_not(found_infeasible_v)), v, Aidx, Bidx


def cvx_dec_culling(vsAB_tf:Sequence[jnp.ndarray], max_ncd:int, valid_mask_vtx:Sequence[jnp.ndarray]=None, epsilon:float=1e-6):
    '''
    args:
        vsAB_tf (... 2 ND NV 3)
        max_ncd : maximum number of culling
        valid_mask_vtx (... 2 ND NV 1)
    
    returns:
        vsAspread (... NC NV 3)
        vsBspread (... NC NV 3)
        valid_mask_dc_Aspread (... NC NV 1)
        valid_mask_dc_Bspread (... NC NV 1)
    '''
    if valid_mask_vtx is None:
        valid_mask_vtx = jnp.all(jnp.abs(vsAB_tf) < 1e3, axis=-1, keepdims=True)
    ncd = vsAB_tf.shape[-3]
    valid_mask_dc = np.any(valid_mask_vtx, axis=-2)[...,0]
    vsA, vsB = vsAB_tf[...,0,:,:,:], vsAB_tf[...,1,:,:,:]
    valid_mask_dc = jnp.broadcast_to(valid_mask_dc, jnp.broadcast_shapes(valid_mask_dc.shape, vsAB_tf[...,0,0].shape))

    # culling
    muAB = jnp.sum(vsAB_tf*valid_mask_vtx, axis=-2, keepdims=True)/(jnp.sum(valid_mask_vtx, axis=-2, keepdims=True) + epsilon)
    rAB = jnp.max(jnp.where(valid_mask_vtx, jnp.linalg.norm(vsAB_tf - muAB, axis=-1, keepdims=True), 0), axis=-2, keepdims=True) 
    valid_mask_dcA, valid_mask_dcB = valid_mask_dc[...,0,:], valid_mask_dc[...,1,:]
    muA, muB = muAB[...,0,:,:,:], muAB[...,1,:,:,:]
    rA, rB = rAB[...,0,:,:,:], rAB[...,1,:,:,:]
    dif = jnp.linalg.norm(muA[...,:,None,:,:] - muB[...,None,:,:,:], axis=-1, keepdims=True)
    dif = dif - rA[...,:,None,:,:] - rB[...,None,:,:,:]
    vmdc_spread = jnp.logical_and(valid_mask_dcA[...,:,None,None,None], valid_mask_dcB[...,None,:,None,None])
    # dif = jnp.where(vmdc_spread, dif, 1)
    dif = jnp.where(vmdc_spread, dif, 1e5)
    dif_flat = einops.rearrange(dif, '... i j p q -> ... (i j) p q')
    
    # top k index
    pidx = jnp.argsort(dif_flat, axis=-3)[...,:max_ncd,0,0]

    pidxA = pidx//ncd
    pidxB = pidx%ncd
    vsAspread = jnp.take_along_axis(vsA, pidxA[...,None,None], axis=-3)
    vsBspread = jnp.take_along_axis(vsB, pidxB[...,None,None], axis=-3)
    valid_mask_dc_Aspread = jnp.take_along_axis(valid_mask_dcA, pidxA, axis=-1)
    valid_mask_dc_Bspread = jnp.take_along_axis(valid_mask_dcB, pidxB, axis=-1)

    # assert jnp.all(jnp.any(valid_mask_dc_Aspread, -1))
    # assert jnp.all(jnp.any(valid_mask_dc_Bspread, -1))

    return vsAspread, vsBspread, valid_mask_dc_Aspread, valid_mask_dc_Bspread


def GJK_cvx_dec(vsA:Sequence[jnp.ndarray], vsB:Sequence[jnp.ndarray], pqAB:Sequence[jnp.ndarray], itr_limit:int=6, max_ncd:int=3, collision_only:bool=False):
    '''
    args:
        vsA (... ND NV 3) : ND dimension is for decompositions
        vsB (... ND NV 3) : ND dimension is for decompositions
        pqAB ((... 2 3), (... 2 4)) quaternion order (x y z w)
        itr_limit : GJK iteration limit
        max_ncd : culling elements number limit
        collision_only : if True, output only collision results / else min distance

    return:
        col_res (... ) if collision_only is Ture
        else (col_res, mindist, minvec) ((... ), (... ), (... 3))
    '''
    ncd = vsA.shape[-3]
    nvx = vsA.shape[-2]

    vsAB = jnp.stack([vsA, vsB], axis=-4)

    # valid_mask_vtx = (vsAB[...,0:1] < 1000)
    valid_mask_vtx = (jnp.all(jnp.abs(vsAB) < 1000, -1, keepdims=True))
    # valid_mask_dc = (vsAB[...,0,0] < 1000)
    valid_mask_dc = valid_mask_vtx[...,0,0]
    if pqAB is not None:
        pqAB = jax.tree_util.tree_map(lambda x: x[...,None,None,:], pqAB)
        vsAB = tutil.pq_action(*pqAB, vsAB)
    vsA, vsB = vsAB[...,0,:,:,:], vsAB[...,1,:,:,:]
    valid_mask_dc = jnp.broadcast_to(valid_mask_dc, jnp.broadcast_shapes(valid_mask_dc.shape, vsAB[...,0,0].shape))

    vsAspread, vsBspread, valid_mask_dc_Aspread, valid_mask_dc_Bspread = cvx_dec_culling(vsAB, max_ncd, valid_mask_vtx)
    
    invalid_vsA = jnp.all(jnp.any(jnp.abs(vsAspread)>1e4, -1), -1)
    invalid_vsB = jnp.all(jnp.any(jnp.abs(vsBspread)>1e4, -1), -1)
    invalid_vtx = jnp.logical_or(invalid_vsA, invalid_vsB)

    col_res_spread = GJK_3d(vsAspread, vsBspread, itr_limit=itr_limit, collision_only=collision_only)
    # assert jnp.all(jnp.abs(col_res_spread[1]) < 10000)
    if not collision_only:
        colmask = col_res_spread[0].astype(jnp.float32)
        min_dist_spread = (1-colmask) * jnp.linalg.norm(col_res_spread[1], axis=-1) - colmask
        min_dist_spread =jnp.where(jnp.logical_and(valid_mask_dc_Aspread, valid_mask_dc_Bspread), min_dist_spread, 1e6)
        min_cvxd_idx = jnp.argmin(min_dist_spread, axis=-1)
        minvec = jnp.take_along_axis(col_res_spread[1], min_cvxd_idx[...,None,None], axis=-2)[...,0,:]
        mindist = jnp.take_along_axis(min_dist_spread, min_cvxd_idx[...,None], axis=-1)[...,0]
    
        # assert jnp.all(jnp.abs(mindist) < 10000)

        col_res_spread = col_res_spread[0]

    col_res = jnp.all(jnp.stack([col_res_spread, valid_mask_dc_Aspread, valid_mask_dc_Bspread], -1), -1)
    col_res = jnp.any(col_res, axis=(-1,))
    if collision_only:
        return col_res
    else:
        return col_res, mindist, minvec

if __name__ == "__main__":
    jkey = jax.random.PRNGKey(0)
    rng = np.random.default_rng(12345)
    # create shapes
    sA = generate_convex_shape(rng, 20)
    sB = generate_convex_shape(rng, 20)
    nb = 100
    vA = einops.repeat(sA[0], 'i j -> k i j', k=nb)
    vB = einops.repeat(sB[0], 'i j -> k i j', k=nb)

    GJK_3d_jit = jax.jit(GJK_3d, static_argnames=('collision_only',))
    GJK_3d_old_jit = jax.jit(GJK_3d_old, static_argnames=('collision_only',))

    pos = jax.random.uniform(jkey, shape=(nb, 2, 3), dtype=jnp.float32, minval=-0.5, maxval=0.5)
    cst1 = time.time()
    res = GJK_3d_jit(vA+pos[...,0:1,:], vB+pos[...,1:2,:], collision_only=False)
    cet1 = time.time()
    cst2 = time.time()
    res = GJK_3d_old_jit(vA+pos[...,0:1,:], vB+pos[...,1:2,:], collision_only=False)
    cet2 = time.time()
    print(f"compile time new: {cet1-cst1} // compile time old: {cet2-cst2}")
    itr_no = 1000
    st = time.time()
    for i in range(itr_no):
        pos = jax.random.uniform(jkey, shape=(nb, 2, 3), dtype=jnp.float32, minval=-0.5, maxval=0.5)
        res = GJK_3d_jit(vA+pos[...,0:1,:], vB+pos[...,1:2,:], collision_only=False)
        res2 = GJK_3d_old_jit(vA+pos[...,0:1,:], vB+pos[...,1:2,:], collision_only=False)
        assert jnp.sum(res[0]!=res2[0]) < 1e-6
        assert jnp.sum(jnp.abs(res[1]-res2[1])) < 1e-6
    et = time.time()
    print((et-st)/itr_no, (et-st)/itr_no/nb)