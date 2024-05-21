from __future__ import annotations
import os
# os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"   
# os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"    # NVISII will crash when showed multiple devices with jax.

import jax.numpy as jnp
import jax
import flax
import random
import numpy as np
import sys
from typing import List, Tuple, Callable
import matplotlib.pyplot as plt
from functools import partial
from pathlib import Path
import pickle
import open3d as o3d
import pybullet as pb
import time, logging
import yaml
import einops

BASEDIR = Path(__file__).parent.parent
if BASEDIR not in sys.path:
    sys.path.insert(0, str(BASEDIR))

import util.model_util as mutil
import util.render_util as rutil
import util.transform_util as tutil
import util.cvx_util as cxutil
import util.franka_util as fkutil
import util.environment_util as env_util
import util.io_util as ioutil
import util.inference_util as ifutil
from imm_pb_util.bullet_envs.robot import PandaGripper, FrankaPanda

REACHABILITY_POS_LIMIT = 0.003
REACHABILITY_QUAT_LIMIT = 0.10

# REACHABILITY_POS_LIMIT = 0.005
# REACHABILITY_QUAT_LIMIT = 0.13

def apply_transf(obj, pos, quat, rot_configs):
    return obj.apply_pq_z(pos, quat, rot_configs)


def generate_grasp_candidates(
        jkey: jax.Array, 
        target_obj: cxutil.CvxObjects, 
        default_hand_objs: cxutil.CvxObjects, 
        n_grasps: int, 
        models: flax.linen.Module,
        min_n_valid: int=20,
        env_type: str='table',
        env_obj: cxutil.CvxObjects=None
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    
    # Grasp generation
    grasp_valid_mask = jnp.zeros((n_grasps,)).astype(int)
    grasp_scores = jnp.zeros((n_grasps,))
    prand = jnp.zeros((n_grasps, 3))
    qrand = jnp.zeros((n_grasps, 4))
    state = (prand, qrand, grasp_valid_mask, grasp_scores, jkey)

    # Generation while loop...
    condition = partial(__generate_grasp_condition, min_n_valid=min_n_valid)
    body      = partial(__generate_grasp_with_colocc_body, 
                        n_grasps = n_grasps,
                        target_obj = target_obj, 
                        default_hand_objs = default_hand_objs, 
                        models = models,
                        env_type=env_type,
                        env_obj = env_obj)
    prand, qrand, grasp_valid_mask, grasp_scores, jkey = jax.lax.while_loop(condition, body, state)

    return prand, qrand, grasp_scores



def generate_grasp_candidates_grasp_region_filter(
        jkey: jax.Array, 
        target_obj: cxutil.LatentObjects, 
        default_hand_objs: cxutil.LatentObjects, 
        n_grasps: int, 
        models: flax.linen.Module,
        min_n_valid: int=20,
        env_type: str='table',
        plane_params:jnp.ndarray=None,
        env_obj: cxutil.LatentObjects=None,
        non_selected_target_objs: cxutil.LatentObjects=None,
        robust_grasp_nsample:int=0,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    
    # Grasp generation
    grasp_valid_mask = jnp.zeros((n_grasps,)).astype(int)
    # grasp_scores = jnp.zeros((n_grasps,))
    prand = jnp.zeros((n_grasps, 3))
    qrand = jnp.zeros((n_grasps, 4))
    state = (prand, qrand, grasp_valid_mask, None, 0, jkey)

    # Generation while loop...
    condition = partial(__generate_grasp_condition, min_n_valid=min_n_valid)
    body      = partial(__generate_grasp_with_occ_grasp_region, 
                        n_grasps = n_grasps,
                        target_obj = target_obj, 
                        env_type=env_type,
                        plane_params=plane_params,
                        models = models)
    prand, qrand, grasp_region_valid_mask, _, _, jkey = jax.lax.while_loop(condition, body, state)
    jkey, subkey = jax.random.split(jkey)
    # reduce size
    prand, qrand, grasp_region_valid_mask = jax.tree_util.tree_map(lambda x: x[:min_n_valid], (prand, qrand, grasp_region_valid_mask))
    if non_selected_target_objs is not None:
        target_obj = target_obj.concat(non_selected_target_objs, axis=0)
        col_valid_mask = jax.vmap(partial(get_grasp_collision_check, hand_p=prand, hand_q=qrand, grasp_valid_mask=grasp_region_valid_mask, models=models, default_hand_objs=default_hand_objs, jkey=jkey, env_obj=env_obj))(target_obj)
        grasp_scores = jax.vmap(partial(get_grasp_scores, hand_p=prand, hand_q=qrand, models=models, jkey=subkey, robust_grasp_nsample=robust_grasp_nsample))(target_obj, grasp_valid_mask=jnp.logical_and(grasp_region_valid_mask, col_valid_mask))
        grasp_scores = jnp.mean(grasp_scores, 0)
    else:
        col_valid_mask = get_grasp_collision_check(target_obj, prand, qrand, grasp_region_valid_mask, models, default_hand_objs, jkey, env_obj)
        grasp_scores = get_grasp_scores(target_obj, prand, qrand, jnp.logical_and(grasp_region_valid_mask, col_valid_mask), models, jkey=subkey, robust_grasp_nsample=robust_grasp_nsample)

    return prand, qrand, grasp_scores


def __generate_grasp_condition(state: Tuple, min_n_valid: int):
    """More than min_n_valid valid grasps"""
    prand, qrand, mask, scores, itr, key = state
    return jnp.logical_and(jnp.sum(mask) < min_n_valid, itr<50)

def __generate_grasp_with_colocc_body(
        state: Tuple, 
        n_grasps: int, 
        target_obj: cxutil.CvxObjects,
        default_hand_objs: cxutil.CvxObjects,
        models: mutil.Models,
        env_type:str,
        env_obj=None
) -> Tuple:
    """????"""
    prand, qrand, mask, _, jkey = state

    # Yaw randomization only
    jkey, subkey = jax.random.split(jkey)
    x = jnp.zeros((n_grasps,))
    y = jnp.zeros((n_grasps,))
    z = jnp.pi * jax.random.uniform(subkey, (n_grasps,))
    z_rot_axisangles = jnp.stack((x,y,z), axis=1)
    x_180 = jnp.array([1.,0.,0.,0.]) # Rotate inward
    q_yaw = tutil.qmulti(x_180, tutil.aa2q(z_rot_axisangles))

    if env_type == 'shelf':
        cabinet_out_dir = tutil.aa2q(jnp.array([0, -jnp.pi/8, 0]))
        q_yaw = tutil.qmulti(tutil.qmulti(cabinet_out_dir, x_180), tutil.aa2q(z_rot_axisangles))

    # Orientation randommization by maximum 0.5 * pi
    jkey, subkey1, subkey2 = jax.random.split(jkey, 3)
    axes = jax.random.uniform(subkey1, (n_grasps, 3))
    axes = tutil.normalize(axes)
    angles =  jnp.pi/8 * jax.random.uniform(subkey2, (n_grasps,1))
    axisangles = angles * axes
    new_qrand = tutil.qmulti(q_yaw, tutil.aa2q(axisangles))

    # new_qrand = q_yaw

    # Generate random positions of ee-base
    jkey, subkey = jax.random.split(jkey)
    new_prand = target_obj.pos \
              + jnp.array([0.16, 0.16, 0.2]) * jax.random.uniform(subkey, (n_grasps, 3))\
              - jnp.array([0.08, 0.08, 0.0])
    
    # new_prand = target_obj.pos \
    #           + jnp.array([0.08, 0.16, 0.2]) * jax.random.uniform(subkey, (n_grasps, 3))\
    #           - jnp.array([0.08, 0.08, 0.0])
    
    prand_app = new_prand
    qrand_app = new_qrand

    # prand_app = jnp.array([-0.11549459,  0.12326922,  0.2105391 ], dtype=jnp.float32)[None]
    # qrand_app = jnp.array([ 0.99751717, -0.06753471,  0.01955359, -0.00403331], dtype=jnp.float32)[None]

    # Generate n=N_GRASPS transformed latent objects
    batched_transf = jax.vmap(apply_transf, in_axes=(None, 0, 0, None))
    # collision_check_handobjs = jax.tree_map(lambda x: x[jnp.array([0,3])], default_hand_objs) # gripper hand and box
    collision_check_handobjs = jax.tree_map(lambda x: x[jnp.array([0])], default_hand_objs) # gripper hand only
    transf_hand_objs = batched_transf(collision_check_handobjs, prand_app, qrand_app, models.rot_configs) # [n_grasp, 2]

    occ_check_pos = fkutil.get_gripper_end_AABB_points(scale=jnp.array([1.2,1.2,1.1]))
    occ_check_pos = tutil.pq_action(prand_app[:,None,None,:], qrand_app[:,None,None,:], occ_check_pos[None]) # [n_grasp,2,14,3]

    if env_obj is None:
        # gripper tip collision test
        occ_res = models.apply('occ_predictor', target_obj, occ_check_pos.reshape(-1, 3), jkey) # [n_grasp, 4, 16]
        occ_res = occ_res.reshape(n_grasps, 2, -1)
        occ_res = jnp.max(occ_res, axis=-1) # [n_grasp, 2]
        tip_occ = jnp.max(occ_res, axis=-1) # [n_grasp]
        # tip_validity = tip_occ < -1.
        tip_validity = tip_occ < 0.

        target_obj_repeated = target_obj.extend_and_repeat_outer_shape(n_grasps, axis=0)    # [n_grasp, 1]
        # Collision between target object and transformed hand
        col_res = models.pairwise_collision_prediction(transf_hand_objs, target_obj_repeated, jkey)[:, :, 0] # [n_grasp, 2, 1]
    
    else:
        # gripper tip collision test
        occ_res = models.apply('occ_predictor', env_obj, occ_check_pos.reshape(-1, 3), jkey) # [n_grasp, 4, 16]
        n_env_obj = env_obj.outer_shape[0]
        occ_res = occ_res.reshape(n_env_obj, n_grasps, 2, -1) # [n_env_obj, ...]
        occ_res = jnp.max(occ_res, axis=-1) # [n_env_obj, n_grasp, 2]
        tip_occ = jnp.max(occ_res, axis=-1) # [n_env_obj, n_grasp]
        tip_occ = jnp.max(tip_occ, axis=0) # [n_grasp]
        # tip_validity = tip_occ < -1.
        tip_validity = tip_occ < 0.

        target_obj_repeated = env_obj.extend_and_repeat_outer_shape(n_grasps, axis=0)    # [n_grasp, n_obj]
        # Collision between target object and transformed hand
        col_res = models.pairwise_collision_prediction(transf_hand_objs, target_obj_repeated, jkey) # [n_grasp, 2, n_obj]
        col_res = jnp.max(col_res, axis=-1)

    

    # Filter grasps
    # grasp_valid_mask = jnp.logical_and(col_res[...,0] < 0, col_res[...,1] > 0)
    grasp_valid_mask = col_res[...,0] < 1.0
    grasp_valid_mask = jnp.array(grasp_valid_mask, dtype=int)

    # collision region valid check
    finger_end_pos, finger_end_quat = fkutil.get_gripper_center_from_ee_pq((prand_app, qrand_app))
    finger_end_R = tutil.q2R(finger_end_quat)
    finger_end_y_dir = finger_end_R[..., 1]
    finger_end_1 = finger_end_pos + 0.04*finger_end_y_dir
    finger_end_2 = finger_end_pos - 0.04*finger_end_y_dir
    colregion_check_pos = finger_end_1[:,None,:] + jnp.linspace(0, 1, 50)[...,None]*(finger_end_2[:,None,:] - finger_end_1[:,None,:])
    colrg_occ_res = models.apply('occ_predictor', target_obj, colregion_check_pos.reshape(-1, 3), jkey)
    colrg_occ_res = colrg_occ_res.reshape(n_grasps, -1)
    colrg_occ = jnp.max(colrg_occ_res, axis=-1) # [n_grasp]
    colrg_validity = colrg_occ > -1
    grasp_valid_mask = jnp.logical_and(grasp_valid_mask, colrg_validity)

    validity_origin = jnp.logical_and(tip_validity, grasp_valid_mask)

    # validation stack
    sort_idx = jnp.argsort(validity_origin.astype(jnp.int32), axis=0)
    pqrand = jnp.c_[prand, qrand]
    new_pqrand = jnp.c_[new_prand, new_qrand]
    new_pqrand = new_pqrand[sort_idx]
    pqrand = jnp.where(mask[...,None], pqrand, new_pqrand)
    mask = jnp.logical_or(mask, validity_origin[sort_idx]).astype(jnp.int32)

    # reordering
    sort_idx = jnp.argsort(-mask, axis=0)
    pqrand = pqrand[sort_idx]
    mask = mask[sort_idx]
    pqrand = jnp.where(mask[...,None], pqrand, 0)

    # grasp_scores = get_grasp_scores(target_obj, prand, qrand, mask, models)
    grasp_scores = get_grasp_scores(target_obj, pqrand[...,:3], pqrand[...,3:], mask, models)
    mask = jnp.where(grasp_scores==0, 0, 1)
    
    return pqrand[...,:3], pqrand[...,3:], mask, grasp_scores, jkey



def __generate_grasp_with_occ_grasp_region(
        state: Tuple, 
        n_grasps: int, 
        target_obj: cxutil.CvxObjects,
        env_type:str,
        models: mutil.Models,
        plane_params:jnp.ndarray=None,
) -> Tuple:
    """????"""
    prand, qrand, mask, _, itr, jkey = state

    # x-axis: franka forward direction

    # Yaw randomization only
    jkey, subkey = jax.random.split(jkey)
    x_180 = jnp.array([1.,0.,0.,0.]) # Rotate inward # z-axis down

    # Orientation randommization by maximum 0.5 * pi
    jkey, subkey1, subkey2 = jax.random.split(jkey, 3)
    if env_type == 'shelf':
        z_euler_ang = jnp.array([0,0,1]) * jax.random.uniform(subkey, (n_grasps,1), minval=-np.pi/6, maxval=np.pi/6)
        y_euler_ang = jnp.array([0,1,0]) * jax.random.uniform(subkey1, (n_grasps,1), minval=np.pi/5, maxval=np.pi/2)
        # x_euler_ang = jnp.array([1,0,0]) * jax.random.uniform(subkey2, (n_grasps,1), minval=-np.pi/5, maxval=np.pi/5)
        z_euler_ang2 = jnp.array([0,0,1]) * jax.random.uniform(subkey2, (n_grasps,1), minval=-np.pi/5, maxval=np.pi/5)
        new_qrand = tutil.qmulti(x_180, tutil.aa2q(z_euler_ang))
        new_qrand = tutil.qmulti(new_qrand, tutil.aa2q(y_euler_ang))
        # new_qrand = tutil.qmulti(new_qrand, tutil.aa2q(x_euler_ang))
        new_qrand = tutil.qmulti(new_qrand, tutil.aa2q(z_euler_ang2))
    elif env_type == 'table':
        z_euler_ang_1 = jnp.array([0,0,1]) * jax.random.uniform(subkey1, (n_grasps,1), minval=-np.pi, maxval=np.pi)
        # z_euler_ang_1 = jnp.array([0,0,1]) * jax.random.uniform(subkey1, (n_grasps,1), minval=-np.pi/5., maxval=np.pi/5.)
        # z_euler_ang_1 = jnp.array([0,0,1]) * jax.random.uniform(subkey1, (n_grasps,1), minval=-np.pi/4., maxval=np.pi/4.)
        x_euler_ang = jnp.array([1,0,0]) * jax.random.uniform(subkey2, (n_grasps,1), minval=-np.pi/6, maxval=np.pi/6)
        # y_euler_ang_2 = jnp.array([0,1,0]) * jax.random.uniform(subkey, (n_grasps,1), minval=-np.pi/8, maxval=np.pi/8)
        y_euler_ang_2 = jnp.array([0,1,0]) * jax.random.uniform(subkey, (n_grasps,1), minval=-np.pi/6, maxval=np.pi/6)
        new_qrand = tutil.qmulti(x_180, tutil.aa2q(z_euler_ang_1))
        new_qrand = tutil.qmulti(new_qrand, tutil.aa2q(x_euler_ang))
        new_qrand = tutil.qmulti(new_qrand, tutil.aa2q(y_euler_ang_2))
    elif env_type == 'side':
        z_euler_ang_1 = jnp.array([0,0,1]) * jax.random.uniform(subkey1, (n_grasps,1), minval=-np.pi, maxval=np.pi)
        x_euler_ang = jnp.array([1,0,0]) * jax.random.uniform(subkey2, (n_grasps,1), minval=-np.pi/20+np.pi/2.0, maxval=np.pi/20+np.pi/2.0)
        z_euler_ang_2 = jnp.array([0,0,1]) * jax.random.uniform(subkey, (n_grasps,1), minval=-np.pi, maxval=np.pi)
        new_qrand = tutil.qmulti(x_180, tutil.aa2q(z_euler_ang_1))
        new_qrand = tutil.qmulti(new_qrand, tutil.aa2q(x_euler_ang))
        new_qrand = tutil.qmulti(new_qrand, tutil.aa2q(z_euler_ang_2))
    elif env_type == 'approach':
        z_euler_ang_1 = jnp.array([0,0,1]) * jax.random.uniform(subkey1, (n_grasps,1), minval=-np.pi, maxval=np.pi)
        x_euler_ang = jnp.array([1,0,0]) * jax.random.uniform(subkey2, (n_grasps,1), minval=-np.pi/10, maxval=np.pi/10)
        y_euler_ang_2 = jnp.array([0,1,0]) * jax.random.uniform(subkey, (n_grasps,1), minval=-np.pi/4, maxval=-np.pi/8)
        new_qrand = tutil.qmulti(x_180, tutil.aa2q(z_euler_ang_1))
        new_qrand = tutil.qmulti(new_qrand, tutil.aa2q(x_euler_ang))
        new_qrand = tutil.qmulti(new_qrand, tutil.aa2q(y_euler_ang_2))

    # Generate random positions of ee-base
    jkey, subkey = jax.random.split(jkey)
    new_prand = target_obj.pos  + jax.random.uniform(subkey, shape=(n_grasps, 3), minval=jnp.array([-0.12, -0.12, 0]), maxval=jnp.array([0.12, 0.12, 0.25]))
    
    occ_check_pnts_last_body = jnp.array([[-0.07,0,-0.18], [0.07,0.,-0.18], [0,0.07,-0.18], [0,-0.07,-0.18],
                                          [-0.07,0,-0.09], [0.07,0.,-0.09], [0,0.07,-0.09], [0,-0.07,-0.09]])
    occ_check_pnts_last_body = tutil.pq_action(new_prand[:,None,:], new_qrand[:,None,:], occ_check_pnts_last_body[None]) # [n_grasp,8,3]

    # collision region valid check
    grasp_len = 0.09
    body_to_gripper_tip_len = 0.04
    interpolation_resolution = 30
    finger_end_pos, finger_end_quat = fkutil.get_gripper_center_from_ee_pq((new_prand, new_qrand))
    finger_end_R = tutil.q2R(finger_end_quat)
    finger_end_y_dir = finger_end_R[..., 1]
    finger_end_z_dir = finger_end_R[..., 2]
    finger_end_1 = finger_end_pos + grasp_len/2*finger_end_y_dir
    finger_end_2 = finger_end_pos - grasp_len/2*finger_end_y_dir
    colregion_check_pnts = finger_end_1[:,None,:] + jnp.linspace(0, 1, interpolation_resolution)[...,None]*(finger_end_2[:,None,:] - finger_end_1[:,None,:])
    colrg_occ_res = models.apply('occ_predictor', target_obj, colregion_check_pnts.reshape(-1, 3), jkey)
    colrg_occ_res = colrg_occ_res.reshape(n_grasps, -1)

    # get contact points
    pnt1_idx = jnp.argmax((colrg_occ_res>0)*1000 + jnp.arange(interpolation_resolution), axis=-1) # (NGRASP,)
    hit_pos_1 = jnp.take_along_axis(colregion_check_pnts, pnt1_idx[...,None,None], -2).squeeze(-2) # (NGRASP, 3)
    pnt2_idx = jnp.argmax((colrg_occ_res>0)*1000 + jnp.arange(interpolation_resolution)[::-1], axis=-1) # (NGRASP,)
    hit_pos_2 = jnp.take_along_axis(colregion_check_pnts, pnt2_idx[...,None,None], -2).squeeze(-2) # (NGRASP, 3)
    hit_pos_stack = jnp.stack([hit_pos_1, hit_pos_2], -2) # (NGRASP, 2, 3)
    grasp_center_pos = jnp.mean(hit_pos_stack, axis=-2) # (NGRASP, 3)

    gripper_center_shift = grasp_center_pos - finger_end_pos

    # approximated tip condition: end points should not be inside of the object
    colrg_occ_tip = jnp.max(colrg_occ_res[...,(0,-1)], axis=-1) # [n_grasp]
    colrg_occ_tip_valid_mask = colrg_occ_tip < 0 # [n_grasp]

    # approximate gripper condition
    colgripper_check_pnts = -finger_end_z_dir[:,None,:]*body_to_gripper_tip_len + colregion_check_pnts
    colgripper_check_pnts = jnp.concatenate([colgripper_check_pnts, occ_check_pnts_last_body], axis=-2)
    colgripper_occ_res = models.apply('occ_predictor', target_obj, colgripper_check_pnts.reshape(-1, 3), jkey)
    colgripper_occ_res = colgripper_occ_res.reshape(n_grasps, -1)
    colgripper_occ_tip = jnp.max(colgripper_occ_res, axis=-1) # [n_grasp]
    colgripper_occ_tip_valid_mask = colgripper_occ_tip < 0 # [n_grasp]

    if plane_params is not None:
        if isinstance(plane_params, tuple) or isinstance(plane_params, list):
            plane_params = plane_params[0]
        occ_check_pnts_gripper_tip = fkutil.get_gripper_end_AABB_points(scale=jnp.array([0.9,0.9,0.9]))
        occ_check_pnts_gripper_tip = tutil.pq_action(new_prand[:,None,None,:], new_qrand[:,None,None,:], occ_check_pnts_gripper_tip[None]) # [n_grasp,2,14,3]
        occ_check_pnts_gripper_tip = occ_check_pnts_gripper_tip.reshape((occ_check_pnts_gripper_tip.shape[0], -1, 3)) # [n_grasp,28,3]
        pln_col_check_qpnts = jnp.concatenate([colgripper_check_pnts, occ_check_pnts_gripper_tip], -2)
        pln_col_res = jnp.all(jnp.sum(pln_col_check_qpnts[...,None,:]*plane_params[...,:3], axis=-1) > plane_params[...,3], axis=-2) # (n_grasp, NPNTS)
        pln_col_valid_mask = jnp.all(pln_col_res, axis=-1)
        pln_col_valid_mask = jnp.where(pln_col_valid_mask, itr<50, True)
        colgripper_occ_tip_valid_mask = jnp.logical_and(colgripper_occ_tip_valid_mask, pln_col_valid_mask)
    # approximate gripper plane contact

    col_occ_valid_mask = jnp.logical_and(colrg_occ_tip_valid_mask, colgripper_occ_tip_valid_mask)

    colrg_occ = jnp.max(colrg_occ_res, axis=-1) # [n_grasp]
    colrg_validity = colrg_occ > -1
    validity_origin = jnp.logical_and(col_occ_valid_mask, colrg_validity)

    # validation stack
    sort_idx = jnp.argsort(validity_origin.astype(jnp.int32), axis=0)
    pqrand = jnp.c_[prand, qrand]
    new_pqrand = jnp.c_[new_prand + gripper_center_shift, new_qrand] # conpensate center shift
    new_pqrand = new_pqrand[sort_idx]
    pqrand = jnp.where(mask[...,None], pqrand, new_pqrand)
    mask = jnp.logical_or(mask, validity_origin[sort_idx]).astype(jnp.int32)

    # reordering
    sort_idx = jnp.argsort(-mask, axis=0)
    pqrand = pqrand[sort_idx]
    mask = mask[sort_idx]
    pqrand = jnp.where(mask[...,None], pqrand, 0)

    return pqrand[...,:3], pqrand[...,3:], mask, None, itr+1, jkey

def get_grasp_collision_check(
        target_obj: cxutil.LatentObjects,
        hand_p: jnp.ndarray,
        hand_q: jnp.ndarray,
        grasp_valid_mask: jnp.ndarray,
        models: mutil.Models,
        default_hand_objs: cxutil.LatentObjects,
        jkey,
        env_obj: cxutil.LatentObjects=None)->jnp.ndarray:
    
    n_grasps = hand_p.shape[0]
    
    if len(target_obj.outer_shape) == 0:
        target_obj = target_obj.extend_outer_shape(axis=0)
    
    # Generate n=N_GRASPS transformed latent objects
    batched_transf = jax.vmap(apply_transf, in_axes=(None, 0, 0, None))
    # collision_check_handobjs = jax.tree_map(lambda x: x[jnp.array([0,3])], default_hand_objs) # gripper hand and box
    collision_check_handobjs = jax.tree_util.tree_map(lambda x: x[jnp.array([0])], default_hand_objs) # gripper hand only
    transf_hand_objs = batched_transf(collision_check_handobjs, hand_p, hand_q, models.rot_configs) # [n_grasp, 2]

    occ_check_pos = fkutil.get_gripper_end_AABB_points(scale=jnp.array([1.2,1.2,1.1]))
    occ_check_pos = tutil.pq_action(hand_p[:,None,None,:], hand_q[:,None,None,:], occ_check_pos[None]) # [n_grasp,2,14,3]

    if env_obj is None:
        # gripper tip collision test
        occ_res = models.apply('occ_predictor', target_obj, occ_check_pos.reshape(-1, 3), jkey) # [n_grasp, 4, 16]
        occ_res = occ_res.reshape(n_grasps, 2, -1)
        occ_res = jnp.max(occ_res, axis=-1) # [n_grasp, 2]
        tip_occ = jnp.max(occ_res, axis=-1) # [n_grasp]
        # tip_validity = tip_occ < -1.
        tip_validity = tip_occ < 0.

        target_obj_repeated = target_obj.extend_and_repeat_outer_shape(n_grasps, axis=0)    # [n_grasp, 1]
        # Collision between target object and transformed hand
        col_res = models.pairwise_collision_prediction(transf_hand_objs, target_obj_repeated, jkey)[:, :, 0] # [n_grasp, 2, 1]
    
    else:
        # gripper tip collision test
        occ_res = models.apply('occ_predictor', env_obj, occ_check_pos.reshape(-1, 3), jkey) # [n_grasp, 4, 16]
        n_env_obj = env_obj.outer_shape[0]
        occ_res = occ_res.reshape(n_env_obj, n_grasps, 2, -1) # [n_env_obj, ...]
        occ_res = jnp.max(occ_res, axis=-1) # [n_env_obj, n_grasp, 2]
        tip_occ = jnp.max(occ_res, axis=-1) # [n_env_obj, n_grasp]
        tip_occ = jnp.max(tip_occ, axis=0) # [n_grasp]
        # tip_validity = tip_occ < -1.
        tip_validity = tip_occ < 0.

        target_obj_repeated = env_obj.extend_and_repeat_outer_shape(n_grasps, axis=0)    # [n_grasp, n_obj]
        # Collision between target object and transformed hand
        col_res = models.pairwise_collision_prediction(transf_hand_objs, target_obj_repeated, jkey) # [n_grasp, 2, n_obj]
        col_res = jnp.max(col_res, axis=-1)
    

    # Filter grasps
    # grasp_valid_mask = jnp.logical_and(col_res[...,0] < 0, col_res[...,1] > 0)
    col_mask = col_res[...,0] < 1.0
    col_mask = jnp.array(col_mask, dtype=int)

    # # collision region valid check
    # finger_end_pos, finger_end_quat = fkutil.get_gripper_center_from_ee_pq((hand_p, hand_q))
    # finger_end_R = tutil.q2R(finger_end_quat)
    # finger_end_y_dir = finger_end_R[..., 1]
    # finger_end_1 = finger_end_pos + 0.04*finger_end_y_dir
    # finger_end_2 = finger_end_pos - 0.04*finger_end_y_dir
    # colregion_check_pos = finger_end_1[:,None,:] + jnp.linspace(0, 1, 50)[...,None]*(finger_end_2[:,None,:] - finger_end_1[:,None,:])
    # colrg_occ_res = models.apply('occ_predictor', target_obj, colregion_check_pos.reshape(-1, 3), jkey)
    # colrg_occ_res = colrg_occ_res.reshape(n_grasps, -1)
    # colrg_occ = jnp.max(colrg_occ_res, axis=-1) # [n_grasp]
    # colrg_validity = colrg_occ > -1
    # grasp_valid_mask = jnp.logical_and(grasp_valid_mask, colrg_validity)

    col_mask = jnp.logical_and(tip_validity, col_mask)
    mask = jnp.logical_and(col_mask, grasp_valid_mask)

    # # validation stack
    # sort_idx = jnp.argsort(validity_origin.astype(jnp.int32), axis=0)
    # pqrand = jnp.c_[prand, qrand]
    # new_pqrand = jnp.c_[new_prand, new_qrand]
    # new_pqrand = new_pqrand[sort_idx]
    # pqrand = jnp.where(mask[...,None], pqrand, new_pqrand)
    # mask = jnp.logical_or(mask, validity_origin[sort_idx]).astype(jnp.int32)

    # # reordering
    # sort_idx = jnp.argsort(-mask.astype(jnp.int32), axis=0)
    # pqrand = jnp.c_[hand_p, hand_q][sort_idx]
    # mask = mask[sort_idx]
    # pqrand = jnp.where(mask[...,None], pqrand, 0)

    return mask
    


def get_grasp_scores(
        target_obj: cxutil.CvxObjects,
        hand_p: jnp.ndarray,
        hand_q: jnp.ndarray,
        grasp_valid_mask: cxutil.CvxObjects,
        models: flax.linen.Module,
        jkey: jnp.ndarray,
        robust_grasp_nsample: int=50,
) -> jnp.ndarray:
    if len(target_obj.outer_shape) == 0:
        target_obj = target_obj.extend_outer_shape(0)
    

    # Grasp score check
    finger_end_pos, finger_end_quat = fkutil.get_gripper_center_from_ee_pq((hand_p, hand_q))
    if robust_grasp_nsample!=0 and robust_grasp_nsample is not None:
        # add noise to measure fobustness
        origin_p_shape = finger_end_pos.shape
        p_noise_scale = 0.008
        ang_noise_scale = 4/180 * np.pi
        jkey, subkey = jax.random.split(jkey)
        finger_end_pos = finger_end_pos[...,None,:] + p_noise_scale*jax.random.normal(subkey, shape=finger_end_pos.shape[:-1] + (robust_grasp_nsample,3))
        jkey, subkey = jax.random.split(jkey)
        finger_end_quat = tutil.qmulti(finger_end_quat[...,None,:], tutil.aa2q(ang_noise_scale*jax.random.normal(subkey, shape=finger_end_quat.shape[:-1] + (robust_grasp_nsample,3))))
        finger_end_pos, finger_end_quat = jax.tree_util.tree_map(lambda x: einops.rearrange(x, 'i j ... -> (i j) ...'), (finger_end_pos, finger_end_quat))
        grasp_valid_mask = einops.repeat(grasp_valid_mask, 'i ... -> (i r) ...', r=robust_grasp_nsample)
    finger_end_R = tutil.q2R(finger_end_quat)
    finger_end_y_dir = finger_end_R[..., 1]

    # # From these finger_end, ray is casted toward the object center
    # finger_end_1 = finger_end_pos + 0.05*finger_end_y_dir
    # finger_end_2 = finger_end_pos - 0.05*finger_end_y_dir
    finger_end_1 = finger_end_pos + 0.04*finger_end_y_dir
    finger_end_2 = finger_end_pos - 0.04*finger_end_y_dir

    int_pnts = finger_end_1[:,None] + jnp.linspace(0,1,80)[:,None]*(finger_end_2 - finger_end_1)[:,None]
    int_occ_res = models.apply('occ_predictor', target_obj, int_pnts.reshape(-1, 3), jax.random.PRNGKey(0))
    int_occ_res = int_occ_res.reshape(target_obj.outer_shape[0], int_pnts.shape[0], int_pnts.shape[1]) # (1, NS, 80)
    pnt1_idx = jnp.argmax((int_occ_res>0)*1000 + jnp.arange(int_pnts.shape[1]), axis=-1)
    hit_pos_1 = jnp.take_along_axis(int_pnts[None], pnt1_idx[...,None,None], -2).squeeze(-2)
    pnt2_idx = jnp.argmax((int_occ_res>0)*1000 + jnp.arange(int_pnts.shape[1])[::-1], axis=-1)
    hit_pos_2 = jnp.take_along_axis(int_pnts[None], pnt2_idx[...,None,None], -2).squeeze(-2)
    hit_pos_stack = jnp.stack([hit_pos_1, hit_pos_2], 1)
    origin_shape = hit_pos_stack.shape
    raw_normal_12 = jax.grad(lambda x: jnp.sum(models.apply('occ_predictor', target_obj, x, jax.random.PRNGKey(0))))(hit_pos_stack.reshape(-1,3))
    raw_normal_12 = raw_normal_12.reshape(origin_shape)
    raw_normal_1, raw_normal_2 = raw_normal_12[:,0], raw_normal_12[:,1]

    center_pos = target_obj.pos
    # ?
    hit_pos_1, raw_normal_1, hit_pos_2, raw_normal_2, center_pos, int_occ_res \
        = jax.tree_util.tree_map(
            lambda x: jnp.squeeze(x,0), 
            (hit_pos_1, raw_normal_1, hit_pos_2, raw_normal_2, center_pos, int_occ_res))

    # end point collision mask
    gripper_tip_collision_mask = jnp.logical_and(int_occ_res[...,0] < 0, int_occ_res[...,-1] < 0) # (NS,)

    # Raw_normal means the normal vectors on the hit_pos
    raw_normal_1 = tutil.normalize(raw_normal_1)
    raw_normal_2 = tutil.normalize(raw_normal_2)

    # Similarity metric (force closure)
    # Calculate cosine similarity between the gripper direction (intersect) and surface normal direction
    cos_sim_1 = jnp.abs(jnp.sum(raw_normal_1*finger_end_y_dir, axis=-1)) # cosine similarity
    cos_sim_2 = jnp.abs(jnp.sum(raw_normal_2*finger_end_y_dir, axis=-1))
    similarity = jnp.where(cos_sim_1<cos_sim_2, cos_sim_1, cos_sim_2)
    valid_similarity_mask = similarity > jnp.cos(30/180.*np.pi)
    # similarity = tutil.normalize(similarity)

    # Intersect means the vector between two ray hitted grasping points on the surface
    intersect = hit_pos_2 - hit_pos_1

    # Validity metric
    # If the gripping width is wider than 8cm, it cannot be grasped.
    intersect_len = jnp.linalg.norm(intersect, axis=-1)
    width_validity_1 = jnp.where(intersect_len<0.075, 1, 0)
    width_validity_2 = jnp.where(intersect_len>0.002, 1, 0)
    # ?
    finger_contact_pnt = (hit_pos_2 + hit_pos_1)*0.5
    # torque = jnp.linalg.norm(jnp.cross((finger_end_pos[:, :] - center_pos[None, :]), jnp.array([0., 0., -1.])), axis=-1)
    torque = jnp.linalg.norm(jnp.cross((finger_contact_pnt - center_pos[None, :]), jnp.array([0., 0., -1.])), axis=-1)
    torque = tutil.normalize(torque)

    center_dist_reward = jnp.exp(-jnp.sum((finger_contact_pnt - finger_end_pos)**2,-1))
    center_dist_reward = tutil.normalize(center_dist_reward)
    # ?
    # total_score = grasp_valid_mask * width_validity * similarity * (1./torque)
    # total_score = grasp_valid_mask * width_validity_1 * width_validity_2 * (similarity + center_dist_reward)
    total_score = grasp_valid_mask * width_validity_1 * width_validity_2 * gripper_tip_collision_mask * valid_similarity_mask *similarity

    if robust_grasp_nsample!=0 and robust_grasp_nsample is not None:
        total_score = einops.rearrange(total_score, '(r i) ... -> r i ...', r=origin_p_shape[0])
        total_score = jnp.mean(total_score, 1)

    return total_score


def grasp_pose_from_latent_considering_collision(models:mutil.Models, target_obj, robot_base_pos, robot_base_quat, init_q, jkey, 
                                                 env_objs=None, approach_offset=jnp.array([0,0,-0.05]),
                            n_grasps_parall_computation=100, n_grasps_output=50, n_grasps_reachability_check=10, robust_sample_no=50,
                            plane_params=None, env_type='table', deeper_depth=0.0, non_selected_obj_pred:cxutil.LatentObjects=None,
                            vis=False):

    if len(target_obj.outer_shape) == 0:
        target_obj = jax.tree_map(lambda x: x[None], target_obj)
    default_hand_objs, panda_link_obj, inside_latent_obj = fkutil.get_default_hand_latent_objs(models)

    if non_selected_obj_pred is not None:
        # print("using uncertainty grasp")
        # idx_mask = jnp.arange(inf_aux_info.x_pred.outer_shape[0]) != inf_aux_info.obs_max_idx
        # idx_where = jnp.where(idx_mask)
        # non_selected_obj_pred = jax.tree_util.tree_map(lambda x: x[idx_where], inf_aux_info.x_pred)
        # if 'implicit_baseline' in models.dif_args and models.dif_args.implicit_baseline:
        #     jkey, subkey = jax.random.split(jkey)
        #     non_selected_obj_pred = non_selected_obj_pred.register_pcd_from_latent(models, models.npoint, subkey)
        
        # instance_seg = inf_aux_info.instance_seg # (NS NO NI NJ)

        # obs_sort_idx = jnp.argsort(-inf_aux_info.obs_preds, axis=-1)
        # obj_pred_sorted, instance_seg_sorted = jax.tree_util.tree_map(lambda x:x[obs_sort_idx], (inf_aux_info.x_pred, inf_aux_info.instance_seg))
        # # instance_seg_sorted # (NS NO NI NJ)
        # target_obj_idx = jnp.argmin(jnp.linalg.norm(obj_pred_sorted.pos[0] - target_obj.pos, axis=-1))

        # iou_non_selected_objs = jnp.sum(instance_seg_sorted[0,target_obj_idx] * instance_seg_sorted[1:,], axis=(-1,-2)) # (NS-1, NO)
        # iou_non_selected_objs = iou_non_selected_objs/(jnp.sum(instance_seg_sorted[0,target_obj_idx] + instance_seg_sorted[1:,], axis=(-1,-2)) - iou_non_selected_objs) # (NS-1, NO)
        # non_selected_obj_pred = jax.tree_util.tree_map(lambda x: x[1:],obj_pred_sorted)

        nobj = non_selected_obj_pred.outer_shape[-1]
        distance_to_obj = jnp.linalg.norm(non_selected_obj_pred.pos - target_obj.pos, axis=-1)
        target_idx_in_non_selected_obj_pred = jnp.argmin(distance_to_obj, axis=-1) # (NS,)
        min_distance_to_obj = jnp.min(distance_to_obj, axis=-1)
        non_selected_target_objs = non_selected_obj_pred.take_along_outer_axis(target_idx_in_non_selected_obj_pred, 1) # (NS, 1)
        non_selected_target_objs = non_selected_target_objs.squeeze_outer_shape(1)
        valid_distance_threshold = 0.035
        non_selected_target_objs = non_selected_target_objs.replace(pos=jnp.where(min_distance_to_obj[...,None] < valid_distance_threshold, non_selected_target_objs.pos, jnp.array([0,0,10.])))
        # non_selected_env_idx = jnp.where(jnp.arange(nobj-1) >= target_idx_in_non_selected_obj_pred[...,None], jnp.arange(nobj-1)+1, jnp.arange(nobj-1))
        # non_selected_env_objs = non_selected_obj_pred.take_along_outer_axis(non_selected_env_idx, 1)
        # non_selected_env_objs = jax.tree_util.tree_map(lambda x: jnp.reshape(x, (-1,)+x.shape[2:]), non_selected_env_objs)
        # env_objs = env_objs.concat(non_selected_env_objs, axis=0)
    else:
        non_selected_target_objs = None
    
    if robot_base_pos is not None:
        hand_p, hand_q, grasp_scores = generate_grasp_candidates_grasp_region_filter(jkey, target_obj, default_hand_objs.drop_gt_info(color=True), 
                                                                                    n_grasps_parall_computation, models, n_grasps_output,
                                                                                    plane_params=plane_params, env_type=env_type, 
                                                                                    robust_grasp_nsample=robust_sample_no,
                                                                                    non_selected_target_objs=non_selected_target_objs)
        sorted_idx = jnp.flip(jnp.argsort(grasp_scores))
        grasp_scores = grasp_scores[sorted_idx[:n_grasps_reachability_check]]
        best_pq = (hand_p[sorted_idx[:n_grasps_reachability_check]], hand_q[sorted_idx[:n_grasps_reachability_check]])
    else:
        hand_p, hand_q, grasp_scores = generate_grasp_candidates_grasp_region_filter(jkey, target_obj, default_hand_objs.drop_gt_info(color=True), 
                                                                                    n_grasps_parall_computation, models, n_grasps_output,
                                                                                    plane_params=plane_params, env_type=env_type, 
                                                                                    robust_grasp_nsample=robust_sample_no,
                                                                                    env_obj=env_objs, non_selected_target_objs=non_selected_target_objs)
        best_pq = (hand_p, hand_q)
    approaching_pq = (best_pq[0] + jnp.einsum('...ij,j',tutil.q2R(best_pq[1]), approach_offset), best_pq[1])

    # FINETUNING FOR REAL GRASPING: DEEPER GRASP
    best_pq = (best_pq[0] + tutil.q2R(best_pq[1])[..., :, 2] * deeper_depth, best_pq[1])

    if robot_base_pos is not None:
        best_joint_pos, (pos_cost, quat_cost) = jax.vmap(partial(fkutil.Franka_IK_numetrical, robot_base_pos=robot_base_pos, robot_base_quat=robot_base_quat,
                                                                itr_no = 300, output_cost=True, grasp_basis=False, 
                                                                compensate_last_joint=True), (None,0))(init_q, best_pq)
        approaching_joint_pos, (approaching_pos_cost, approaching_quat_cost) =\
            jax.vmap(partial(fkutil.Franka_IK_numetrical, robot_base_pos=robot_base_pos, robot_base_quat=robot_base_quat,
                            itr_no=300, output_cost=True, grasp_basis=False, compensate_last_joint=True), (0,0))(best_joint_pos, approaching_pq)

        # # flip gripper with symmetricity
        # grasp_flip_q = jnp.stack([best_joint_pos,
        #         best_joint_pos.at[:,-1].add(-np.pi),
        #         best_joint_pos.at[:,-1].add(np.pi)], axis=0)
        # min_idx = jnp.argmin(jnp.linalg.norm(grasp_flip_q - init_q, axis=-1), 0)
        # add_ang = jnp.array([0, -np.pi, np.pi])[min_idx] # (NS, )
        # # best_joint_pos_select = grasp_flip_q[min_idx]
        # best_joint_pos = best_joint_pos.at[:,-1].add(add_ang)
        # approaching_joint_pos = approaching_joint_pos.at[:,-1].add(add_ang)
        # best_pq = tutil.pq_multi(*best_pq, jnp.zeros(3), tutil.aa2q(add_ang[:,None]*jnp.array([0,0,1.])))
        # approaching_pq = tutil.pq_multi(*approaching_pq, jnp.zeros(3), tutil.aa2q(add_ang[:,None]*jnp.array([0,0,1.])))

        # reachability
        reachability_mask_grasp = jnp.logical_and(pos_cost < REACHABILITY_POS_LIMIT, quat_cost < REACHABILITY_QUAT_LIMIT)
        reachability_mask_approaching = jnp.logical_and(approaching_pos_cost < REACHABILITY_POS_LIMIT, approaching_quat_cost < REACHABILITY_QUAT_LIMIT)

        reachability_mask = jnp.logical_and(reachability_mask_grasp, reachability_mask_approaching)

    else:
        reachability_mask = jnp.ones_like(best_pq[0][...,0], dtype=bool)

    col_mask = jnp.ones_like(reachability_mask)
    col_cost = 0
    if env_objs is not None and robot_base_pos is not None:
        # object interactions
        camera_body = cxutil.create_box(np.array([0.025, 0.14, 0.03]), 32, 64)
        camera_body = camera_body.set_z_with_models(jax.random.PRNGKey(0), models)
        gripper_width = 0.05
        cchecker = ifutil.SimpleCollisionChecker(models, panda_link_obj.drop_gt_info(color=True), 
                                              env_objs.drop_gt_info(color=True), plane_params, robot_base_pos, robot_base_quat, 
                                              gripper_width, camera_body)
        col_res_grasp, col_cost_grasp = jax.vmap(cchecker.check_q, (None, 0))(jkey, best_joint_pos)
        col_res_approach, col_cost_approach = jax.vmap(cchecker.check_q, (None, 0))(jkey, approaching_joint_pos)
        col_mask = jnp.logical_not(jnp.logical_or(col_res_grasp, col_res_approach))
        col_cost = col_cost_grasp + col_cost_approach

    if vis:
        from examples.visualize_occ import create_mesh_from_latent
        from imm_pb_util.bullet_envs.manipulation import FrankaManipulation
        from imm_pb_util.bullet_envs.env import SimpleEnv

        mesh_pred = create_mesh_from_latent(jkey, models, target_obj.translate(-target_obj.pos), visualize=False)

        o3d.io.write_triangle_mesh('latent_mesh.obj', mesh_pred)
        
        from pybullet_utils.bullet_client import BulletClient
        bc = BulletClient(connection_mode=pb.GUI)

        obj_id = bc.createMultiBody(baseMass=0.0, baseVisualShapeIndex=pb.createVisualShape(pb.GEOM_MESH, fileName='latent_mesh.obj'))
        
        bc.resetBasePositionAndOrientation(obj_id, np.array(target_obj.pos)[0], [0,0,0,1])
        config_file_path = Path(__file__).parent.resolve() / "pb_examples/pb_cfg" / "pb_real_eval_debug.yaml"
        with open(config_file_path, "r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        config["robot_params"]["franka_panda"]["base_pos"] = robot_base_pos
        config["robot_params"]["franka_panda"]["base_orn"] = bc.getEulerFromQuaternion(robot_base_quat)
        robot = FrankaPanda(bc, config)
        manip = FrankaManipulation(bc, robot, config)
        env = SimpleEnv(bc, config, False)

        panda_gripper = PandaGripper(bc, Path("imm_pb_util")/"urdf"/"panda_gripper"/"panda_gripper_visual.urdf")
        panda_gripper.transparent(0.7)
        approaching_gripper = PandaGripper(bc, Path("imm_pb_util")/"urdf"/"panda_gripper"/"panda_gripper_visual.urdf")
        approaching_gripper.transparent(0.7)

        for i in range(n_grasps_reachability_check):
            print(f'grasp trial: {i} // reachability: {reachability_mask[i]} // grasp_scores: {grasp_scores[i]} // cost: {pos_cost[i]}/{quat_cost[i]} // {approaching_pos_cost[i]}/{approaching_quat_cost[i]}')
            # if reachability_mask[i]:
            # _, joint_pos = manip.solve_ik_analytical(best_pq[0][i], tutil.qmulti(best_pq[1][i], tutil.aa2q(jnp.array([0,0,-np.pi/4]))))
            # robot.reset(joint_pos)
            # robot.reset(approaching_joint_pos[i])
            robot.reset(best_joint_pos[i])

            panda_gripper.set_gripper_pose(best_pq[0][i], tutil.qmulti(best_pq[1][i], tutil.aa2q(jnp.array([0,0,-np.pi/4]))))

            approaching_gripper.set_gripper_pose(approaching_pq[0][i], tutil.qmulti(approaching_pq[1][i], tutil.aa2q(jnp.array([0,0,-np.pi/4]))))

            print("hold")
    grasp_entire_end_t = time.time()

    # select best Q
    mask_total = jnp.logical_and(reachability_mask, col_mask)
    # mask_total = jnp.logical_and(mask_total, grasp_scores>1.1)
    max_idx = jnp.argmax(grasp_scores + mask_total - 0.1*col_cost)
    if robot_base_pos is None:
        best_pq_select, approaching_pq_select, grasp_score_select =\
          jax.tree_map(lambda x:x[max_idx], (best_pq, approaching_pq, grasp_scores))
        virtual_base_pq = (best_pq_select[0] - tutil.q2R(best_pq_select[1])[..., 2] * 1.0, best_pq_select[1])
        grasp_aux_info = (grasp_score_select, best_pq, approaching_pq, grasp_scores, max_idx)
        return best_pq_select, approaching_pq_select, virtual_base_pq, grasp_aux_info

    mask_info = (grasp_scores, reachability_mask, col_mask)
    best_joint_pos_select, approaching_joint_pos_select, best_pq_select, approaching_pq_select, mask_info_select =\
          jax.tree_map(lambda x:x[max_idx], (best_joint_pos, approaching_joint_pos, best_pq, approaching_pq, mask_info))
    
    grasp_aux_info = (mask_info_select, best_joint_pos, approaching_joint_pos, best_pq, approaching_pq, max_idx, mask_info)

    return best_joint_pos_select, approaching_joint_pos_select, best_pq_select, approaching_pq_select, grasp_aux_info

def grasp_pose_from_latent_without_robot(models:mutil.Models, target_obj:cxutil.LatentObjects, env_obj:cxutil.LatentObjects, 
                                         jkey, non_selected_obj_pred:cxutil.LatentObjects=None, n_grasps:int=500, robust_sample_no = 50, env_type='table', vis:bool=False):
    '''
    args:
        target_obj: object predictions outer shape - (1, ...)
        env_obj: object predictions outer shape - (NO-1, ...)
        n_grasps: number of output grasp candidates
    
        
    return:
        grasp_pq: (pos, quat) - ((3,), (4,))
        approach_pq: (pos, quat) - ((3,), (4,))
        virtual_base_pq: (pos, quat) - ((3,), (4,))

        
    for the jit,
    grasp_func_jit = jax.jit(partial(grasp_pose_from_latent_without_robot, models, n_grasps=500, vis=False))
    grasp_func_jit(target_obj, env_obj, jkey)
    '''
    # assert len(obj_pred.outer_shape) == 1 # should have object dimension

    n_grasps_parall_computation = int(n_grasps*1.5)
    n_grasps_output = n_grasps

    # fix env as table
    plane_params = jnp.array([[0,0,1,0.]])

    return grasp_pose_from_latent_considering_collision(models, target_obj, None, None, None, jkey, env_objs=env_obj,
                        n_grasps_parall_computation=n_grasps_parall_computation, n_grasps_output=n_grasps_output, 
                        n_grasps_reachability_check=None, plane_params=plane_params, robust_sample_no = robust_sample_no, env_type=env_type, non_selected_obj_pred=non_selected_obj_pred,
                        vis=vis)



def main():

    # Config
    SEED = 0
    BUILD = ioutil.BuildMetadata.from_str("32_64_1_v4")
    PANDA_DIR_PATH = BASEDIR/"data"/"PANDA"/str(BUILD)
    MODEL_DIR_PATH = BASEDIR/"checkpoints"/"pretraining"/"01152024-074410"
    # MODEL_DIR_PATH = BASEDIR/"logs_dif"/"01222024-004518"
    mesh_path = 'data/NOCS_val/32_64_1_v4/bowl-e816066ac8281e2ecf70f9641eb97702.obj'

    base_len = 0.65
    categorical_scale = {'can':1/base_len*0.15, 'bottle':1/base_len*0.15, 'bowl':1/base_len*0.2, 
                                    'camera':1/base_len*0.15, 'laptop':1/base_len*0.3, 'mug':1/base_len*0.12}

    # Random control
    np.random.seed(SEED)
    random.seed(SEED)
    jkey = jax.random.PRNGKey(SEED)

    # Load representation modelfgrasp_pose_from_latent
    with open(os.path.join(str(MODEL_DIR_PATH), 'saved.pkl'), 'rb') as f:
        loaded = pickle.load(f)
    models = loaded['models']

    # with open('logs_realexp_rrt/01302024-180520/traj_data.pkl', 'rb') as f:
    with open('experiment/exp_data/01242024-180446/traj_data.pkl', 'rb') as f:
        traj_data = pickle.load(f)
    target_obj = jax.tree_util.tree_map(lambda x: x[2:3], traj_data['obj_pred'])

    robot_base_pos = traj_data['robot_base_pos']
    robot_base_quat = traj_data['robot_base_quat']
    plane_params = jnp.array([[0,0,1,0.], [0,0,-1,-0.38], [-1,0,0,-0.21], [0,1,0,-0.415], [0,-1,0,-0.415]]) # real shelf params
    res1 = grasp_pose_from_latent_considering_collision(target_obj, models, robot_base_pos, robot_base_quat, jkey, 
                                                        n_grasps_parall_computation=100, n_grasps_output=50, n_grasps_reachability_check=20,
                                                        plane_params=plane_params,
                                                        vis=True, env_type='shelf')
    res2 = grasp_pose_from_latent(target_obj, models, robot_base_pos, robot_base_quat, jkey, n_grasps=100, valid_n_grasps=50, vis=False, pcds=traj_data['pcds'], colors=traj_data['colors'])

    # # Create latent panda
    default_hand_objs, panda_link_obj, inside_latent_obj = fkutil.get_default_hand_latent_objs(models)

    ##### debug visualization ##########
    # valid_pnts = default_hand_objs.vtx_tf[jnp.where(default_hand_objs.vtx_valid_mask.squeeze(-1))]
    # occ_check_pos = fkutil.get_gripper_end_AABB_points(scale=jnp.array([1.1,1.1,1.1])).reshape(-1,3)
    # finger_end_pos, finger_end_quat = fkutil.get_gripper_center_from_ee_pq((np.array([0,0,0]), np.array([0,0,0,1])))
    # finger_end_R = tutil.q2R(finger_end_quat)
    # finger_end_y_dir = finger_end_R[..., 1]
    # finger_end_1 = finger_end_pos + 0.04*finger_end_y_dir
    # finger_end_2 = finger_end_pos - 0.04*finger_end_y_dir
    # colregion_check_pos = finger_end_1[None,:] + jnp.linspace(0, 1, 50)[...,None]*(finger_end_2[None,:] - finger_end_1[None,:])

    # # draw panda
    # from imm_pb_util.bullet_envs.robot import PandaGripper
    # from pybullet_utils.bullet_client import BulletClient
    # import pybullet as pb
    # bc = BulletClient(connection_mode=pb.GUI)
    # panda_gripper = PandaGripper(bc, Path("imm_pb_util")/"urdf"/"panda_gripper"/"panda_gripper_visual.urdf")

    # print(1)


    # import open3d as o3d
    # pcd1 = o3d.geometry.PointCloud()
    # pcd1.points = o3d.utility.Vector3dVector(valid_pnts)
    # pcd1.paint_uniform_color([1,0,0])
    # pcd2 = o3d.geometry.PointCloud()
    # pcd2.points = o3d.utility.Vector3dVector(occ_check_pos)
    # pcd2.paint_uniform_color([0,1,0])
    # pcd3 = o3d.geometry.PointCloud()
    # # pcd3.points = o3d.utility.Vector3dVector(np.stack([finger_end_pos, finger_end_1, finger_end_2]))
    # pcd3.points = o3d.utility.Vector3dVector(colregion_check_pos)
    # pcd3.paint_uniform_color([0,0,1])
    # o3d.visualization.draw_geometries([pcd1, pcd2, pcd3])
    # ##### debug visualization ##########

    # Load a target object to grasp
    jkey, obj_key = jax.random.split(jkey)
    target_color = jnp.array([0,0.9,0.1])

    jkey, zkey = jax.random.split(jkey)
    # Load objects
    target_obj = env_util.create_cvx_objects(jkey, [mesh_path], BUILD, categorical_scale['bowl'])
    # Init z vectors.
    target_obj = target_obj.set_z_with_models(zkey, models, keep_gt_info=True)

    target_obj = env_util.set_color(target_obj, target_color)

    # depend on the object
    default_target_pq = (jnp.array([0, 0, 0]), jnp.array([0.7071068, 0, 0, 0.7071068 ]))
    target_obj = target_obj.apply_pq_z(*default_target_pq, models.rot_configs) # mug cup canonical pose




    with open('traj_data.pkl', 'rb') as f:
        traj_data = pickle.load(f)
    target_obj = jax.tree_map(lambda x: x[2:3], traj_data['obj_pred'])


    #######################

    # from examples.visualize_occ import create_mesh_from_latent
    # mesh_pred = create_mesh_from_latent(jkey, models, target_obj, visualize=True)

    # o3d.io.write_triangle_mesh('latent_mesh.obj', mesh_pred)

    # with open('traj_data.pkl', 'rb') as f:
    #     loaded_data = pickle.load(f)
    
    
    # from pybullet_utils.bullet_client import BulletClient
    # bc = BulletClient(connection_mode=pb.GUI)

    # bc.createMultiBody(baseMass=0.0, basePosition=np.array(target_obj.pos),
    #                                             baseVisualShapeIndex=pb.createVisualShape(pb.GEOM_MESH, fileName='latent_mesh.obj'))
    
    # # robot, manip, env = init_real_domain(bc, config)

    # traj = loaded_data['traj']
    # pcds = loaded_data['pcds']
    # colors = loaded_data['colors']

    # indices = np.random.choice(pcds.shape[0], 100000, replace=False)

    # bc.addUserDebugPoints(pcds[indices], colors[indices])

    # # grasp_pq = (jnp.array([0.02296626, -0.04986086,  0.13384877]), jnp.array([ 0.8228157,  -0.53122866,  0.17292632, -0.10424449]))

    # grasp_pq = (jnp.array([-0.1519148,  -0.15942332,  0.18203075]), jnp.array([ 0.9029448,  -0.37465575,  0.19434744, -0.08094996]))

    # panda_gripper = PandaGripper(bc, Path("imm_pb_util")/"urdf"/"panda_gripper"/"panda_gripper_visual.urdf")
    # panda_gripper.transparent(0.7)
    # panda_gripper.set_gripper_pose(grasp_pq[0], tutil.qmulti(grasp_pq[1], tutil.aa2q(jnp.array([0,0,-np.pi/4]))))

    # # print("hold")

    #######################


    # SDF function
    sdf_ratio = 900
    sdf = partial(rutil.scene_sdf, models=models, sdf_ratio=sdf_ratio)

    # for i in range(10):
    #     # Rendering
    #     camera_pos = jnp.array([0.6-i*0.05,0.15+i*0.05,0.2])

    #     # pixel_size = (400, 400)
    #     # render_scene_oifrdf = jax.jit(partial(rutil.cvx_render_scene, pixel_size=pixel_size, models=models, sdf=sdf, camera_pos=camera_pos))
    #     # rgb_latent = render_scene_oifrdf(target_obj)

    #     rgb_latent = rutil.cvx_render_scene(target_obj, models, sdf=partial(rutil.scene_sdf, models=models), 
    #                               floor_offset=-0.05, 
    #                             #   pixel_size=(500, 500),
    #                               pixel_size=(64, 64),
    #                               target_pos=jnp.array([0,0,0]), 
    #                               camera_pos=camera_pos)
        
    #     plt.figure()
    #     plt.imshow(rgb_latent)
    #     # plt.savefig(f'test_{i}.jpg')

    # Grasp generation
    # gen_grasp_jit = jax.jit(lambda gk, to, dho: generate_grasp_candidates(gk, to, dho, N_GRASPS, models))

    N_GRASPS = 100
    jkey, grasp_key = jax.random.split(jkey)
    # hand_p, hand_q, grasp_valid_mask = generate_grasp_candidates(
    #     grasp_key, target_obj, default_hand_objs, N_GRASPS, models)
    # hand_p, hand_q, grasp_valid_mask = gen_grasp_jit(
    #     grasp_key, target_obj, default_hand_objs)
    # print(f"# valid grasps: {jnp.sum(grasp_valid_mask)}")
    
    # # Grasp evaluation
    # grasp_score_jit = jax.jit(lambda target_obj, hand_p, hand_q, grasp_valid_mask: get_grasp_scores(target_obj, hand_p, hand_q, grasp_valid_mask, models))
    # # grasp_scores = get_grasp_scores(target_obj, hand_p, hand_q, grasp_valid_mask, sdf, models, default_hand_objs)
    # grasp_scores = grasp_score_jit(target_obj, hand_p, hand_q, grasp_valid_mask)
    # sorted_idx = jnp.flip(jnp.argsort(grasp_scores))

    # hand_p, hand_q, grasp_scores = generate_grasp_candidates(
    #     grasp_key, target_obj, default_hand_objs.drop_gt_info(color=True), N_GRASPS, models)
    
    # gen_grasp_jit = jax.jit(lambda gk, to, dho: generate_grasp_candidates(gk, to, dho, N_GRASPS, models))
    gen_grasp_jit = jax.jit(lambda gk, to, dho: generate_grasp_candidates_grasp_region_filter(gk, to, dho, N_GRASPS, models))

    hand_p, hand_q, grasp_scores = gen_grasp_jit(grasp_key, target_obj, default_hand_objs.drop_gt_info(color=True))
    
    sorted_idx = jnp.flip(jnp.argsort(grasp_scores))

    gripper = 0.05 # gripper center offset from hand frame

    pqeh = jnp.array([0,0,0.]), tutil.aa2q(jnp.array([0.,0,-np.pi/4]))
    pqhg = jnp.array([0,0,0.08]), tutil.aa2q(jnp.array([0.,0,0]))                 # This will control inside area 
    pqhlg = jnp.array([0,0,0.0584]), tutil.aa2q(jnp.array([0.,0,0]))
    pqhrg = jnp.array([0,0,0.0584]), tutil.aa2q(jnp.array([0.,0,np.pi]))

    # pose of hand and two fingers w.r.t end-effector frame
    default_pos = jnp.array([0., 0., 0.])
    default_quat = jnp.array([0., 0., 0., 1.])
    default_pq = (default_pos, default_quat)

    hand_pose = tutil.pq_multi(*default_pq, *pqeh)
    hand_pq = []
    hand_pq.append(hand_pose)
    hand_pq.append(tutil.pq_multi(*tutil.pq_multi(*hand_pose, *pqhlg), jnp.array([0,gripper,0]), tutil.aa2q(jnp.array([0.,0,0]))))
    hand_pq.append(tutil.pq_multi(*tutil.pq_multi(*hand_pose, *pqhrg), jnp.array([0,gripper,0]), tutil.aa2q(jnp.array([0.,0,0]))))
    hand_pq.append(tutil.pq_multi(*hand_pose, *pqhg))

    # Rendering
    camera_pos = jnp.array([0.5,0.25,0.3])
    pixel_size = (400, 400)
    render_scene_oifrdf = jax.jit(partial(rutil.cvx_render_scene, pixel_size=pixel_size, models=models, sdf=sdf, camera_pos=camera_pos))
    for i in range(0, 1):
        best_hand_p = hand_p[i] # hand_p[sorted_idx[i]]
        best_hand_q = hand_q[i] # hand_q[sorted_idx[i]]

        print(f'{i} grasp pose: ', best_hand_p, best_hand_q)

        mesh = False

        if mesh:
            # Rendering
            target_o3d_vis = o3d.io.read_triangle_mesh(mesh_path)
            target_o3d_vis.compute_vertex_normals()

            hand_o3d_list = []
            panda_link_order = ['hand', 'finger', 'finger', 'gripper_inside']
            for linkname in panda_link_order:
                filename = str(PANDA_DIR_PATH/f"{linkname}.obj")
                hand_o3d = o3d.io.read_triangle_mesh(filename)
                hand_o3d.compute_vertex_normals()
                hand_o3d_list.append(hand_o3d)

            transformed_o3d_list = []
            target_o3d_vis.scale(categorical_scale['bowl'], np.zeros((3)))
            target_o3d_vis.transform(tutil.pq2H(*default_target_pq))
            transformed_o3d_list.append(target_o3d_vis)

            for link_idx in range(4):
                link_pq = hand_pq[link_idx]
                rand_pq = (best_hand_p, best_hand_q)
                pq = tutil.pq_multi(*rand_pq, *link_pq)

                link_o3d = hand_o3d_list[link_idx]

                link_o3d.transform(tutil.pq2H(*pq))
                transformed_o3d_list.append(link_o3d)

            coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)

            o3d.visualization.draw_geometries([*transformed_o3d_list, coord_frame])


        best_hand_objs = fkutil.transform_ee_from_pq(
            (best_hand_p, best_hand_q), panda_link_obj, inside_latent_obj, models.rot_configs)
        hand_objs_without_inside = jax.tree_map(lambda x: x[:-1], best_hand_objs)
        total_objs_without_inside = jax.tree_map(lambda *x: jnp.concatenate(x), hand_objs_without_inside.drop_gt_info(color=True), target_obj)
        rgb_latent = render_scene_oifrdf(total_objs_without_inside)
        plt.figure()
        plt.imshow(rgb_latent)
        if grasp_scores[sorted_idx[i]]==0.:
            plt.savefig(f'grasp_{i}_fail_pose1.jpg')
        else: plt.savefig(f'grasp_{i}_pose1.jpg')

    camera_pos = jnp.array([-0.5,-0.25,0.3])
    pixel_size = (400, 400)
    render_scene_oifrdf = jax.jit(partial(rutil.cvx_render_scene, pixel_size=pixel_size, models=models, sdf=sdf, camera_pos=camera_pos))
    for i in range(0, 1):
        best_hand_p = hand_p[sorted_idx[i]]
        best_hand_q = hand_q[sorted_idx[i]]
        best_hand_objs = fkutil.transform_ee_from_pq(
            (best_hand_p, best_hand_q), panda_link_obj, inside_latent_obj, models.rot_configs)
        hand_objs_without_inside = jax.tree_map(lambda x: x[:-1], best_hand_objs)
        total_objs_without_inside = jax.tree_map(lambda *x: jnp.concatenate(x), hand_objs_without_inside.drop_gt_info(color=True), target_obj)
        rgb_latent = render_scene_oifrdf(total_objs_without_inside)
        plt.figure()
        plt.imshow(rgb_latent)
        if grasp_scores[sorted_idx[i]]==0.:
            plt.savefig(f'grasp_{i}_fail_pose2.jpg')
        else: plt.savefig(f'grasp_{i}_pose2.jpg')


        #############################

        from examples.visualize_occ import create_mesh_from_latent
        
        mesh_pred = create_mesh_from_latent(jkey, models, target_obj.translate(-target_obj.pos), visualize=True)

        o3d.io.write_triangle_mesh('latent_mesh.obj', mesh_pred)

        with open('traj_data.pkl', 'rb') as f:
            loaded_data = pickle.load(f)
        
        
        from pybullet_utils.bullet_client import BulletClient
        bc = BulletClient(connection_mode=pb.GUI)

        obj_id = bc.createMultiBody(baseMass=0.0, baseVisualShapeIndex=pb.createVisualShape(pb.GEOM_MESH, fileName='latent_mesh.obj'))
        
        bc.resetBasePositionAndOrientation(obj_id, np.array(target_obj.pos)[0], [0,0,0,1])
        # robot, manip, env = init_real_domain(bc, config)

        pcds = loaded_data['pcds']
        colors = loaded_data['colors']

        indices = np.random.choice(pcds.shape[0], 100000, replace=False)

        bc.addUserDebugPoints(pcds[indices], colors[indices])

        panda_gripper = PandaGripper(bc, Path("imm_pb_util")/"urdf"/"panda_gripper"/"panda_gripper_visual.urdf")
        panda_gripper.transparent(0.7)
        panda_gripper.set_gripper_pose(best_hand_p, tutil.qmulti(best_hand_q, tutil.aa2q(jnp.array([0,0,-np.pi/4]))))

        print("hold")

        #############################


    


if __name__=="__main__":
    main()