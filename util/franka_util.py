from __future__ import annotations
import jax.numpy as jnp
import jax
import numpy as np
import os, sys
from typing import List, Tuple, Dict
from pathlib import Path

BASEDIR = Path(__file__).parent.parent
if BASEDIR not in sys.path:
    sys.path.insert(0, str(BASEDIR))
# if __name__ == '__main__':
#     sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import util.transform_util as tutil
from functools import partial
import util.cvx_util as cxutil
import optax

FRANKA_Q_MIN = np.array([-2.3093, -1.5133, -2.4937, -2.7478, -2.4800, 0.8521, -2.6895])
FRANKA_Q_MAX = np.array([2.3093, 1.5133, 2.4937, -0.4461, 2.4800, 3.7094, 2.6895])

def Franka_FK(
        q: jnp.ndarray, 
        gripper: float = 0.05
) -> Tuple[List[Tuple[jnp.ndarray, jnp.ndarray]], Tuple[jnp.ndarray, jnp.ndarray]]:
    '''Franka forward kinematics
    
    Args:
        q (jnp.ndarray): shape=[7,] Ignores finger value currently.
        gripper (float): Grasp center offset from hand joint frame.
    
    Returns:
        link_pq (List[Tuple[jnp.ndarray, jnp.ndarray]]): (pos, quat) of each link volumetric center. List of length 11.
        grasp_center_pq (Tuple[jnp.ndarray, jnp.ndarray]): (pos, quat) of the gripper center.
        
    '''
    # Validate input shape
    if q.ndim != 1:
        raise ValueError("q dim must be 1.")
    if q.shape[0] != 7:
        raise ValueError("q shape must be 7")

    # DH-parameters
    pq01  = jnp.array([0,0,0.3330]),        tutil.aa2q(jnp.array([0.,0,0]))
    pq12  = jnp.array([0,0,0.]),            tutil.aa2q(jnp.array([-np.pi/2.,0,0]))  # Simplified
    pq23  = jnp.array([0,-0.3160,0.]),      tutil.aa2q(jnp.array([np.pi/2.,0,0]))
    pq34  = jnp.array([0.0880,0,0.]),       tutil.aa2q(jnp.array([np.pi/2.,0,0]))
    pq45  = jnp.array([-0.0880,0.3840,0.]), tutil.aa2q(jnp.array([-np.pi/2.,0,0]))
    pq56  = jnp.array([0,0,0.]),            tutil.aa2q(jnp.array([np.pi/2.,0,0]))   # Simplified
    pq67  = jnp.array([0.0880,0,0.]),       tutil.aa2q(jnp.array([np.pi/2.,0,0]))
    pq7e  = jnp.array([0,0,0.1070]),        tutil.aa2q(jnp.array([0.,0,0]))
    pqeh  = jnp.array([0,0,0.]),            tutil.aa2q(jnp.array([0.,0,-np.pi/4]))
    pqhg  = jnp.array([0,0,0.1034]),        tutil.aa2q(jnp.array([0.,0,0]))         # This will control gripper offset
    pqhlg = jnp.array([0,0,0.0584]),        tutil.aa2q(jnp.array([0.,0,0]))
    pqhrg = jnp.array([0,0,0.0584]),        tutil.aa2q(jnp.array([0.,0,np.pi]))

    pq_dict = {(0,1):pq01, (1,2):pq12, (2,3):pq23, (3,4):pq34, (4,5):pq45, 
               (5,6):pq56, (6,7):pq67, (7,8):tutil.pq_multi(*pq7e, *pqeh)}
    
    # Link 0 at the origin
    zero_pq = jnp.array([0.,0,0]), tutil.aa2q(jnp.array([0.,0,0]))
    link_pq = [zero_pq]
    # Propagate transforms
    for i, j in pq_dict.keys():
        link_pq_cur = tutil.pq_multi(*link_pq[-1], *pq_dict[(i,j)])
        if j == len(pq_dict.keys()):
            pq_joint = link_pq_cur
        else:
            pq_joint = tutil.pq_multi(*link_pq_cur, jnp.array([0,0,0]), tutil.aa2q(jnp.array([0.,0,q[i]])))
        link_pq.append(pq_joint)
    # Gripper links
    hand_pq = link_pq[-1]
    link_pq.append(tutil.pq_multi(*tutil.pq_multi(*hand_pq, *pqhlg), jnp.array([0,gripper,0]), tutil.aa2q(jnp.array([0.,0,0]))))
    link_pq.append(tutil.pq_multi(*tutil.pq_multi(*hand_pq, *pqhrg), jnp.array([0,gripper,0]), tutil.aa2q(jnp.array([0.,0,0]))))
    # Get grasp center pose
    grasp_center_pq = tutil.pq_multi(*hand_pq, *pqhg)

    return link_pq, grasp_center_pq


def cost_func(q, goal_pq, rest_q=None, damped=False, grasp_basis=False):
    '''
    goal_pq is based on grasp target
    '''
    pq_link, cur_pq = Franka_FK(q)
    if not grasp_basis:
        cur_pq = pq_link[-3]
    pq_dif = tutil.pq_multi(*tutil.pq_inv(*cur_pq), *goal_pq)
    pos_dif = pq_dif[0]
    # ang_dif = tutil.q2aa(pq_dif[1])
    ang_dif = tutil.q2R(pq_dif[1]) - jnp.eye(3)
    pos_loss = jnp.sum(pos_dif**2)
    quat_loss = jnp.sum(ang_dif**2)

    # joint limit potential function
    margin = 0.02
    # joint_limit_loss = 0
    joint_limit_loss = jnp.sum(10*jnp.maximum(FRANKA_Q_MIN - q + margin, 0.0)**2) + jnp.sum(10*jnp.maximum(q - FRANKA_Q_MAX + margin, 0.0)**2)
    # joint_limit_loss = jnp.sum(-jnp.log(q-FRANKA_Q_MIN)*0.0001) + jnp.sum(-jnp.log(FRANKA_Q_MAX-q)*0.0001)

    # null_loss = 0.
    # if rest_q is not None:
    #     null_loss = 0.001*jnp.sum((rest_q - q)**2)
    null_loss = 0.004*jnp.sum((q[...,0])**2)

    total_loss = pos_loss + 0.02*quat_loss + joint_limit_loss + null_loss
    if damped:
        total_loss += 1e-7*jnp.sum(q**2)
    return total_loss, (pos_loss, quat_loss)


# @partial(jax.jit, static_argnames=('output_cost', 'grasp_basis'))
def Franka_IK_numetrical(cur_q, goal_pq, robot_base_pos=jnp.zeros(3), robot_base_quat=jnp.array([0.,0,0,1.]), 
                         itr_no=300, output_cost=False, grasp_basis=True, compensate_last_joint=False):
    '''
    goal_pq is based on grasp target
    '''
    if cur_q is None:
        cur_q = jnp.array([0,0,0,-np.pi*0.5,0,np.pi*0.5,0])
    
    goal_pq = tutil.pq_multi(*tutil.pq_inv(robot_base_pos, robot_base_quat), *goal_pq)

    grad_func = jax.grad(cost_func, has_aux=True)
    q = cur_q
    # lr = 3e-1
    lr = 3e-2
    optimizer = optax.adamw(lr)
    opt_state = optimizer.init(q)
    def body_func(i, carry):
        q, _, opt_state = carry
        # grad, cost = grad_func(q, goal_pq, True, grasp_basis)
        grad, cost = grad_func(q, goal_pq, cur_q, False, grasp_basis)
        updates, opt_state = optimizer.update(grad, opt_state, q)
        q = optax.apply_updates(q, updates)
        # grad = jnp.linalg.norm(grad, axis=-1, keepdims=True).clip(0,0.5)*tutil.normalize(grad)
        # q -= lr*grad
        return q, cost, opt_state
    q, cost, _ = jax.lax.fori_loop(0, itr_no, body_func, (cur_q, (0,0), opt_state))
    # print(jnp.sqrt(cost[0]), jnp.sqrt(cost[1]))
    if compensate_last_joint:
        q = q.at[...,-1].add(-np.pi/4)
    if output_cost:
        return q, (jnp.sqrt(cost[0]), jnp.sqrt(cost[1]))
    else:
        return q


def sample_random_configuration(jkey: jax.Array, n=0) -> jnp.ndarray:
    """Random sample franka configuration"""
    # q_min = jnp.array([-2.3093, -1.5133, -2.4937, -2.7478, -2.4800, 0.8521, -2.6895])
    # q_max = jnp.array([2.3093, 1.5133, 2.4937, -0.4461, 2.4800, 3.7094, 2.6895])

    if n==0 or n==1:
        return jax.random.uniform(jkey, shape=(7,), dtype=jnp.float32, minval=FRANKA_Q_MIN, maxval=FRANKA_Q_MAX)
    else:
        return jax.random.uniform(jkey, shape=(n, 7,), dtype=jnp.float32, minval=FRANKA_Q_MIN, maxval=FRANKA_Q_MAX)


def sampler_elipsoidal_heuristic(jkey, start, goal):
    sg_dir = start-goal
    sg_dir_norm = jnp.linalg.norm(sg_dir, axis=-1, keepdims=True)
    sg_dir_normalized = sg_dir/sg_dir_norm
    traj = jax.random.normal(jkey, shape=(7,))
    _, jkey = jax.random.split(jkey)
    r = jax.random.uniform(jkey, shape=(1,))**(1/7)
    _, jkey = jax.random.split(jkey)
    traj = traj/jnp.linalg.norm(traj, axis=-1, keepdims=True)
    traj = traj * r
    
    traj_sg = jnp.sum(sg_dir_normalized * traj, axis=-1, keepdims=True) * sg_dir_normalized
    traj_perp = traj - traj_sg

    traj = traj_sg * sg_dir_norm*0.8 + traj_perp*sg_dir_norm*0.5 + (start+goal)*0.5
    traj = traj.clip(FRANKA_Q_MIN, FRANKA_Q_MAX)

    return traj

def sample_biased_random_configuration(
        jkey: jax.Array, 
        start_q: jnp.ndarray,
        goal_q: jnp.ndarray,
        backward_sample_std = 0.5, 
        backward_sample_ratio = 0.3
) -> jnp.ndarray:
    """Random sample franka configuration biased towards a goal configuration"""
    jkey, fwd_key, bwd_key = jax.random.split(jkey, 3)
    # fwd = sample_random_configuration(fwd_key)
    fwd = sampler_elipsoidal_heuristic(fwd_key, start_q, goal_q)
    bwd = goal_q + jax.random.normal(bwd_key, shape=(7,)) * backward_sample_std
    
    return jnp.where(jax.random.uniform(jkey)>backward_sample_ratio, fwd, bwd)


def last_mile_approach(cur_q, goal_pq, resolution=100, output_cost=False):
    """???"""
    grad_func = jax.grad(cost_func, has_aux=True)

    lr = 3e-1
    def body_func(carry, x):
        q = carry
        grad, cost = grad_func(q, goal_pq, True)
        grad = jnp.linalg.norm(grad, axis=-1, keepdims=True).clip(0,0.5)*tutil.normalize(grad)
        q -= lr*grad
        return q, (q, cost)
    _, (q, cost) = jax.lax.scan(body_func, cur_q, None, length=resolution)
    # print(jnp.sqrt(cost[0]), jnp.sqrt(cost[1]))
    if output_cost:
        return q, (jnp.sqrt(cost[0]), jnp.sqrt(cost[1]))
    else:
        return q


def obj_target_to_grasp(obj_target_pq, grasp_pq_wrt_obj):
    return tutil.pq_multi(*obj_target_pq, *grasp_pq_wrt_obj)


def transform_panda_from_q(
        q: jnp.ndarray,
        panda_objs: cxutil.LatentObjects,
        rot_configs: Dict,
        gripper_width: float=0.05,
) -> cxutil.LatentObjects:
    """Returns the latent objects of franka links with cartesian pose from the joint q.

    Args:
        q (jnp.ndarray): shape=[7,]
        panda_objs (cxutil.LatentObjects, optional): Panda links. outer_shape=[#links,]
        rot_config (Dict): Rot config to rotate latent.

    Returns:
        cxutil.LatentObjects: Transformed links. outer_shape=[#links,]
    """
    if q.ndim != 1:
        raise ValueError("Double check q.dim==1.")
    if len(panda_objs.outer_shape) != 1:
        raise ValueError("Panda objects outer shape must be [#links,].")
    
    link_list = []
    link_pq_list, _ = Franka_FK(q, gripper_width) # DH Link frame
    num_panda_obj_links = panda_objs.outer_shape[0]
    for i, link_pq in enumerate(link_pq_list[:num_panda_obj_links]):
        cur_link_obj: cxutil.LatentObjects = jax.tree_map(lambda x: x[i, ...], panda_objs)
        cur_link_obj = cur_link_obj.apply_pq_z(*link_pq, rot_configs)   
        cur_link_obj = cur_link_obj.broadcast(cur_link_obj.outer_shape)
        link_list.append(cur_link_obj)

    return jax.tree_map(lambda *x: jnp.stack(x, axis=0), *link_list)



def transform_panda_from_q_cvx(
        q: jnp.ndarray,
        panda_objs: cxutil.CvxObjects,
) -> cxutil.CvxObjects:
    """Returns the latent objects of franka links with cartesian pose from the joint q.

    Args:
        q (jnp.ndarray): shape=[7,]
        panda_objs (cxutil.LatentObjects, optional): Panda links. outer_shape=[#links,]
        rot_config (Dict): Rot config to rotate latent.

    Returns:
        cxutil.LatentObjects: Transformed links. outer_shape=[#links,]
    """
    if q.ndim != 1:
        raise ValueError("Double check q.dim==1.")
    if len(panda_objs.outer_shape) != 1:
        raise ValueError("Panda objects outer shape must be [#links,].")
    
    link_list = []
    link_pq_list, _ = Franka_FK(q, 0) # DH Link frame
    num_panda_obj_links = panda_objs.outer_shape[0]
    for i, link_pq in enumerate(link_pq_list[:num_panda_obj_links]):
        cur_link_obj: cxutil.CvxObjects = jax.tree_map(lambda x: x[i, ...], panda_objs)
        cur_link_obj = cur_link_obj.apply_pq_vtxpcd(*link_pq)   
        cur_link_obj = cur_link_obj.broadcast(cur_link_obj.outer_shape)
        link_list.append(cur_link_obj)

    return jax.tree_map(lambda *x: jnp.stack(x, axis=0), *link_list)

def transform_grasped_obj_from_q(
        q: jnp.ndarray,
        query_obj: cxutil.LatentObjects,
        grasp_pq_wrt_obj: Tuple[jnp.ndarray, jnp.ndarray],
        rot_configs: Dict
) -> cxutil.LatentObjects:
    """Returns the grapsed latent objects transformed from the joint q.

    Args:
        q (jnp.ndarray): shape=[7,]
        query_obj (cxutil.LatentObjects, optional): Holding object. outer_shape=[1,]
        grasp_pq_wrt_obj (Tuple[jnp.ndarray, jnp.ndarray], optional): No outer shape
        rot_config (Dict): Rot config to rotate latent.

    Returns:
        cxutil.LatentObjects: Transformed objects. outer_shape=[1,]
    """
    if q.ndim != 1:
        raise ValueError("Double check q.dim==1.")
    if len(query_obj.outer_shape) != 1:
        raise ValueError("Grasped object outer shape must have dimension of 1.")
    if query_obj.outer_shape[0] != 1:
        raise ValueError("Grasped object outer shape must be [1,]")

    _, ee_pq = Franka_FK(q)
    obj_pq = tutil.pq_multi(*ee_pq, *tutil.pq_inv(*grasp_pq_wrt_obj))
    query_obj_transformed = query_obj.apply_pq_z(*obj_pq, rot_configs)
    query_obj_transformed = query_obj_transformed.broadcast(query_obj_transformed.outer_shape)

    return query_obj_transformed


def transform_panda_and_grasped_obj_from_q(
        q: jnp.ndarray,
        panda_objs: cxutil.LatentObjects,
        query_obj: cxutil.LatentObjects,
        grasp_pq_wrt_obj: Tuple[jnp.ndarray, jnp.ndarray],
        rot_configs: Dict
) -> Tuple[cxutil.LatentObjects, cxutil.LatentObjects]:
    """Transform panda objects + grapsed objects together

    Args:
        q (jnp.ndarray): shape=[7,]
        panda_objs (cxutil.LatentObjects): Panda links. outer_shape=[#links,]
        query_obj (cxutil.LatentObjects): Holding object. outer_shape=[1,]
        grasp_pq_wrt_obj (Tuple[jnp.ndarray, jnp.ndarray]): No outer shape
        rot_configs (Dict): Rot config to rotate latent.

    Returns:
        panda_objs_transformed (cxutil.LatentObjects): Transformed panda_objs. outer_shape=[#links,]
        query_obj_transformed (cxutil.LatentObjects): Transformed query_obj. outer_shape=[1,]
    """
    if q.ndim != 1:
        raise ValueError("Double check q.dim==1. Use vmap for this function")    
    if len(panda_objs.outer_shape) != 1:
        raise ValueError("Panda objects outer shape must be [#links,].")
    if len(query_obj.outer_shape) != 1:
        raise ValueError("Grasped object outer shape must have dimension of 1.")
    if query_obj.outer_shape[0] != 1:
        raise ValueError("Grasped object outer shape must be [1,]")
    
    link_list = []
    link_pq_list, ee_pq = Franka_FK(q)             # DH Link frame
    num_panda_obj_links = panda_objs.outer_shape[0]
    for i, link_pq in enumerate(link_pq_list[:num_panda_obj_links]):
        cur_link_obj: cxutil.LatentObjects = jax.tree_map(lambda x: x[i, ...], panda_objs)
        cur_link_obj = cur_link_obj.apply_pq_z(*link_pq, rot_configs)   
        cur_link_obj = cur_link_obj.broadcast(cur_link_obj.outer_shape)
        link_list.append(cur_link_obj)
    panda_objs_transformed = jax.tree_map(lambda *x: jnp.stack(x, axis=0), *link_list)

    obj_pq = tutil.pq_multi(*ee_pq, *tutil.pq_inv(*grasp_pq_wrt_obj))
    query_obj_transformed = query_obj.apply_pq_z(*obj_pq, rot_configs)
    query_obj_transformed = query_obj_transformed.broadcast(query_obj_transformed.outer_shape)

    return panda_objs_transformed, query_obj_transformed


def transform_ee_from_pq(
        ee_posquat: Tuple[jnp.ndarray, jnp.ndarray],
        panda_objs: cxutil.LatentObjects, 
        inside_obj: cxutil.LatentObjects,
        rot_configs: Dict
) -> cxutil.LatentObjects:
    """Returns the latent objects of links of robot hand and two grippers with cartesian pose.

    Args:
        ee_posquat (Tuple[jnp.ndarray, jnp.ndarray], optional): Tuple([..., 3], [..., 4]).
        panda_objs (cxutil.LatentObjects): Panda links.
        inside_obj (cxutil.LatentObjects): latent of gripper inside area.
        rot_configs (Dict): rotational config from models.
    Returns:
        cxutil.LatentObjects: Transformed objects/links
    """

    # latent robot links of hand and two fingers
    hand_obj:cxutil.LatentObjects = jax.tree_map(lambda x: x[-3:], panda_objs)

    gripper = 0.05 # gripper center offset from hand frame

    pqeh = jnp.array([0,0,0.]), tutil.aa2q(jnp.array([0.,0,-np.pi/4]))
    pqhg = jnp.array([0,0,0.09]), tutil.aa2q(jnp.array([0.,0,0]))                 # This will control inside area 
    pqhlg = jnp.array([0,0,0.0584]), tutil.aa2q(jnp.array([0.,0,0]))
    pqhrg = jnp.array([0,0,0.0584]), tutil.aa2q(jnp.array([0.,0,np.pi]))

    # pose of hand and two fingers w.r.t end-effector frame
    hand_pose = tutil.pq_multi(*ee_posquat, *pqeh)
    hand_pq = []
    hand_pq.append(hand_pose)
    hand_pq.append(tutil.pq_multi(*tutil.pq_multi(*hand_pose, *pqhlg), jnp.array([0,gripper,0]), tutil.aa2q(jnp.array([0.,0,0]))))
    hand_pq.append(tutil.pq_multi(*tutil.pq_multi(*hand_pose, *pqhrg), jnp.array([0,gripper,0]), tutil.aa2q(jnp.array([0.,0,0]))))

    link_list = []

    for i, link_pq in enumerate(hand_pq):  
        cur_link_obj:cxutil.LatentObjects = jax.tree_map(lambda x: x[i, jnp.newaxis, ...], hand_obj)  # Keep batch dim
        cur_link_obj = cur_link_obj.apply_pq_z(*link_pq, rot_configs) 
        cur_link_obj = cur_link_obj.broadcast(cur_link_obj.outer_shape)
        link_list.append(cur_link_obj)

    # inside area between two fingers
    inside_pq = tutil.pq_multi(*hand_pose, *pqhg)
    inside_obj = inside_obj.apply_pq_z(*inside_pq, rot_configs)
    inside_obj = inside_obj.broadcast(inside_obj.outer_shape)
    link_list.append(inside_obj)

    return jax.tree_map(lambda *x: jnp.concatenate(x), *link_list)


def get_gripper_center_from_ee_pq(
        ee_posquat: Tuple[jnp.ndarray, jnp.ndarray]
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Returns the grasp center positions from ee base pose. 

    
    Args:
        ee_posquat (Tuple[jnp.ndarray, jnp.ndarray]): Tuple([..., 3], [..., 4]).

    Returns:
        grasp_center_pq (Tuple[jnp.ndarray, jnp.ndarray]): Tuple([..., 3], [..., 4])
    """
    pqeh = jnp.array([0,0,0.]), tutil.aa2q(jnp.array([0.,0,-np.pi/4]))
    pqhg  = jnp.array([0,0,0.1034]), tutil.aa2q(jnp.array([0.,0,0])) # This will control gripper offset

    # pose of hand and two fingers w.r.t end-effector frame
    hand_pose = tutil.pq_multi(*ee_posquat, *pqeh)
    grasp_center_pq = tutil.pq_multi(*hand_pose, *pqhg)

    return grasp_center_pq


def get_gripper_end_from_ee_pq(
        ee_posquat: Tuple[jnp.ndarray, jnp.ndarray]
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Returns the gripper finger end positions.

    Args:
        ee_posquat (Tuple[jnp.ndarray, jnp.ndarray]): Tuple([..., 3], [..., 4]).
    Returns:
        gripper end center pos and rotation matrix: Tuple([..., 3], [..., 3, 3]).
    """
    pqeh = jnp.array([0,0,0.]), tutil.aa2q(jnp.array([0.,0,-np.pi/4]))
    pqh_tip = jnp.array([0,0,0.0584]), tutil.aa2q(jnp.array([0.,0,0]))

    pqh = tutil.pq_multi(*ee_posquat, *pqeh)
    pq_tip = tutil.pq_multi(*pqh, *pqh_tip)
    pq_tip_R = tutil.q2R(pq_tip[1])

    return pq_tip[0], pq_tip_R



def get_gripper_end_AABB_points(
        scale=1.0
) ->jnp.ndarray:

    aabb_min=np.array([[-0.01047577,  0.04994276,  0.05853301],
        [-0.01048346, -0.07630402,  0.05853301]], dtype=np.float32)
    aabb_max=np.array([[ 0.01048347,  0.07630402,  0.11222918],
        [ 0.01047578, -0.04994276,  0.11222918]], dtype=np.float32)
    base_pnts = np.array([[1.,1,1],[1,1,-1],[1,-1,1],[-1,1,1],[1,-1,-1],[-1,1,-1],[-1,-1,1], [-1,-1,-1],
                           [1,0,0], [-1,0,0], [0,1,0], [0,-1,0], [0,0,1], [0,0,-1],
                           [1,1,0], [-1,1,0], [1,-1,0], [-1,-1,0],
                            [0,1,1], [0,-1,1], [0,1,-1], [0,-1,-1],
                            [1,0,1], [-1,0,1], [1,0,-1], [-1,0,-1],
                            # [0,0,0], [0,0,0.5], [0,0,-0.5],
                            # [1,1,0.5], [1,-1,0.5], [-1,1,0.5], [-1,-1,0.5],
                            # [0.5,0.5,1], [-0.5,0.5,1], [0.5,-0.5,1], [-0.5,-0.5,1],
                           ])
    # base_pnts = jnp.concatenate([base_pnts, jax.random.uniform(jax.random.PRNGKey(0), shape=(100,3), minval=-1, maxval=1)], axis=-2)

    occ_check_pos = base_pnts * (aabb_max - aabb_min)[...,None,:]*0.5*scale + (aabb_max + aabb_min)[...,None,:]*0.5
    # occ_check_pos = base_pnts * (aabb_max - aabb_min)[...,None,:]*0.5*jnp.array([0.5,0.5,1.0]) + (aabb_max + aabb_min)[...,None,:]*0.5
    occ_check_pos = tutil.pq_action(np.zeros(3), tutil.aa2q(jnp.array([0,0,-np.pi/4])), occ_check_pos) # [2,14,3]
    return occ_check_pos

def get_default_hand_latent_objs(models):
    import util.environment_util as env_util
    import util.io_util as ioutil

    BUILD = ioutil.BuildMetadata.from_str("32_64_1_v4")
    PANDA_DIR_PATH = BASEDIR/"data"/"PANDA"/str(BUILD)
    panda_key =jax.random.PRNGKey(0)
    panda_link_obj = env_util.create_panda(panda_key, PANDA_DIR_PATH, BUILD, models)

    # Load gripper inside area (8cm x 4cm x 2cm)
    gripper_inside_path = (PANDA_DIR_PATH)/"gripper_inside.obj"
    inside_color = jnp.array([0.95,0.,0.])
    inside_latent_obj = env_util.create_latent_objects(panda_key, [gripper_inside_path], BUILD, models)
    inside_latent_obj = env_util.set_color(inside_latent_obj, inside_color)
    # Set default pose
    default_pos = jnp.array([0., 0., 0.])
    default_quat = jnp.array([0., 0., 0., 1.])
    default_pq = (default_pos, default_quat)
    default_hand_objs = transform_ee_from_pq(default_pq, panda_link_obj, inside_latent_obj, models.rot_configs)

    # if 'implicit_baseline' in models.dif_args and models.dif_args.implicit_baseline:
    #     jkey = jax.random.PRNGKey(13)
    #     jkey, subkey = jax.random.split(jkey)
    #     default_hand_objs = default_hand_objs.register_pcd_from_latent(models, 1000, subkey)
    #     jkey, subkey = jax.random.split(jkey)
    #     panda_link_obj = panda_link_obj.register_pcd_from_latent(models, 1000, subkey)
    #     jkey, subkey = jax.random.split(jkey)
    #     inside_latent_obj = inside_latent_obj.register_pcd_from_latent(models, 1000, subkey)
    return default_hand_objs, panda_link_obj, inside_latent_obj


if __name__ == '__main__':


    import pickle
    import util.model_util as mutil
    import util.render_util as rutil
    import matplotlib.pyplot as plt

    BASEDIR = os.path.dirname(os.path.dirname(__file__))
    if BASEDIR not in sys.path:
        sys.path.insert(0, BASEDIR)

    cp_dir = 'logs/10262023-012454'
    models = mutil.get_models_from_cp_dir(cp_dir)

    panda_latent_obj = panda_latent(jax.random.PRNGKey(0), models)

    # load gripper inside area 8cm x 4cm x 2cm
    gripper_inside_path = 'data/PANDA/gripper_inside.obj'

    inside_latent_obj = load_latent(gripper_inside_path, models, color=jnp.array([0.2,0.3,0.6,1]))

    # camera_pos = jnp.array([0.25,0.25,0.1])
    camera_pos = jnp.array([0.8,0.8,1.3])
    pixel_size = (200, 200)
    sdf = partial(rutil.scene_sdf, models=models, sdf_ratio=1200)
    render_scene_rdf = jax.jit(partial(rutil.cvx_render_scene, sdf=sdf, pixel_size=pixel_size, models=models, camera_pos=camera_pos))

    pos = jnp.array([0.01, -0.01, 0.01])
    quat = jnp.array([ 0.9238795, 0, 0, 0.3826834 ])
    ee_posquat = (pos, quat)

    # hand_objs = ee_pose_to_obj(ee_posquat, panda_latent_obj, inside_latent_obj, models.rot_configs)
    moved_panda_objs = q_to_obj_tf(
        q=jnp.array([0,0,0, -np.pi/2, 0, np.pi/2, 0]),
        panda_objs=panda_latent_obj,
        rot_configs=models.rot_configs)

    # rgb_latent_rdf = render_scene_rdf(hand_objs)
    rgb_latent_rdf = render_scene_rdf(moved_panda_objs)
    plt.figure()
    plt.imshow(rgb_latent_rdf)
    plt.savefig('franka_hand_with_checker.jpg')