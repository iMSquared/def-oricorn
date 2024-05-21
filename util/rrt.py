# %%
import jax.numpy as jnp
import jax
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
import copy
# from moviepy.editor import ImageSequenceClip
import os


# %%
# functions

# def nn(qpnt_, pnts_list, csq_, k=None):
#     # dist_ = jnp.linalg.norm(pnts_list - qpnt_, axis=-1).at[csq_:].set(1e5)
#     npd = pnts_list.shape[-2]
#     dist_ = jnp.where(jnp.arange(npd) < csq_, jnp.linalg.norm(pnts_list - qpnt_, axis=-1), 1e5)
#     if k is None:
#         return jnp.argmin(dist_, axis=-1)
#     else:
#         idx = jnp.argsort(dist_, axis=-1)[...,:k]
#         return idx, dist_[idx]

def nn(qpnt_, pnts_list, csq_, k=None):
    # dist_ = jnp.linalg.norm(pnts_list - qpnt_, axis=-1).at[csq_:].set(1e5)
    npd = pnts_list.shape[-2]
    dist_ = jnp.where(jnp.arange(npd) < csq_, jnp.linalg.norm(pnts_list - qpnt_, axis=-1), 1e5)
    if k is None:
        return jnp.argmin(dist_, axis=-1)
    else:
        idx = jnp.argsort(dist_, axis=-1)[...,:k]
        return idx, dist_[idx]


def draw_plots(qpt_, pnts_list, parent_id, figname=None, gcd_res = None, last_path=None, goal_reaching_id=None):
    # draw tree
    plt.figure(figsize=[5,5])
    ax = plt.gca()
    ax.add_collection(copy.deepcopy(pc))
    plt.plot(init_pnt[0], init_pnt[1], 'ro')
    plt.plot(gpt[0], gpt[1], 'bo')
    plt.plot(qpt_[0], qpt_[1], 'yo')
    for i in range(1,csq):
        cpt = pnts_list[i]
        ppt = pnts_list[parent_id[i]]
        dpnts = np.stack([cpt, ppt], 0)
        plt.plot(dpnts[:,0], dpnts[:,1], 'g')
    if gcd_res is not None and not gcd_res:
        plt.plot(last_path[:,0], last_path[:,1], 'r-')
        cid = goal_reaching_id
        while parent_id[cid] != -1:
            path_ = jnp.stack([pnts_list[parent_id[cid]], pnts_list[cid]], axis=0)
            plt.plot(path_[:,0], path_[:,1], 'r-')
            cid = parent_id[cid]

    ax.set(xlim=(-1, 1), ylim=(-1, 1))
    # ax.axis('equal')
    if figname is None:
        plt.show()
    else:
        plt.savefig(figname)
    plt.close()

# %%
# problem 3
def one_itr_rrt(jkey, pnts_list, parent_id, cost_list, csq, start_q, goal_q, sampler, path_check, expand_length=0.1, star=False, nn_max_len=None):
    '''
    sampler : jkey => state
    path_check : quert_pnt, pick_pnt, resolution => collision result
    '''

    # sampling
    # qpt = jax.random.uniform(jkey, shape=(2,), dtype=jnp.float32, minval=-1, maxval=1)
    qpt = sampler(jkey, start_q, goal_q)
    _, jkey = jax.random.split(jkey)

    if nn_max_len is None:
        nn_max_len = pnts_list.shape[0]

    # pick node in tree
    connect_node_id = nn(qpt, pnts_list[:nn_max_len], csq)
    _, jkey = jax.random.split(jkey)
    pick_node = pnts_list[connect_node_id]
    dir = (qpt - pick_node)
    dir_norm = jnp.linalg.norm(dir, axis=-1, keepdims=True)
    qpt =  dir / (dir_norm+1e-5) * dir_norm.clip(0, expand_length) + pick_node

    if star:
        K = 5
        connect_node_id_nn, steer_costs = nn(qpt, pnts_list[:nn_max_len], csq, k=K)
        parent_valid_mask = steer_costs < 1e4
        jkey, *path_check_keys = jax.random.split(jkey, K+1)
        path_check_keys = jax.tree_map(lambda *x: jnp.stack(x, axis=0), *path_check_keys)
        steer_cd_cost, col_cost = jax.vmap(path_check, (0, None, 0))(path_check_keys, qpt, pnts_list[connect_node_id_nn])
        steer_costs += 1e5*steer_cd_cost + steer_costs*col_cost # weighted by distance
        parent_costs = cost_list[connect_node_id_nn]
        parent_steer_costs = parent_costs + steer_costs
        parent_steer_costs = jnp.where(parent_valid_mask, parent_steer_costs, 1e5)
        pidx = jnp.argmin(parent_steer_costs, axis=-1)
        min_cost = parent_steer_costs[pidx]
        connect_node_id = connect_node_id_nn[pidx]
        pick_node = pnts_list[connect_node_id]

    # path collision check
    _, jkey = jax.random.split(jkey)
    cd_res, col_cost = path_check(jkey, qpt, pick_node)

    if star:
        # rewire
        mask = min_cost + steer_costs < parent_costs
        connect_node_id_rw = jnp.where(mask, connect_node_id_nn, -1)
        parent_id_updated = parent_id.at[connect_node_id_rw].set(csq)
        parent_id_updated = parent_id_updated.at[csq].set(connect_node_id)
        cost_list_updated = cost_list.at[connect_node_id_rw].set(min_cost + steer_costs)
        cost_list_updated = cost_list_updated.at[csq].set(min_cost)

    # if valid, udpates
    parent_id = jnp.where(cd_res[None], parent_id, parent_id.at[csq].set(connect_node_id))
    pnts_list = jnp.where(cd_res[None], pnts_list, pnts_list.at[csq].set(qpt))
    if star:
        cost_list = jnp.where(cd_res[None], cost_list, cost_list_updated)

    csq = jnp.where(cd_res, csq, csq+1)
    
    return qpt, jkey, pnts_list, parent_id, cost_list, csq

# %%
def one_itr_birrt(jkey, pnts_list, parent_id, goal_pnts_list, goal_parent_id, cost_list, csq, csq_goal, sampler, path_check, expand_length=0.1, star=False):
    '''
    sampler : jkey => state
    path_check : quert_pnt, pick_pnt, resolution => collision result
    '''

    # sampling
    qpt_origin = sampler(jkey)
    _, jkey = jax.random.split(jkey)

    # pick node in tree
    connect_node_id = nn(qpt_origin, pnts_list, csq)
    _, jkey = jax.random.split(jkey)
    pick_node = pnts_list[connect_node_id]
    dir = (qpt_origin - pick_node)
    qpt = expand_length * dir / jnp.linalg.norm(dir) + pick_node

    # pick node in tree
    connect_node_id_goal = nn(qpt_origin, goal_pnts_list, csq_goal)
    _, jkey = jax.random.split(jkey)
    pick_node_goal = goal_pnts_list[connect_node_id_goal]
    dir_goal = (qpt_origin - pick_node_goal)
    qpt_goal = expand_length * dir_goal / jnp.linalg.norm(dir_goal) + pick_node_goal

    # path collision check
    cd_res = path_check(qpt, pick_node, 10)
    cd_res_goal = path_check(qpt_goal, pick_node_goal, 10)

    # if valid, udpates
    parent_id = jnp.where(cd_res[None], parent_id, parent_id.at[csq].set(connect_node_id))
    pnts_list = jnp.where(cd_res[None], pnts_list, pnts_list.at[csq].set(qpt))
    csq = jnp.where(cd_res, csq, csq+1)

    goal_parent_id = jnp.where(cd_res[None], goal_parent_id, goal_parent_id.at[csq_goal].set(connect_node_id))
    goal_pnts_list = jnp.where(cd_res[None], goal_pnts_list, goal_pnts_list.at[csq_goal].set(qpt_goal))
    csq_goal = jnp.where(cd_res_goal, csq_goal, csq_goal+1)
    
    return qpt, jkey, pnts_list, parent_id, cost_list, csq, goal_pnts_list, goal_parent_id, csq_goal

# %%
# problem 4
def one_itr_rrtstar(jkey, pnts_list, parent_id, cost_list, csq):
    # sampling
    qpt = jax.random.uniform(jkey, shape=(2,), dtype=jnp.float32, minval=-1, maxval=1)
    _, jkey = jax.random.split(jkey)

    # pick node in tree
    connect_node_id = nn(qpt, pnts_list, csq)
    pick_node = pnts_list[connect_node_id]
    dir = (qpt - pick_node)
    qpt = 0.1 * dir / jnp.linalg.norm(dir) + pick_node

    # choose parent
    connect_node_id_nn, steer_costs = nn(qpt, pnts_list, csq, k=5)
    parent_valid_mask = steer_costs < 1e4
    steer_cd_cost = jax.vmap(path_check, (None, 0, None))(qpt, pnts_list[connect_node_id_nn], 30)
    steer_costs += 1e5*steer_cd_cost
    parent_costs = cost_list[connect_node_id_nn]
    parent_steer_costs = parent_costs + steer_costs
    parent_steer_costs = jnp.where(parent_valid_mask, parent_steer_costs, 1e5)
    pidx = jnp.argmin(parent_steer_costs, axis=-1)
    min_cost = parent_steer_costs[pidx]
    connect_node_id = connect_node_id_nn[pidx]
    pick_node = pnts_list[connect_node_id]

    # path collision check
    cd_res = path_check(qpt, pick_node, 50)

    # rewire
    mask = min_cost + steer_costs < parent_costs
    connect_node_id_rw = jnp.where(mask, connect_node_id_nn, -1)
    parent_id_updated = parent_id.at[connect_node_id_rw].set(csq)
    parent_id_updated = parent_id_updated.at[csq].set(connect_node_id)
    cost_list_updated = cost_list.at[connect_node_id_rw].set(min_cost + steer_costs)
    cost_list_updated = cost_list_updated.at[csq].set(min_cost)

    # if valid, udpates
    parent_id = jnp.where(cd_res[None], parent_id, parent_id_updated)
    pnts_list = jnp.where(cd_res[None], pnts_list, pnts_list.at[csq].set(qpt))
    cost_list = jnp.where(cd_res[None], cost_list, cost_list_updated)
    csq = jnp.where(cd_res, csq, csq+1)

    return qpt, jkey, pnts_list, parent_id, cost_list, csq


# %%
if __name__ == '__main__':
    from functools import partial

    jkey = jax.random.PRNGKey(0)
    np.random.seed(1)
    FPS = 200
    npd = 400
    nbox = 50

    box_pos = np.random.uniform(low=-1, high=1, size=(nbox, 2))
    box_hscale = np.random.uniform(low=0.01, high=0.15, size=(nbox, 2))
    box_ep = jnp.concatenate([box_pos - box_hscale, box_pos + box_hscale], axis=-1)
    box_vis = [Rectangle(bx[:2], *list(bx[2:] - bx[:2])) for bx in box_ep]
    pc = PatchCollection(box_vis, facecolor=np.array([0.5,0.5,0.5]), alpha=1.0,
                            edgecolor=np.array([0,0,0]))

    def collision_checker(qpnts):
        box_epcd = box_ep
        if len(qpnts.shape) >= 2:
            box_epcd = box_ep[None]
            for _ in range(len(box_epcd.shape) - len(qpnts.shape)):
                qpnts = qpnts[:,None]
        boxes_cd = jnp.all(jnp.concatenate([qpnts < box_ep[...,2:], qpnts > box_ep[...,:2]], axis=-1), axis=-1)
        return jnp.any(boxes_cd, axis=-1)
        

    # qps = np.random.uniform(low=-1, high=1, size=(1000,2))
    # res = collision_checker(qps)

    # init_pnt = np.random.uniform(-1, 1, size=(2,))
    # while collision_checker(init_pnt):
    #     init_pnt = np.random.uniform(-1, 1, size=(2,))

    # gpt = np.random.uniform(-1, 1, size=(2,))
    # while collision_checker(gpt):
    #     gpt = np.random.uniform(-1, 1, size=(2,))

    def init_nodes():
        pnts_list = jnp.zeros((npd, 2))
        parent_id = -2*jnp.ones((npd,), dtype=jnp.int32)

        init_pnt = np.random.uniform(-1, 1, size=(2,))
        while collision_checker(init_pnt):
            init_pnt = np.random.uniform(-1, 1, size=(2,))

        gpt = np.random.uniform(-1, 1, size=(2,))
        while collision_checker(gpt):
            gpt = np.random.uniform(-1, 1, size=(2,))
        
        pnts_list = pnts_list.at[0].set(init_pnt)
        parent_id = parent_id.at[0].set(-1)

        return pnts_list, parent_id, init_pnt, gpt

    def path_check(qpt_, node_pnts_, col_res_no=100):
        # path collision check
        cdqpnts_ = qpt_ + np.linspace(0, 1, num=col_res_no)[...,None] * (node_pnts_ - qpt_)
        cd_res_ = collision_checker(cdqpnts_)
        cd_res_ = jnp.any(cd_res_, axis=-1)
        return cd_res_

    def sampler(jkey):
        return jax.random.uniform(jkey, shape=(2,), dtype=jnp.float32, minval=-1, maxval=1)

    one_itr_rrt_jit = jax.jit(partial(one_itr_rrt, sampler=sampler, path_check=path_check, star=True))

    # init
    # tree expansions
    pnts_list, parent_id, init_pnt, gpt = init_nodes()
    cost_list = 1e5*jnp.ones((parent_id.shape[0],))
    cost_list = cost_list.at[0].set(0.0)

    if not os.path.exists('figure3'):
        os.makedirs('figure3')

    csq = 1
    itr_no = int(1.5*npd)
    for itr in range(itr_no):
        qpt, jkey, pnts_list, parent_id, cost_list, csq = one_itr_rrt_jit(jkey, pnts_list, parent_id, cost_list, csq)
        draw_plots(qpt, pnts_list, parent_id, figname=os.path.join('figure3', str(itr)+'.png'))

    # check goals
    nn_idx = nn(gpt, pnts_list, csq)
    gcd_res =path_check(gpt, pnts_list[nn_idx])
    last_path = jnp.stack([gpt, pnts_list[nn_idx]], axis=0)

    draw_plots(qpt, pnts_list, parent_id, figname=os.path.join('figure3', str(itr_no)+'.png'), gcd_res=gcd_res, last_path=last_path, goal_reaching_id=nn_idx)

    images_list = ['figure3/'+str(i)+'.png' for i in range(itr_no+1)]
    # clip = ImageSequenceClip(images_list, fps=25)
    # clip.speedx(10.0).write_gif("{}.gif".format('problem3')) 




    # rrt star
    one_itr_rrtstar_jit = jax.jit(one_itr_rrtstar)
    # one_itr_pb4_jit = one_itr_pb4

    # init
    # tree expansions
    pnts_list, parent_id, init_pnt, gpt = init_nodes()
    cost_list = 1e5*jnp.ones((parent_id.shape[0],))
    cost_list = cost_list.at[0].set(0.0)

    if not os.path.exists('figure4'):
        os.makedirs('figure4')

    csq = 1
    itr_no = int(1.5*npd)
    for itr in range(itr_no):
        qpt, jkey, pnts_list, parent_id, cost_list, csq = one_itr_rrtstar_jit(jkey, pnts_list, parent_id, cost_list, csq)
        draw_plots(qpt, pnts_list, parent_id, figname=os.path.join('figure4', str(itr)+'.png'))

    # check goals
    nn_idx = nn(gpt, pnts_list, csq)
    gcd_res =path_check(gpt, pnts_list[nn_idx])
    last_path = jnp.stack([gpt, pnts_list[nn_idx]], axis=0)

    draw_plots(qpt, pnts_list, parent_id, figname=os.path.join('figure4', str(itr_no)+'.png'), gcd_res=gcd_res, last_path=last_path, goal_reaching_id=nn_idx)

    # images_list = ['figure4/'+str(i)+'.png' for i in range(itr_no+1)]
    # clip = ImageSequenceClip(images_list, fps=25)
    # clip.speedx(10.0).write_gif("{}.gif".format('problem4')) 