import os, sys
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import jax.numpy as jnp
import jax
import flax.linen as nn
import numpy as np
import matplotlib.pyplot as plt
import optax
import datetime
import argparse
from functools import partial
import einops
import shutil
import pickle
from ott.geometry import pointcloud
from ott.solvers import linear
import time

import argparse
from pathlib import Path
from tqdm import tqdm

try:
    import vessl
    vessl.init()
    vessl_on = True
except:
    vessl_on = False

# Setup import path
import sys
BASEDIR = Path(__file__).parent.parent
if str(BASEDIR) not in sys.path:
    sys.path.insert(0, str(BASEDIR))

import util.diffusion_util as dfutil
import util.cvx_util as cxutil
import util.render_util as rutil
import util.model_util as mutil
import util.camera_util as cutil
import util.structs as structs
import util.train_util as trutil
import util.aug_util as agutil
import util.transform_util as tutil

from dataset.estimator_dataset import EstimatorDataset, pytree_collate

# torch should be followed by importing utils
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader


# define loss
def cal_loss(models:mutil.Models, params, edm_params, dpnts:dict, jkey):
    models_ = models.set_params(params)
    t_samples = dfutil.sample_t_train(jkey, shape=(args.batch_size, args.nrec), edm_params=edm_params, dm_type=args.dm_type, add_t_sample_bias=args.add_t_sample_bias==1)
    t_samples = jnp.sort(t_samples, axis=-1)[...,::-1]
    if args.one_step_test:
        t_samples = jnp.ones_like(t_samples)

    # pose and rotational argmentations
    pos_randomization = jax.random.uniform(jkey, (args.batch_size, 1, 3), minval=-0.1, maxval=0.1)
    _, jkey = jax.random.split(jkey)
    z_rotation = jax.random.uniform(jkey, (args.batch_size, 1, 1), minval=-np.pi, maxval=np.pi)
    _, jkey = jax.random.split(jkey)
    z_rotation = jnp.c_[jnp.zeros_like(z_rotation), jnp.zeros_like(z_rotation), z_rotation]
    z_rotation = tutil.aa2q(z_rotation)

    rgbs = dpnts["rgbs"]
    if rgbs is not None:

        # data augmentation
        rgbs = agutil.translation(jkey, rgbs)
        _, jkey = jax.random.split(jkey)
        rgbs = agutil.gaussian_blur(jkey, rgbs)
        _, jkey = jax.random.split(jkey)

        # sensor dropout
        nv = rgbs.shape[-4]
        nv_sample = jax.random.randint(jkey, rgbs.shape[:-4], 2, int(nv*1.8), dtype=jnp.int32)
        _, jkey = jax.random.split(jkey)
        nv_sample = jnp.arange(nv) < nv_sample[...,None]
        nv_sample = jax.random.permutation(jkey, nv_sample, axis=-1)
        _, jkey = jax.random.split(jkey)
        rgbs = jnp.where(nv_sample[...,None,None,None], rgbs, 0).astype(rgbs.dtype)

        cam_posquats = dpnts["cam_info"]["cam_posquats"].astype(jnp.float32)
        cam_posquats = tutil.pq_multi(pos_randomization, z_rotation, cam_posquats[...,:3], cam_posquats[...,3:])
        cam_posquats = jnp.concatenate(cam_posquats, axis=-1)

        img_feat = models_.apply('img_encoder', rgbs, cam_posquats, dpnts["cam_info"]["cam_intrinsics"].astype(jnp.float32), train=True)
        
        # segmentation loss
        if args.train_seg and 'seg' in dpnts:
            img_feat_valid_idx = jax.vmap(lambda jk, prob: jax.random.choice(jk, jnp.arange(nv), shape=(2,), replace=False, p=prob))(jax.random.split(jkey, rgbs.shape[0]), nv_sample)
            _, jkey = jax.random.split(jkey)
            img_feat_valid = structs.ImgFeatures(None, 
                                        None, 
                                        jnp.take_along_axis(img_feat.img_feat, img_feat_valid_idx[...,None,None,None], axis=-4))
            if args.separate_seg_model:
                rgbs_valid = jnp.take_along_axis(rgbs, img_feat_valid_idx[...,None,None,None], axis=-4)
                seg_pred_logit = models_.apply('seg_predictor', rgbs_valid, train=True)
            else:
                seg_pred_logit = models_.apply('seg_predictor', img_feat_valid, train=True)
            seg_gt = jnp.take_along_axis(dpnts["seg"], img_feat_valid_idx[...,None,None], axis=-3).astype(jnp.float32)
            seg_pred = nn.sigmoid(seg_pred_logit).clip(1e-5, 1-1e-5).squeeze(-1)
            assert seg_gt.shape == seg_pred.shape
            seg_loss = seg_gt*jnp.log(seg_pred) + (1-seg_gt)*jnp.log(1-seg_pred) # (NB NC NI NJ)
            seg_loss = -jnp.sum(seg_loss, axis=(-1,-2))
            seg_loss = jnp.mean(seg_loss, axis=-1)
            seg_loss = args.train_seg*jnp.mean(seg_loss)
        else:
            seg_loss = 0
    else:
        img_feat = None
        seg_loss = 0
    obj_info = dpnts["obj_info"]

    x0_pred = cxutil.CvxObjects().init_obj_info(obj_info)
    x0_pred = x0_pred.apply_pq_vtx(pos_randomization, z_rotation)
    x0_pred_origin = x0_pred.set_z_with_models(jkey, models_, True, dc_center_no=args.volume_points_no)
    
    # invalid obj padding
    if x0_pred_origin.outer_shape[-1] > 1:
        prob = x0_pred_origin.obj_valid_mask.astype(jnp.float32)
        prob = prob/jnp.sum(prob, axis=-1, keepdims=True)
        # valid_idx = jax.vmap(lambda jk, a, p: jax.random.choice(jk, a, shape=(prob.shape[-1],), p=p))(jax.random.split(jkey, prob.shape[0]), einops.repeat(jnp.arange(x0_pred_origin.outer_shape[-1], dtype=jnp.int32), 'i -> r i', r=prob.shape[0]), prob)
        valid_idx = jax.vmap(lambda jk, a, p: jax.random.choice(jk, a, shape=(args.nparticles,), p=p))(jax.random.split(jkey, prob.shape[0]), 
                                                                    einops.repeat(jnp.arange(x0_pred_origin.outer_shape[-1], dtype=jnp.int32), 'i -> r i', r=prob.shape[0]), 
                                                                    prob)
        valid_h = jnp.take_along_axis(x0_pred_origin.h, valid_idx[...,None], axis=-2)
        x0_pred_origin = x0_pred_origin.set_h(jnp.where(x0_pred_origin.obj_valid_mask[...,None], x0_pred_origin.h, valid_h[...,:args.ds_obj_no,:]))
        x0_pred_pad = x0_pred_origin.drop_gt_info().concat(cxutil.LatentObjects().init_h(valid_h[...,args.ds_obj_no:,:], x0_pred_origin.latent_shape), axis=1)
        if args.nparticles==args.ds_obj_no:
            obj_valid_mask_pad = x0_pred_origin.obj_valid_mask
        else:
            obj_valid_mask_pad = jnp.c_[x0_pred_origin.obj_valid_mask, jnp.zeros((x0_pred_origin.obj_valid_mask.shape[0], args.nparticles-args.ds_obj_no), dtype=bool)]
        # x0_pred_origin = x0_pred_origin.set_h(jnp.where(x0_pred_origin.obj_valid_mask[...,None], x0_pred_origin.h, valid_h))
    else:
        x0_pred_pad = x0_pred_origin.drop_gt_info()
    x0_pred = x0_pred_pad
    _, jkey = jax.random.split(jkey)

    noise_gt = jax.random.normal(jkey, x0_pred.h.shape)
    noise_gt = dfutil.noise_FER_projection(noise_gt, x0_pred.latent_shape, models_.rot_configs)
    _, jkey = jax.random.split(jkey)

    dif_loss = 0
    obs_loss = 0
    gan_loss = 0
    if args.cond_mask_ratio >= 1.0:
        batch_condition_mask = None
    else:
        batch_condition_mask = jax.random.uniform(jkey, shape=(args.batch_size,)) <= args.cond_mask_ratio
    _, jkey = jax.random.split(jkey)
    noise_pred = None
    for i in range(args.nrec):
        if i==0:
            if args.one_step_test==3:
                latent_obj_ptb = x0_pred.set_h(einops.repeat(models_.learnable_queries, 'i ... -> r i ...', r=x0_pred.outer_shape[0]))
            else:
                latent_obj_ptb = dfutil.forward_process_obj(x0_pred, t_samples[...,i], jkey, noise=noise_gt, dm_type=args.dm_type, rot_configs=models_.rot_configs, deterministic_mask=True)
                # noise padding with 
                noise_pad = dfutil.noise_FER_projection(jax.random.normal(jkey, latent_obj_ptb.h.shape), x0_pred.latent_shape, models_.rot_configs)
                _, jkey = jax.random.split(jkey)
                # noise_pad = jnp.where(jax.random.uniform(jkey, noise_pad[...,:1].shape)<0.5, noise_pad, jnp.zeros_like(noise_pad))# ???
                noise_pad = jnp.where(jax.random.uniform(jkey, noise_pad[...,:1].shape)<0.5, noise_pad, latent_obj_ptb.h) # half - duplication padding / random padding
                latent_obj_ptb = latent_obj_ptb.set_h(jnp.where(obj_valid_mask_pad[...,None], latent_obj_ptb.h, noise_pad))
                _, jkey = jax.random.split(jkey)
                latent_obj_ptb = jax.lax.stop_gradient(latent_obj_ptb)
        else:
            if args.one_step_test==3:
                latent_obj_ptb = x0_pred
            else:
                if batch_condition_mask is not None:
                    noise_pad = dfutil.noise_FER_projection(jax.random.normal(jkey, noise_pred.shape), x0_pred.latent_shape, models_.rot_configs)
                    noise_pred = jnp.where(batch_condition_mask[...,None,None], noise_pred, noise_pad)
                    _, jkey = jax.random.split(jkey)
                    # x0_pred = x0_pred.set_h(jnp.where(batch_condition_mask[...,None,None], x0_pred.h, x0_pred_origin.h))
                    x0_pred = x0_pred.set_h(jnp.where(batch_condition_mask[...,None,None], x0_pred.h, x0_pred_pad.h))

                origin_noise_mask = jax.random.uniform(jkey, shape=(args.batch_size,)) <= args.skip_rec_ratio
                _, jkey = jax.random.split(jkey)
                noise_pad = dfutil.noise_FER_projection(jax.random.normal(jkey, noise_pred.shape), x0_pred.latent_shape, models_.rot_configs)
                noise_pred = jnp.where(origin_noise_mask[...,None,None], noise_pred, noise_pad)
                _, jkey = jax.random.split(jkey)
                # x0_pred = x0_pred.set_h(jnp.where(origin_noise_mask[...,None,None], x0_pred.h, x0_pred_origin.h))
                x0_pred = x0_pred.set_h(jnp.where(origin_noise_mask[...,None,None], x0_pred.h, x0_pred_pad.h))
                # latent_obj_ptb = dfutil.forward_process_obj(x0_pred, t_samples[...,i], jkey, noise=noise_pred, dm_type=args.dm_type, , rot_configs=models_.rot_configs)
                latent_obj_ptb = dfutil.forward_process_obj(x0_pred, t_samples[...,i], jkey, noise=noise_pred, dm_type=args.dm_type, rot_configs=models_.rot_configs, deterministic_mask=True)
                _, jkey = jax.random.split(jkey)

                # padding ptb obj variables
                noise_ptb_mask1 = jnp.logical_not(jnp.logical_or(obj_valid_mask_pad[...,None], batch_condition_mask[...,None,None]))
                noise_ptb_mask2 = jnp.logical_not(jnp.logical_or(obj_valid_mask_pad[...,None], origin_noise_mask[...,None,None]))
                noise_ptb_mask = jnp.logical_or(noise_ptb_mask1, noise_ptb_mask2)
                noise_pad = dfutil.noise_FER_projection(jax.random.normal(jkey, latent_obj_ptb.h.shape), x0_pred.latent_shape, models_.rot_configs)
                _, jkey = jax.random.split(jkey)
                # noise_pad = jnp.where(jax.random.uniform(jkey, noise_pad[...,:1].shape)<0.5, noise_pad, jnp.zeros_like(noise_pad)) # ??
                noise_pad = jnp.where(jax.random.uniform(jkey, noise_pad[...,:1].shape)<0.5, noise_pad, latent_obj_ptb.h) # half - duplication padding / random padding
                latent_obj_ptb = latent_obj_ptb.set_h(jnp.where(noise_ptb_mask, noise_pad, latent_obj_ptb.h))
                _, jkey = jax.random.split(jkey)
            latent_obj_ptb = jax.lax.stop_gradient(latent_obj_ptb)

        x0_pred, conf_pred = models_.apply('denoiser', latent_obj_ptb, img_feat, t_samples[...,i], batch_condition_mask=batch_condition_mask, confidence_out=True, train=True, rngs={'dropout':jkey})
        _, jkey = jax.random.split(jkey)
        if not args.one_step_test:
            noise_pred = dfutil.get_noise_pred(t_samples[...,i], latent_obj_ptb, x0_pred, args.dm_type)

        loss_weight = 1
        if args.dm_type == 'edm' and args.add_loss_weight == 1:
            loss_weight = (t_samples[...,i]**2 + edm_params.sigma_data**2)/(t_samples[...,i]*edm_params.sigma_data)**2

        if args.loss_type == 'ec':
            pairwise_cost_func = trutil.L1Cost
        elif args.loss_type == 'cf':
            pairwise_cost_func = trutil.CFCost
        elif args.loss_type == 'em':
            pairwise_cost_func = trutil.DFCost
        
        if args.train_loss_type == 'ec':
            train_pairwise_cost_func = trutil.L1Cost
        elif args.train_loss_type == 'cf':
            train_pairwise_cost_func = partial(trutil.CFCost, dif_func=args.dif_func)
        elif args.train_loss_type == 'em':
            train_pairwise_cost_func = trutil.DFCost
            
        def em_loss(obj1:cxutil.LatentObjects, obj2:cxutil.LatentObjects): # ot for particles
            if args.loss_type == 'cf':
                geom_ott = pointcloud.PointCloud(obj1.h, obj2.h, cost_fn=pairwise_cost_func(models.latent_shape, 
                                                                                            args.dc_pos_loss_coef, 
                                                                                            args.matching_pos_factor*args.pos_loss_coef, only_pos=True)) # Euclidian distance
            else:
                geom_ott = pointcloud.PointCloud(obj1.h, obj2.h, cost_fn=pairwise_cost_func(models.latent_shape, 
                                                                                            args.matching_dc_factor*args.dc_pos_loss_coef, 
                                                                                            args.matching_pos_factor*args.pos_loss_coef)) # Euclidian distance
            geom_ott_loss = pointcloud.PointCloud(obj1.h, obj2.h, cost_fn=train_pairwise_cost_func(models.latent_shape, args.dc_pos_loss_coef, args.pos_loss_coef)) # Euclidian distance
            if obj1.h.shape[-2] == 1 or obj2.h.shape[-2] == 1:
                weight_mat = 1
                cost_mat = weight_mat * geom_ott_loss.cost_matrix
                cost_mat = jnp.sum(cost_mat, axis=-1)
                weight_mat_reduced = 1
            else:
                b_weight = (obj2.obj_valid_mask).astype(jnp.float32)
                b_weight = b_weight/jnp.sum(b_weight, axis=-1, keepdims=True)
                ot_res = linear.solve(geom_ott, a=None, b=b_weight)
                weight_mat = jax.lax.stop_gradient(ot_res.matrix)
                weight_mat = jnp.where(obj2.obj_valid_mask[...,None,:], (weight_mat >= jnp.max(weight_mat, axis=-2, keepdims=True)-1e-5).astype(jnp.float32), 0)
                weight_mat = jax.lax.stop_gradient(weight_mat)
                cost_mat = (1-weight_mat) * 1e5 +  jnp.where(obj2.obj_valid_mask[...,None,:], geom_ott_loss.cost_matrix, 0)
                weight_mat_reduced = (jnp.argmin(cost_mat, axis=-2,keepdims=True) == jnp.arange(cost_mat.shape[-2])[...,None]).astype(jnp.float32)
                weight_mat_reduced = jnp.where(obj2.obj_valid_mask[...,None,:], weight_mat_reduced, 0)
                cost_mat = jnp.where(obj2.obj_valid_mask, jnp.min(cost_mat, axis=-2), 0)
            return jnp.mean(cost_mat, axis=-1), weight_mat_reduced
        
        if args.obj_matching=='em':
            dif_loss_, weigt_mat = jax.vmap(em_loss)(x0_pred, x0_pred_origin) # EM loss
            dif_loss_ = loss_weight*dif_loss_
            conf_label = jnp.any(weigt_mat==1, axis=-1).astype(jnp.float32)[...,None]
        elif args.obj_matching=='cf':
            pairwise_all = jax.vmap(train_pairwise_cost_func(models.latent_shape, args.dc_pos_loss_coef, args.pos_loss_coef).all_pairs_pairwise)(x0_pred.h, x0_pred_origin.h)
            dif_loss_ = jnp.mean(jnp.min(pairwise_all, axis=-1) + jnp.min(pairwise_all, axis=-2), axis=-1)
            dif_loss_ = loss_weight*dif_loss_
            conf_label = (jnp.argmax(pairwise_all, axis=-2,keepdims=True)==jnp.arange(pairwise_all.shape[-2])[...,None])
            conf_label = jnp.any(conf_label, axis=-1, keepdims=True).astype(jnp.float32)
        elif args.obj_matching=='ec':
            dif_loss_ = jax.vmap(jax.vmap(train_pairwise_cost_func(models.latent_shape, args.dc_pos_loss_coef, args.pos_loss_coef).pairwise))(x0_pred.h, x0_pred_origin.h)
            dif_loss_ = jnp.where(x0_pred_origin.obj_valid_mask, dif_loss_, 0)
            dif_loss_ = loss_weight*dif_loss_
            # dif_loss_ = pairwise_cost_func(models.latent_shape, args.dc_pos_loss_coef, args.pos_loss_coef).pairwise(x0_pred.h, x0_pred_origin.h)
            conf_label = x0_pred_origin.obj_valid_mask.astype(jnp.float32)[...,None]
        dif_loss_ = jnp.mean(dif_loss_)
        dif_loss += dif_loss_

        if x0_pred_origin.outer_shape[-1] > 1:
            # conf loss
            assert conf_pred.shape == conf_label.shape
            conf_pred_sig = jax.nn.sigmoid(conf_pred).clip(1e-5, 1-1e-5)
            conf_loss = conf_label*jnp.log(conf_pred_sig) + (1-conf_label)*jnp.log(1-conf_pred_sig)
            conf_loss = -jnp.mean(conf_loss, axis=(-1,-2))
            conf_loss = args.conf_loss_weight*jnp.mean(conf_loss)
        else:
            conf_loss = 0

        if args.train_obs_model:
            # obs preds
            t_samples_obs = jax.random.uniform(jkey, t_samples[...,i].shape, minval=0, maxval=jnp.minimum(t_samples[...,i], 0.05))
            _, jkey = jax.random.split(jkey)
            obj_ptb_gt = dfutil.forward_process_obj(x0_pred_origin, t_samples_obs, jkey, noise=None, dm_type=args.dm_type, deterministic_mask=True, rot_configs=models_.rot_configs)
            _, jkey = jax.random.split(jkey)
            obs_positive = models_.apply('obs_model', jax.lax.stop_gradient(obj_ptb_gt), img_feat, t_samples_obs, train=True, rngs={'dropout':jkey})
            _, jkey = jax.random.split(jkey)
            obs_positive = jnp.where(x0_pred_origin.obj_valid_mask[...,None], obs_positive, 0)
            _, jkey = jax.random.split(jkey)

            obj_ptb_false = dfutil.forward_process_obj(x0_pred, t_samples_obs, jkey, noise=noise_pred, dm_type=args.dm_type, deterministic_mask=True, rot_configs=models_.rot_configs)
            _, jkey = jax.random.split(jkey)
            obs_negative = models_.apply('obs_model', jax.lax.stop_gradient(obj_ptb_false), img_feat, t_samples_obs, train=True, rngs={'dropout':jkey})
            _, jkey = jax.random.split(jkey)

            obs_loss_ = jnp.log(jax.nn.sigmoid(obs_positive).clip(1e-5, 1-1e-5)) + jnp.log(1-jax.nn.sigmoid(obs_negative).clip(1e-5, 1-1e-5))
            obs_loss_ = -jnp.mean(obs_loss_, axis=(-1,-2))
            obs_loss_ = args.conf_loss_weight*jnp.mean(obs_loss_)
            obs_loss += obs_loss_

            if args.train_obs_model==2:
                obs_pred_for_gan = models_.apply('obs_model', obj_ptb_false, jax.lax.stop_gradient(img_feat), t_samples_obs, train=False, rngs={'dropout':jkey})
                gan_loss_ = jnp.log(1-jax.nn.sigmoid(obs_pred_for_gan).clip(1e-5, 1-1e-5))
                gan_loss_ = jnp.mean(gan_loss_, axis=(-1,-2))
                gan_loss_ = jnp.mean(gan_loss_)
                gan_loss += gan_loss_

    loss = dif_loss + obs_loss + conf_loss + gan_loss + seg_loss

    return loss, {'dif_loss':dif_loss, 'obs_loss':obs_loss, 'conf_loss':conf_loss, 'gan_loss':gan_loss, 'seg_loss':seg_loss}


def main(args: argparse.Namespace):
    if jax.device_count('gpu') == 0:
        print('no gpu found. End process')
        return
    else:
        print('device found: ', jax.devices())
    jkey = jax.random.PRNGKey(args.seed)

    if args.one_step_test in [1, 2]:
        args.nrec = 1
        args.add_c_skip = 0
        # args.nparticles = 1
        args.cond_mask_ratio = 1.0
    
    if args.one_step_test == 3:
        args.add_c_skip = 1

    # Configs
    DATA_DIR = Path(args.data_dir)
    BATCH_SIZE = args.batch_size
    NUM_WORKERS = args.num_workers

    train_dataset = EstimatorDataset(args.dataset, DATA_DIR, args.category, args.ds_size_limit, args.ds_obj_no, 'train')
    eval_dataset = EstimatorDataset(args.dataset, DATA_DIR, args.category, None, args.ds_obj_no, 'test')
    train_loader = DataLoader(
        dataset = train_dataset, 
        batch_size = BATCH_SIZE,
        num_workers = NUM_WORKERS,
        collate_fn = pytree_collate,
        pin_memory = False, # Only for torch
        shuffle=True,
        drop_last = True)
    eval_loader = DataLoader(
        dataset = eval_dataset, 
        batch_size = BATCH_SIZE//2,
        num_workers = NUM_WORKERS,
        collate_fn = pytree_collate,
        pin_memory = False, # Only for torch
        shuffle=True,
        drop_last = True)
    train_ds_len = len(train_dataset)//BATCH_SIZE
    # Stats
    print(f"Train examples: {len(train_dataset)}")
    print(f"Eval examples: {len(eval_dataset)}")

    # # visualize dataset
    # nviews = train_dataset[0]["rgbs"].shape[0]
    # pixel_size = train_dataset[0]["rgbs"].shape[1:3]
    # cvx_render_func = jax.jit(partial(rutil.cvx_render_scene, pixel_size=pixel_size))
    # for k in range(10):
    #     plt.figure()
    #     for i in range(2):
    #         obj = cxutil.CvxObjects().init_obj_info(train_dataset[2*k+i]['obj_info'])
    #         for j in range(nviews):
    #             vis_intrinsic = train_dataset[2*k+i]["cam_info"]["cam_intrinsics"][j].astype(np.float32)
    #             vis_cam_posquat = train_dataset[2*k+i]["cam_info"]["cam_posquats"][j].astype(np.float32)
    #             vis_cam_pos, vis_cam_quat = vis_cam_posquat[:3], vis_cam_posquat[3:]
    #             cvx_rendered = cvx_render_func(obj, intrinsic=vis_intrinsic, camera_pos=vis_cam_pos, camera_quat=vis_cam_quat)
    #             plt.subplot(2*2,nviews,nviews*2*i+j+1)
    #             plt.imshow(train_dataset[2*k+i]["rgbs"][j])
    #             plt.subplot(2*2,nviews,nviews*(2*i+1)+j+1)
    #             plt.imshow(cvx_rendered)
        # plt.show()
    # # visualize dataset
    train_data_sample = train_dataset[0]

    optimizer = optax.adam(1e-4)
    if args.global_representation:
        if args.implicit_baseline:
            cp_dir = 'checkpoints/pretraining/implicit'
            print('load pretraining model from implicit function baselines')
        elif args.dataset=='NOCS':
            cp_dir = 'checkpoints/pretraining/01152024-074410'
        elif args.category=='cherrypick2':
            cp_dir = 'checkpoints/pretraining/01082024-074428' # global representations
        elif args.category=='cherrypick':
            cp_dir = 'checkpoints/pretraining/12282023-121639' # global representations
        elif args.category in ['none', 'all']:
            cp_dir = 'checkpoints/pretraining/12292023-095943' # global representations
    else:
        cp_dir = 'checkpoints/pretraining/12122023-003910' # 64-32

    if args.checkpoint_dir!='none':
        util_dir = BASEDIR/'util'
        if str(util_dir) not in sys.path:
            sys.path.insert(0, str(util_dir))
        with open(os.path.join(args.checkpoint_dir, 'saved.pkl'), 'rb') as f:
            raw_saved = pickle.load(f)
        models = mutil.get_models_from_cp_dir(args.checkpoint_dir)
        params = raw_saved['params']
        opt_state = raw_saved['opt_state']
        try:
            train_steps_cummulative = raw_saved['train_steps_cummulative']
            train_epochs_cummulative = raw_saved['train_epochs_cummulative']
        except:
            train_steps_cummulative = 0
            train_epochs_cummulative = 0
        print(f'load checkpoints: {args.checkpoint_dir}, continue at {train_steps_cummulative} steps, {train_epochs_cummulative} epochs')
        if args.update_pretraining_weight:
            with open(os.path.join(cp_dir, 'saved.pkl'), 'rb') as f:
                pretrain_models = pickle.load(f)['models']
            models = models.set_params(pretrain_models.pretraining_params)
            del pretrain_models
            print('pretraining weights are updated')
    else:
        print('no checkpoints. init from scrach')
        # models = mutil.get_models_from_cp_dir(cp_dir)
        with open(os.path.join(cp_dir, 'saved.pkl'), 'rb') as f:
            models = pickle.load(f)['models']
        train_data_sample = train_dataset[0:1]
        models = models.init_dif_model_scenedata(args, train_data_sample)
        params = models.params
        opt_state = optimizer.init(models.params)
        train_steps_cummulative = 0
        train_epochs_cummulative = 0
    
    models.cal_statics()

    models.rot_configs['noise_projection'] = args.noise_projection

    cal_loss_grad = jax.grad(partial(cal_loss, models), has_aux=True)

    edm_params = dfutil.EdmParams()

    def train_func(train_batch, params, opt_state, jkey):
        _, jkey = jax.random.split(jkey)
        grad, loss_dict = cal_loss_grad(params, edm_params, train_batch, jkey)
        nan_mask = jnp.any(jnp.array(jax.tree_util.tree_flatten(jax.tree_util.tree_map(lambda x: jnp.any(jnp.isnan(x)), grad))[0])).astype(jnp.float32)
        grad = jax.tree_util.tree_map(lambda x: jnp.where(jnp.isnan(x), 0, x), grad)
        grad = jax.tree_util.tree_map(lambda x: (1-nan_mask)*x, grad)
        grad, _ = optax.clip_by_global_norm(1.0).update(grad, None)
        _, jkey = jax.random.split(jkey)
        updates, opt_state = optimizer.update(grad, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, jkey, loss_dict
    
    @jax.jit
    def model_apply_param_jit(params, x, cond, t, cond_mask, jk):
        models_ = models.set_params(params)
        return models_.apply('denoiser', x, cond, t, cond_mask, confidence_out=True, rngs={'dropout':jk})

    def eval_func(models:mutil.Models, eval_batch:dict, params, jkey, itr, logs_dir, tb_writer:SummaryWriter):
        _, jkey = jax.random.split(jkey)
        models_ = models.set_params(params)

        obj_info = eval_batch["obj_info"]
        ns = eval_batch["cam_info"]['cam_intrinsics'].shape[0]
        
        pos_randomization = jax.random.uniform(jkey, (ns, 1, 3), minval=-0.1, maxval=0.1)
        _, jkey = jax.random.split(jkey)
        z_rotation = jax.random.uniform(jkey, (ns, 1, 1), minval=-np.pi, maxval=np.pi)
        _, jkey = jax.random.split(jkey)
        z_rotation = jnp.c_[jnp.zeros_like(z_rotation), jnp.zeros_like(z_rotation), z_rotation]
        z_rotation = tutil.aa2q(z_rotation)


        gt_obj_cvx = cxutil.CvxObjects().init_obj_info(obj_info)
        gt_obj_cvx = gt_obj_cvx.apply_pq_vtx(pos_randomization, z_rotation)
        gt_obj = gt_obj_cvx.set_z_with_models(jkey, models_, dc_center_no=args.volume_points_no)

        nd = gt_obj.nd
        nf = gt_obj.nf
        nz = gt_obj.nz
        nh = gt_obj.h.shape[-1]
        base_shape = (nd, nf, nz)

        vis_idx = 0
        intrinsic = eval_batch["cam_info"]['cam_intrinsics'].astype(np.float32)
        cam_posquat = eval_batch["cam_info"]['cam_posquats'].astype(np.float32)
        cam_posquat = tutil.pq_multi(pos_randomization, z_rotation, cam_posquat[...,:3], cam_posquat[...,3:])
        cam_posquat = jnp.concatenate(cam_posquat, axis=-1)
        vis_intrinsic = intrinsic[vis_idx,0]
        vis_cam_posquat = cam_posquat[vis_idx,0]
        pixel_size = (int(vis_intrinsic[1]), int(vis_intrinsic[0]))
        vis_cam_pos, vis_cam_quat = vis_cam_posquat[:3], vis_cam_posquat[3:]
        sdf_func = partial(rutil.scene_sdf, models=models_)
        latent_render_func = lambda: jax.jit(partial(rutil.cvx_render_scene, models=models_, sdf=sdf_func, pixel_size=pixel_size, 
                                                     intrinsic=vis_intrinsic, camera_pos=vis_cam_pos, camera_quat=vis_cam_quat, seg_out=True))
        latent_render_func_dict = {}
        latent_render_func_dict[args.ds_obj_no] = latent_render_func()
        latent_render_func_dict[args.nparticles] = latent_render_func()
        gt_obj_sq = jax.tree_util.tree_map(lambda x: x[vis_idx], gt_obj)

        # object to pixel
        def render_img(obj:cxutil.LatentObjects):
            rgb_, seg_ = latent_render_func_dict[obj.outer_shape[0]](obj)
            pixel_coord, out_ = cutil.global_pnts_to_pixel(vis_intrinsic, vis_cam_posquat, obj.dc_centers_tf) # (... NR)
            pixel_coord = np.array(pixel_coord).astype(np.int32).clip(0, vis_intrinsic[:2][::-1]-1).astype(np.int32)
            pixel_coord = np.where(np.isnan(pixel_coord), 0, pixel_coord)
            pixel_coord = np.where(pixel_coord<0, 0, pixel_coord)
            pixel_coord = np.where(pixel_coord>10000, 0, pixel_coord)
            rgb_ = np.array(rgb_)
            for i in range(pixel_coord.shape[0]):
                rgb_[pixel_coord[i,:,0], pixel_coord[i,:,1]] = np.ones(3)
            return rgb_, seg_

        if eval_batch["rgbs"] is None:
            cond = None
        else:
            # img_feat = models_.apply('img_encoder', eval_batch["rgbs"], train=True)
            # cond = structs.ImgFeatures(intrinsic, cam_posquat, img_feat)
            cond = models_.apply('img_encoder', eval_batch["rgbs"], cam_posquat, intrinsic)
            if args.train_seg:
                if args.separate_seg_model:
                    seg_pred = models_.apply('seg_predictor', eval_batch["rgbs"])
                else:
                    seg_pred = models_.apply('seg_predictor', cond)
                seg_pred = nn.sigmoid(seg_pred)
                seg_label = eval_batch["seg"].astype(jnp.float32)
                seg_iou = jnp.sum(seg_pred.squeeze(-1) * seg_label)/(jnp.sum(seg_pred.squeeze(-1) + seg_label - seg_pred.squeeze(-1) * seg_label) + 1e-6)
                seg_pred_vis = seg_pred[vis_idx, 0]
            else:
                seg_iou = 0
        
        # def model_apply_jit(x, cond, t, cond_mask, jk):
        #     return models_.apply('denoiser', x, cond, t, cond_mask, confidence_out=True, rngs={'dropout':jk})
        # # model_apply_jit = lambda x, cond, t, cond_mask, jk: models_.apply('denoiser', x, cond, t, cond_mask, confidence_out=True, rngs={'dropout':jk})
        # model_apply_jit = jax.jit(model_apply_jit)
        model_apply_jit = partial(model_apply_param_jit, params)

        rgb_gt = eval_batch["rgbs"][vis_idx][0] if eval_batch["rgbs"] is not None else np.zeros((*pixel_size, 3))

        # sampling
        if args.one_step_test in [1,2]:
            inf_50_start = time.time()
            x = jax.random.normal(jkey, shape=(ns, args.nparticles, nh))
            x = dfutil.noise_FER_projection(x, base_shape, rot_configs=models_.rot_configs)
            x = cxutil.LatentObjects().init_h(x, base_shape)
            x_pred_1002, _ = model_apply_jit(x, cond, jnp.array([1.]), None, jkey)
            x_pred_1002 = jax.block_until_ready(x_pred_1002)
            x_pred_1002_vis = jax.tree_map(lambda x: x[vis_idx], x_pred_1002)
            inf_50_end = time.time()
        else:
            _, jkey = jax.random.split(jkey)
            inf_5_start = time.time()
            x_pred_10 = dfutil.euler_sampler_obj((ns, args.nparticles, nh), base_shape, model_apply_jit, cond, jkey, dm_type=args.dm_type, 
                                                 max_time_steps=5, edm_params=edm_params, w=1.0 if args.cond_mask_ratio<1.0 else 1.0, 
                                                 rot_configs=models_.rot_configs, learnable_queries=models_.learnable_queries, output_obj_no=args.ds_obj_no)
            x_pred_10 = jax.block_until_ready(x_pred_10)
            inf_5_end = time.time()
            _, jkey = jax.random.split(jkey)
            inf_50_start = time.time()
            x_pred_100 = dfutil.euler_sampler_obj((ns, args.nparticles, nh), base_shape, model_apply_jit, cond, jkey, dm_type=args.dm_type, 
                                                  max_time_steps=50, edm_params=edm_params, w=1.0 if args.cond_mask_ratio<1.0 else 1.0, 
                                                  rot_configs=models_.rot_configs, learnable_queries=models_.learnable_queries, output_obj_no=args.ds_obj_no)
            x_pred_100 = jax.block_until_ready(x_pred_100)
            inf_50_end = time.time()
            _, jkey = jax.random.split(jkey)
            x_pred_1002, (x_dif_list, x_pred_list, _) = dfutil.euler_sampler_obj((ns, args.nparticles, nh), base_shape, model_apply_jit, cond, jkey, 
                                                                                dm_type=args.dm_type, max_time_steps=50, edm_params=edm_params, w=1.0, 
                                                                                rot_configs=models_.rot_configs, deterministic=True, sequence_out=True, 
                                                                                learnable_queries=models_.learnable_queries, output_obj_no=args.ds_obj_no)
            x_pred_1002 = jax.block_until_ready(x_pred_1002)
            x_pred_1003 = dfutil.euler_sampler_obj((ns, args.nparticles, nh), base_shape, model_apply_jit, cond, jkey, 
                                                   dm_type=args.dm_type, max_time_steps=20, edm_params=edm_params, w=1.0, 
                                                   rot_configs=models_.rot_configs, deterministic=True, 
                                                   learnable_queries=models_.learnable_queries, output_obj_no=args.ds_obj_no)
            _, jkey = jax.random.split(jkey)

            x_pred_10_vis, x_pred_100_vis, x_pred_1002_vis, x_pred_1003_vis, x_dif_list_vis, x_pred_list_vis = \
                jax.tree_map(lambda x: x[vis_idx], (x_pred_10, x_pred_100, x_pred_1002, x_pred_1003, x_dif_list, x_pred_list))
            
            def rgb_mix(rgb, pred, seg=None):
                alpha = 0.7
                if seg is not None:
                    pred_ = np.where(seg[...,None]>=0, pred, 0)
                else:
                    pred_ = pred
                return (1-alpha)*rgb/255. + pred_*alpha
            
            try:
                if (args.implicit_baseline==0) and (itr%(args.save_interval*10) == 0):
                    video_logs_dir = os.path.join(logs_dir, 'video')
                    os.makedirs(video_logs_dir, exist_ok=True)
                    video_arr = []
                    with rutil.VideoWriter(os.path.join(video_logs_dir, f'dif_{itr}.mp4'), fps=15) as vid:
                        for t in x_dif_list_vis:
                            video_arr.append(rgb_mix(rgb_gt, *render_img(t)))
                            vid(video_arr[-1])
                    tb_writer.add_video(
                        tag='dif_video',
                        vid_tensor=np.stack(video_arr)[None].transpose(0,1,4,2,3),
                        global_step=itr, fps=15)
                    video_arr = []
                    with rutil.VideoWriter(os.path.join(video_logs_dir, f'pred_{itr}.mp4'), fps=15) as vid:
                        for t in x_pred_list_vis:
                            video_arr.append(rgb_mix(rgb_gt, *render_img(t)))
                            vid(video_arr[-1])
                    tb_writer.add_video(
                        tag='pred_progress',
                        vid_tensor=np.stack(video_arr)[None].transpose(0,1,4,2,3),
                        global_step=itr, fps=15)
            except:
                print('fail to save videos')
        
        if args.one_step_test in [1,2]:
            rgb_10, seg_10 = render_img(x_pred_1002_vis)
            rgb_latent_gt, seg_gt = render_img(gt_obj_sq)

            fig_ = plt.figure()
            plt.subplot(2,2,1)
            plt.imshow(rgb_gt)
            plt.axis('off')
            plt.subplot(2,2,2)
            plt.imshow(rgb_latent_gt)
            plt.axis('off')
            plt.subplot(2,2,3)  
            plt.imshow(rgb_10)
            plt.axis('off')
            # plt.close()
            
        else:
            if args.implicit_baseline==0:
                if args.dm_type == 'edm':
                    t_list = [0.05, 0.5, 10.0]
                elif args.dm_type in ['ddpm', 'ddpm_noise']:
                    t_list = [0.1, 0.5, 1.0]
                ptb_rec_obj_list = []
                for t_ in t_list:
                    ptb_rec_obj_list.append(dfutil.perturb_recover_obj(gt_obj, model_apply_jit, cond, jnp.array([t_]), jkey, dm_type=args.dm_type, rot_configs=models_.rot_configs))
                    _, jkey = jax.random.split(jkey)
                ptb_rec_obj_list = jax.tree_map(lambda x: x[vis_idx], ptb_rec_obj_list)
                vis_obj_list = (gt_obj_sq.drop_gt_info(), x_pred_10_vis, x_pred_100_vis, x_pred_1002_vis, x_pred_1003_vis, *ptb_rec_obj_list)

                # if eval_batch["rgbs"] is None:
                #     obs_values = jnp.zeros((vis_obj_stack.shape[0],))
                # else:
                #     cond_obs_test = jax.tree_map(lambda x: x[vis_idx:vis_idx+1], cond)
                #     obs_values_per_obj = jax.jit(partial(models.apply,'obs_model'))(vis_obj_stack, cond_obs_test, jnp.array([0.008])).squeeze(-1)
                #     obs_values = obs_values_per_obj.mean(-1)

                rgb_list = []
                for vo in vis_obj_list:
                    rgb_list.append(render_img(vo)[0])

                fig_ = plt.figure()
                plt.subplot(3,3,1)
                if args.train_seg:
                    plt.imshow(rgb_mix(rgb_gt, seg_pred_vis))
                else:
                    plt.imshow(rgb_gt)
                plt.axis('off')
                for i in range(8):
                    plt.subplot(3,3,i+2)
                    # plt.imshow(rgb_mix(rgb_gt, rgb_list[i]))
                    plt.imshow(rgb_list[i])
                    # plt.title(f'{obs_values[i]:0.3f}')
                    plt.axis('off')
                # plt.close()
                
        if args.implicit_baseline==0:
            fig_.canvas.draw()
            rgb = np.array(fig_.canvas.renderer._renderer)
            plt.close()
        else:
            rgb = None

        # calculate 3D IoU
        def cal_3d_iou(qpnts, occ_label, obj_pred):
            if obj_pred is None:
                return 0
            obj_pred_rep = obj_pred
            occ_pred = models_.apply('occ_predictor', obj_pred_rep, qpnts)

            # calculate global IoU
            occ_pred = jnp.max(occ_pred, axis=-2)
            occ_label = jnp.max(occ_label, axis=-2)
            iou_mean = jnp.sum(jnp.logical_and(occ_pred > 0, occ_label > 0.5))/(jnp.sum(jnp.logical_or(occ_pred > 0, occ_label > 0.5))+1e-5)
            return iou_mean
        
        def cal_2d_iou(seg_label, obj_pred):
            if obj_pred is None:
                return 0
            seg_pred = rutil.obj_segmentation(models_, seg_label.shape[-2:], obj_pred, cond, smooth=1)
            seg_global = jnp.max(seg_pred, axis=-4) # (NS, NC, NI, NJ)
            seg_label = seg_label.astype(jnp.float32)
            iou_pred = jnp.sum(seg_label*seg_global, (-1,-2,-3))/(jnp.sum(jnp.maximum(seg_label,seg_global), (-1,-2,-3)) + 1e-6)
            return jnp.mean(iou_pred)


        if args.implicit_baseline==0:
            # optimization loss
            pixel_reduce_ratio = 2
            def opt_loss(h_in, cond, seg_obs):
                x_pred_ = cxutil.LatentObjects().init_h(h_in, models_.latent_shape)
                seg = rutil.obj_segmentation(models_, pixel_size, x_pred_, cond, 2, smooth=3)
                seg_global = jnp.max(seg, axis=-4) # (NS, NC, NI, NJ)
                obs_preds = jnp.sum(seg_obs*seg_global, (-1,-2,-3))/jnp.sum(1e-6 + seg_obs + seg_global - seg_obs*seg_global, (-1,-2,-3))
                return -jnp.sum(obs_preds), obs_preds
            
            opt_loss_jit = jax.jit(jax.grad(opt_loss, has_aux=True))
            seg_obs = eval_batch["seg"]
            def refinement(x_pred, seg_obs):
                rf_start_t = time.time()
                seg_obs_shape = seg_obs.shape
                seg_obs = cutil.resize_img(einops.rearrange(seg_obs[...,None], 'i j ... -> (i j) ...'), 
                                        (pixel_size[0]//pixel_reduce_ratio, pixel_size[1]//pixel_reduce_ratio), 'nearest').squeeze(-1).astype(jnp.float32)
                seg_obs = einops.rearrange(seg_obs, '(i j) ... -> i j ...', i=seg_obs_shape[0])
                cur_h = x_pred.h
                optimizer = optax.adamw(1e-2)
                opt_state = optimizer.init(cur_h)
                for i in range(4):
                    grad, loss_per_obj = opt_loss_jit(cur_h, cond, seg_obs)
                    updates, opt_state = optimizer.update(grad, opt_state, cur_h)
                    cur_h = optax.apply_updates(cur_h, updates)
                rf_end_t = time.time()
                return x_pred.set_h(cur_h), rf_end_t - rf_start_t
            x_ref1, ref_obs_t = refinement(x_pred_1002, seg_obs)
            if args.separate_seg_model:
                seg_obs_pred = models_.apply('seg_predictor', eval_batch["rgbs"])
            else:
                seg_obs_pred = models_.apply('seg_predictor', jax.lax.stop_gradient(cond))
            seg_obs_pred = jax.nn.sigmoid(seg_obs_pred).squeeze(-1)
            x_ref2, ref_pred_t = refinement(x_pred_1002, seg_obs_pred)
        else:
            x_ref1, x_ref2 = None, None
        qpnts_tf = tutil.pq_action(pos_randomization[...,None,:], z_rotation[...,None,:], eval_batch["qpnts"].astype(jnp.float32))

        gt_eval_iou = cal_3d_iou(qpnts_tf, eval_batch["occ_label"], gt_obj)
        if args.one_step_test in [1,2]:
            eval_metric_dict = {
                'eval_seg_iou': seg_iou,
                'eval_iou_gt':gt_eval_iou,
                'eval_iou_3':cal_3d_iou(qpnts_tf, eval_batch["occ_label"], x_pred_1002),
                'eval_seg_iou_3':cal_2d_iou(eval_batch["seg"], x_pred_1002),
                'eval_seg_iou_rf':cal_2d_iou(eval_batch["seg"], x_ref1),
                'eval_seg_iou_rf2':cal_2d_iou(eval_batch["seg"], x_ref2),
                'eval_iou_rf':cal_3d_iou(qpnts_tf, eval_batch["occ_label"], x_ref1),
                'eval_iou_rf2':cal_3d_iou(qpnts_tf, eval_batch["occ_label"], x_ref2),
                'eval_inf_50_time':inf_50_end - inf_50_start,
                **models.cal_statics(),
                }
        else:
            eval_metric_dict = {
                'eval_seg_iou': seg_iou,
                'eval_iou_gt':gt_eval_iou,
                'eval_seg_iou_1':cal_2d_iou(eval_batch["seg"], x_pred_10),
                'eval_seg_iou_2':cal_2d_iou(eval_batch["seg"], x_pred_100),
                'eval_seg_iou_3':cal_2d_iou(eval_batch["seg"], x_pred_1002),
                'eval_seg_iou_4':cal_2d_iou(eval_batch["seg"], x_pred_1003),
                'eval_seg_iou_rf':cal_2d_iou(eval_batch["seg"], x_ref1),
                'eval_seg_iou_rf2':cal_2d_iou(eval_batch["seg"], x_ref2),
                'eval_iou_1':cal_3d_iou(qpnts_tf, eval_batch["occ_label"], x_pred_10),
                'eval_iou_2':cal_3d_iou(qpnts_tf, eval_batch["occ_label"], x_pred_100),
                'eval_iou_3':cal_3d_iou(qpnts_tf, eval_batch["occ_label"], x_pred_1002),
                'eval_iou_4':cal_3d_iou(qpnts_tf, eval_batch["occ_label"], x_pred_1003),
                'eval_iou_rf':cal_3d_iou(qpnts_tf, eval_batch["occ_label"], x_ref1),
                'eval_iou_rf2':cal_3d_iou(qpnts_tf, eval_batch["occ_label"], x_ref2),
                'eval_inf_5_time':inf_5_end - inf_5_start,
                'eval_inf_50_time':inf_50_end - inf_50_start,
                **models.cal_statics(),
                }

        return rgb, eval_metric_dict
    
    now = datetime.datetime.now() # current date and time
    date_time = now.strftime("%m%d%Y-%H%M%S")
    logs_dir = os.path.join(args.logs_dir, date_time)
    os.makedirs(logs_dir, exist_ok=True)
    writer = SummaryWriter(logs_dir)
    writer.add_text('args', args.__str__(), 0)

    if args.debug:
        for eval in eval_loader:
            eval_ds = eval
            break
        eval_func(models, eval_ds, params, jkey, 0, logs_dir, writer)
        print('end debug eval')
        train_func_jit = train_func
    else:
        train_func_jit = jax.jit(train_func)

    shutil.copy(__file__, logs_dir)
    if args.checkpoint_dir!='none':
        shutil.copy(os.path.join(args.checkpoint_dir, 'dif_model_util.py'), os.path.join(logs_dir, 'dif_model_util.py'))
        shutil.copy(os.path.join(args.checkpoint_dir, 'model_util.py'), os.path.join(logs_dir, 'model_util.py'))
    else:
        shutil.copy(os.path.join(BASEDIR, 'util/dif_model_util.py'), os.path.join(logs_dir, 'dif_model_util.py'))
        shutil.copy(os.path.join(BASEDIR, 'util/model_util.py'), os.path.join(logs_dir, 'model_util.py'))

    # Train loop
    ema_params = params
    ema_ratio = 0.9999
    for ep in range(10000):

        # Max epoch control
        if train_epochs_cummulative >= args.max_epochs:
            break

        # Train routine from here...
        print("Train loop")
        train_loss_dict = None
        tr_itr_cnt = 0
        nan_cnt = 0
        for tr_itr, train_batch in enumerate(tqdm(train_loader, total=train_ds_len)):
            params, opt_state, jkey, train_loss_dict_ = train_func_jit(train_batch, params, opt_state, jkey)
            isnan_loss = jnp.any(jnp.isnan(jnp.array(jax.tree_util.tree_flatten(train_loss_dict_)[0])))
            if isnan_loss:
                nan_cnt += 1
            if train_loss_dict is None:
                if not isnan_loss:
                    train_loss_dict = train_loss_dict_
                    tr_itr_cnt += 1
            else:
                if not isnan_loss:
                    train_loss_dict = jax.tree_map(lambda x, y: x+y, train_loss_dict, train_loss_dict_)
                    tr_itr_cnt += 1
            if train_steps_cummulative > 10:
                ema_params = jax.tree_map(lambda x,y: (1-ema_ratio)*x+ema_ratio*y, ema_params, params)
            else:
                ema_params = params
            _, jkey = jax.random.split(jkey)

        # Logging
        train_steps_cummulative += tr_itr_cnt
        train_epochs_cummulative += 1
        train_loss_dict = jax.tree_map(lambda x: x/tr_itr_cnt, train_loss_dict)
        train_loss_dict['nan_count'] = nan_cnt
        train_loss_dict['train_steps_cummulative'] = train_steps_cummulative
        train_loss_dict['train_epochs_cummulative'] = train_epochs_cummulative
        print(train_loss_dict)
        # print(f'During training, we have {nan_cnt} nan values')
        # Eval loop
        # eval: sg.SceneCls.SceneData
        if train_epochs_cummulative%args.save_interval == 0:
            print("Eval loop")
            for eval in eval_loader:
                eval_ds = eval
                break
            
            print(f'start evaluations')
            rgb, eval_metric_dict = eval_func(models, eval_ds, ema_params, jkey, train_epochs_cummulative, logs_dir, writer)
            _, jkey = jax.random.split(jkey)

            print(f'save checkpoints to {logs_dir}')
            with open(os.path.join(logs_dir, 'saved.pkl'), 'wb') as f:
                pickle.dump({
                    'params': params, 
                    'ema_params': ema_params, 
                    'dif_args': models.dif_args, 
                    'models': models.set_params(params),
                    'opt_state': opt_state, 
                    'train_steps_cummulative': train_steps_cummulative,
                    'train_epochs_cummulative': train_epochs_cummulative
                }, f)
            with open(os.path.join(logs_dir, f'saved{(train_epochs_cummulative//50)*50}.pkl'), 'wb') as f:
                pickle.dump({
                    'params': params, 
                    'ema_params': ema_params, 
                    'dif_args': models.dif_args,
                    'models': models.set_params(params),
                    'opt_state': opt_state, 
                    'train_steps_cummulative': train_steps_cummulative,
                    'train_epochs_cummulative': train_epochs_cummulative
                }, f)
        # if itr%args.log_interval == 0:
            log_dict = {**train_loss_dict, **eval_metric_dict}
            if args.implicit_baseline==0:
                writer.add_image('pred', (np.moveaxis(rgb[...,:3], -1, 0)), train_epochs_cummulative)
            for k in log_dict:
                writer.add_scalar(k, np.array(log_dict[k]), train_epochs_cummulative)
            if vessl_on:
                base_name = "mv_"
                log_dict = {base_name+k: log_dict[k] for k in log_dict}
                vessl.log(step=train_epochs_cummulative, payload=log_dict)
                if (args.implicit_baseline==0) and (train_epochs_cummulative%10 == 0):
                    vs_images = [vessl.Image(data=np.array(rgb), caption=f'initr{train_epochs_cummulative:05.0f}')]
                    vessl.log({"Dif inf": vs_images})
            print(train_epochs_cummulative, log_dict)

        train_loader.dataset.push()
        if args.ds_obj_no >= 6:
            # push again
            train_loader.dataset.push()
            train_loader.dataset.push()

if __name__=="__main__":        

    parser = argparse.ArgumentParser()
    parser.add_argument("--max_epochs", type=int, default=1500)
    parser.add_argument("--data_dir", type=str, default=f"{str(BASEDIR/'data'/'scene_data')}")
    parser.add_argument("--checkpoint_dir", type=str, default="none")
    parser.add_argument("--update_pretraining_weight", type=int, default=0)
    parser.add_argument("--logs_dir", type=str, default="logs_dif")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dm_type", type=str, default='ddpm')
    parser.add_argument("--img_base_dim", type=int, default=8)
    parser.add_argument("--base_dim", type=int, default=128)
    parser.add_argument("--nparticles", type=int, default=8)
    parser.add_argument("--ds_obj_no", type=int, default=7)
    parser.add_argument("--dc_pos_loss_coef", type=float, default=90)
    parser.add_argument("--pos_loss_coef", type=float, default=200)
    parser.add_argument("--cond_mask_ratio", type=float, default=0.95)
    parser.add_argument("--save_interval", type=int, default=2)
    parser.add_argument("--nrec", type=int, default=4)
    parser.add_argument("--loss_type", type=str, default='ec')
    parser.add_argument("--add_loss_weight", type=int, default=0)
    parser.add_argument("--add_c_skip", type=int, default=0)
    parser.add_argument("--dif_model_type", type=int, default=1,
                        help="shape head type in denoiser: 1-MLP head / 2-FER head with p / 3-FER head without p")
    parser.add_argument("--dif_model_version", type=int, default=1, 
                        help="diffusion model version 1-ours / 2-ours old / 4-PARQ (image cross attention) / 5-Voxel cross attention / 6-Deformable DETR")
    parser.add_argument("--debug", type=int, default=0)
    parser.add_argument("--single_obj_test", type=int, default=0)
    parser.add_argument("--one_step_test", type=int, default=0)
    parser.add_argument("--category", type=str, default="none")
    parser.add_argument("--ds_size_limit", type=int, default=20_000)
    parser.add_argument("--add_t_sample_bias", type=int, default=0)
    parser.add_argument("--train_obs_model", type=int, default=0)
    parser.add_argument("--obj_matching", type=str, default='em')
    parser.add_argument("--global_representation", type=int, default=1)
    parser.add_argument("--pool_context", type=int, default=0)
    parser.add_argument("--volume_points_no", type=int, default=0)
    parser.add_argument("--train_loss_type", type=str, default='cf')
    parser.add_argument("--matching_dc_factor", type=float, default=0)
    parser.add_argument("--matching_pos_factor", type=float, default=10)
    parser.add_argument("--skip_rec_ratio", type=float, default=0.9)
    parser.add_argument("--train_seg", type=float, default=0.003)
    parser.add_argument("--conf_loss_weight", type=float, default=10)
    parser.add_argument("--dataset", type=str, default='NOCS')
    parser.add_argument("--dif_func", type=str, default='se')
    parser.add_argument("--voxel_resolution", type=int, default=25)
    parser.add_argument("--mixing_head", type=int, default=4, help="number of heads in denoiser")
    parser.add_argument("--first_mixing_depth", type=int, default=2, help="number of iteration for Transformer decoder in denoiser")
    parser.add_argument("--second_mixing_depth", type=int, default=0, help="use self-attention in shape head")
    parser.add_argument("--spatial_pe", type=int, default=1, help="use spatial positional encoding in PETR")
    parser.add_argument("--use_p", type=int, default=1, help="use geometric representative points in denoiser")
    parser.add_argument("--noise_projection", type=int, default=1)
    parser.add_argument("--parq_resize", type=int, default=1)
    parser.add_argument("--parq_resize_ratio", type=float, default=0.75)
    parser.add_argument("--recurrent_in_first_mixing", type=int, default=0)
    parser.add_argument("--implicit_baseline", type=int, default=0)
    parser.add_argument("--separate_seg_model", type=int, default=0)
    args = parser.parse_args()
    main(args)
