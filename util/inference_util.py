from functools import partial
import jax.numpy as jnp
import jax
import os, sys
import numpy as np
import copy
from pathlib import Path
import optax
import time
import logging
import einops
import matplotlib.pyplot as plt
import datetime
from typing import Tuple, NamedTuple

BASEDIR = Path(__file__).parent.parent
if str(BASEDIR) not in sys.path:
    sys.path.insert(0, str(BASEDIR))

import util.model_util as mutil
import util.render_util as rutil
import util.cvx_util as cxutil
import util.camera_util as cutil
import util.structs as structs
import util.diffusion_util as dfutil
import util.franka_util as fkutil
import util.environment_util as envutil
import util.rrt as rrtutil
import util.transform_util as tutil
import util.PRM as PRM
import util.io_util as ioutil


class TimeTracker(object):
    
    def __init__(self):
        self.t = {}
        self.dt = {}
    
    def set(self, name):
        if name in self.t:
            self.dt[name] = time.time()-self.t[name]
            # self.print(name)
            self.t.pop(name)
        else:
            self.t[name] = time.time()
    
    def str_dt_all(self):
        str_res = ''
        for k in self.dt:
            str_res = str_res + f'{k} time: {self.dt[k]} // '
        return str_res

    def print_all(self):
        for k in self.dt:
            print(k + f' time: {self.dt[k]}')

    def get_dt(self):
        return copy.deepcopy(self.dt)

    def print(self, name):
        print(name + f' time: {time.time()-self.t[name]}')


class InfCls(object):
    """Warpper for inference"""

    class InferenceAux(NamedTuple):
        x_pred: cxutil.LatentObjects
        obs_preds: jnp.ndarray
        conf: jnp.ndarray
        obs_max_idx: jnp.ndarray
        seg_obs_original_size: jnp.ndarray
        det_mask: jnp.ndarray
        w_dist: jnp.ndarray
        instance_seg: jnp.ndarray


    def __init__(
            self, 
            ckpt_dir: str,
            ns: int, 
            conf_threshold: float, 
            ckpt_itr_no=None,
            overay_obj_no: int = 1, 
            max_time_steps: int = 15, 
            optim_lr: float = 4e-3,
            apply_in_range: bool = False, 
            valid_range: float = 2.,
            scene_obj_no: int = None, 
            cam_offset: jnp.ndarray = jnp.zeros(3), 
            optimization_step:int=100,
            guidance_grad_step:int=5,
            shape_optimization:bool=True,
            gradient_guidance:bool=False,
            early_reduce_sample_size:int=2,
            apply_nms:bool=True,
            save_images: bool = False, 
            save_videos: bool = False,
            log_dir=None,
    ):
        """ Init?
        
        Args:
            ckpt_dir (str): Checkpoint directory
            ns (int): Batch size
            conf_threshold (float): Confidence threshold
            overay_obj_no (int): ???
            max_time_steps (int): ???
            optim_lr (float): ???
            apply_in_range (bool): ???
            valid_range (float): ???
            scene_obj_no (int): ???
            cam_offset (jnp.ndarray): ???
            save_images (bool): ???
            save_videos (bool): ???
        """
        self.models = mutil.get_models_from_cp_dir(ckpt_dir, ckpt_itr_no)
        self.max_obj_no = self.models.dif_args.nparticles
        self.scene_obj_no = scene_obj_no
        self.cam_offset = cam_offset
        self.overay_obj_no = overay_obj_no if not self.models.dif_args.one_step_test else 1
        self.ns = ns if not self.models.dif_args.one_step_test else 1
        self.unique_id = 0
        self.apply_in_range = apply_in_range
        self.valid_range = valid_range
        self.conf_threshold = conf_threshold
        self.max_time_steps = max_time_steps if not self.models.dif_args.one_step_test else 5
        self.optim_lr = optim_lr
        self.save_images = save_images
        self.save_videos = save_videos
        self.pixel_reduce_ratio = 2
        self.save_time_intervals = {'est':[], 'opt':[], 'total':[]}
        self.optimization_step = optimization_step
        self.shape_optimization = shape_optimization
        self.gradient_guidance = gradient_guidance
        self.obj_segment_prediction_smooth_factor = 1.0
        self.guidance_grad_step = guidance_grad_step
        self.early_reduce_sample_size = early_reduce_sample_size if not self.models.dif_args.one_step_test else None
        self.apply_nms = apply_nms if not self.models.dif_args.one_step_test else False

        # Log dir
        if save_videos or save_images:
            if log_dir is None:
                now = datetime.datetime.now() # current date and time
                date_time = now.strftime("%m%d%Y-%H%M%S")
                logs_dir = os.path.join('logs_est', date_time)
                os.makedirs(logs_dir, exist_ok=True)
                self.logs_dir = logs_dir
            else:
                self.logs_dir = log_dir

        # Forward 
        self.get_img_feat = jax.jit(self.__get_img_feat)
        self.model_apply_jit = jax.jit(self.__model_apply_jit)
        self.seg_pred_func = jax.jit(self.__seg_pred_func)
        if self.apply_in_range:
            self.within_range_func = self.__within_range_func
        else:
            self.within_range_func = None
        
        # self.obs_loss_jit = jax.jit(jax.grad(self.__obs_loss, has_aux=True))
        # self.obj_seg_jit = jax.jit(lambda x, cn: rutil.obj_segmentation(self.models, self.models.pixel_size, x, cn, self.pixel_reduce_ratio))
        # jit with obj no
        self.obs_loss_jit_per_obj_no = {True:{}, False:{}}
        def get_obs_loss_jit(obj_no, only_plane=False):
            if obj_no not in self.obs_loss_jit_per_obj_no[only_plane]:
                print(f'comple obs loss jit with obj no: {obj_no} / {only_plane}')
                self.obs_loss_jit_per_obj_no[only_plane][obj_no] = jax.jit(jax.grad(partial(self.__obs_loss, only_plane=only_plane), has_aux=True))
            return self.obs_loss_jit_per_obj_no[only_plane][obj_no]
        self.get_obs_loss_jit = get_obs_loss_jit

        self.obj_seg_jit_per_obj_no = {}
        def get_obj_seg_jit(obj_no, sample_no):
            if sample_no not in self.obj_seg_jit_per_obj_no:
                self.obj_seg_jit_per_obj_no[sample_no] = {}
                print(f'comple obj seg jit with sample_no: {sample_no}')
                self.obj_seg_jit_per_obj_no[sample_no][obj_no] = jax.jit(lambda x, cn: rutil.obj_segmentation(self.models, self.models.pixel_size, x, cn, 
                                                                            self.pixel_reduce_ratio, smooth=self.obj_segment_prediction_smooth_factor, 
                                                                            gaussian_filter=False))
            elif obj_no not in self.obj_seg_jit_per_obj_no[sample_no]:
                    print(f'comple obj seg jit with obj no: {obj_no}')
                    self.obj_seg_jit_per_obj_no[sample_no][obj_no] = jax.jit(lambda x, cn: rutil.obj_segmentation(self.models, self.models.pixel_size, x, cn, 
                                                                                self.pixel_reduce_ratio, smooth=self.obj_segment_prediction_smooth_factor, 
                                                                                gaussian_filter=False))

            return self.obj_seg_jit_per_obj_no[sample_no][obj_no]
        self.get_obj_seg_jit = get_obj_seg_jit

        # Sampler
        self.euler_sampler_obj_jit = jax.jit(partial(dfutil.euler_sampler_obj_fori, 
            x_shape = (self.ns, self.max_obj_no, self.models.nh), 
            base_shape = self.models.latent_shape, 
            model_apply_func = self.model_apply_jit,
            max_time_steps = self.max_time_steps,
            guidance_grad_step=self.guidance_grad_step,
            in_range_func = self.within_range_func,
            dm_type = self.models.dif_args.dm_type, 
            edm_params = None,
            rot_configs = self.models.rot_configs, 
            sequence_out = True, 
            conf_threshold = self.conf_threshold, 
            conf_filter_out = False,
            gradient_guidance_func=get_obs_loss_jit(self.models.dif_args.nparticles) if self.gradient_guidance else None,
            learnable_queries = self.models.learnable_queries ))

        self.euler_sampler_obj_refinement_jit = jax.jit(partial(dfutil.euler_sampler_obj_fori, 
            x_shape = (self.ns, self.max_obj_no, self.models.nh), 
            base_shape = self.models.latent_shape, 
            model_apply_func = self.model_apply_jit,
            max_time_steps = 20,
            guidance_grad_step=self.guidance_grad_step,
            in_range_func = self.within_range_func,
            dm_type = self.models.dif_args.dm_type, 
            edm_params = None,
            rot_configs = self.models.rot_configs, 
            sequence_out = True, 
            conf_threshold = self.conf_threshold, 
            conf_filter_out = False,
            gradient_guidance_func=get_obs_loss_jit(self.models.dif_args.nparticles) if self.gradient_guidance else None,
            start_time_step=15,
            learnable_queries = self.models.learnable_queries ))

        # Some useful functions...
        self.resize_intrinsic_fn_batched = jax.vmap(cutil.resize_intrinsic, in_axes=(0, None, None))
        self.resize_img_fn_batched = jax.vmap(cutil.resize_img, in_axes=(0, None))

        self.pcd_sample_func = jax.jit(partial(cxutil.get_pcd_from_latent_w_voxel, 
                        num_points=self.models.args.npoint, models=self.models, visualize=False))


    def __obs_loss(self, h_in, cond, seg_obs, valid_seg_mask, plane_params, only_plane=False):
        """Observation loss"""
        x_pred_ = cxutil.LatentObjects().init_h(h_in, self.models.latent_shape)
        valid_x_pred_mask = jnp.all(x_pred_.pos<8.0, axis=-1)
        x_pred_ = cxutil.LatentObjects().init_h(jnp.where(valid_x_pred_mask[...,None], h_in, jax.lax.stop_gradient(h_in)), self.models.latent_shape)
        
        if not self.shape_optimization or only_plane:
            # remove shape optimization
            x_pred_ = x_pred_.replace(z=jax.lax.stop_gradient(x_pred_.z), dc_rel_centers=jax.lax.stop_gradient(x_pred_.dc_rel_centers))
        
        if not only_plane:
            seg = rutil.obj_segmentation(self.models, self.models.pixel_size, x_pred_, cond, self.pixel_reduce_ratio, 
                                         smooth=self.obj_segment_prediction_smooth_factor, gaussian_filter=False)
            seg_global = jnp.max(seg, axis=-4) # (NS, NC, NI, NJ)
            seg_global = jnp.where(valid_seg_mask[None,:,None,None], seg_global, 0)
            obs_preds = jnp.sum(seg_obs*seg_global, (-1,-2,-3))/jnp.sum(1e-6 + seg_obs + seg_global - seg_obs*seg_global, (-1,-2,-3))
        else:
            obs_preds = 0

        # add plane constraint
        plane_col_cost = 0
        if plane_params is not None:
            plane_col = self.models.apply('pln_predictor', x_pred_, plane_params[:,:3]*plane_params[:,3:], plane_params[:,:3])
            # surrogate_loss = jnp.where(jnp.min(plane_col, axis=-1) < 0.1, -x_pred_.pos[...,-1], 0)
            # plane_col_cost = jnp.sum(jax.nn.relu(-plane_col), axis=-1)
            plane_col_cost = 0.1*jnp.sum(jnp.abs(plane_col), axis=-1)
            # plane_col_cost = 1e5*jnp.sum(surrogate_loss, axis=-1)
            plane_col_cost = jnp.sum(plane_col_cost)

        return -jnp.sum(obs_preds) + jnp.sum(plane_col_cost), (obs_preds, plane_col_cost)


    def __get_img_feat(self, rgbs, cam_intrinsics, cam_posquats):
        """???"""
        cond = self.models.apply('img_encoder', rgbs, cam_posquats, cam_intrinsics, train=False)
        return cond
    

    def __model_apply_jit(self, x, cond, t, cond_mask, jk):
        """???"""
        if self.models.dif_args.dif_model_version == 3 and cond.img_feat.shape[0]==1 and x.outer_shape[0]!=1:
            cond = jax.tree_map(lambda x: einops.repeat(x, 'i ... -> (r i) ...', r=self.ns), cond)
        return self.models.apply('denoiser', x, cond, t, cond_mask, confidence_out=True, rngs={'dropout':jk})
    

    def __within_range_func(self, pos):
        """???"""
        in_range = jnp.all(jnp.abs(pos[...,:2]) < self.valid_range, axis=-1, keepdims=True)
        in_range = jnp.logical_and(in_range, jnp.abs(pos[...,2:3])<0.3)
        return in_range


    def __seg_pred_func(self, cond):
        """???"""
        # seg = jax.nn.sigmoid(self.models.apply('seg_predictor', cond))
        seg = jax.nn.sigmoid(3*self.models.apply('seg_predictor', cond))
        return seg


    def compile_jit(self, nv: int):
        """Feedforward through jitted functions once for pre-compilation.
        
        Args:
            nv (int): Number of views
        """
        print('start jit-precompiling for estimation')
        if self.models.pixel_size is None:
            pixel_size = (64,112)
        else:
            pixel_size = self.models.pixel_size

        rgbs = jnp.zeros((1, nv, *pixel_size, 3), dtype=jnp.uint8)
        cam_intrinsics = jnp.zeros((1, nv, 6), dtype=jnp.float32)
        cam_posquat = jnp.zeros((1, nv, 7), dtype=jnp.float32)
        cond = self.get_img_feat(rgbs, cam_intrinsics, cam_posquat)
        cond = jax.tree_map(lambda x: einops.repeat(x, 'i ... -> (r i) ...', r=self.ns), cond)
        x_dummy = cxutil.LatentObjects().init_h(jnp.zeros((self.ns, self.max_obj_no, self.models.nh)), self.models.latent_shape)
        # sampling
        jkey = jax.random.PRNGKey(0)
        self.model_apply_jit(x_dummy, cond, jnp.array(1.0), jnp.array([True]), jkey)

        seg_obs_dummy = jnp.zeros((1, nv, pixel_size[0]//self.pixel_reduce_ratio, pixel_size[1]//self.pixel_reduce_ratio))
        # self.obs_loss_jit(x_dummy.h, cond, seg_obs_dummy)

        print('end jit-precompiling for estimation')


    def init_renderer(self, vis_pixel_size, sdf_ratio=2200, table_height=-0.80):
        self.vis_pixel_size = vis_pixel_size
        sdf_func = partial(rutil.scene_sdf, models=self.models, sdf_ratio=sdf_ratio, floor_offset=table_height)
        self.latent_render_func = jax.jit(partial(rutil.cvx_render_scene, models=self.models, 
                                                  sdf=sdf_func, pixel_size=vis_pixel_size, floor_offset=table_height, 
                                                  light_dir=jnp.array([-0.533,0.133,0.866]), seg_out=True))


    # object to pixel
    def render_img(
            self, 
            obj: cxutil.LatentObjects, 
            vis_intrinsic: jnp.ndarray, 
            vis_cam_posquat: jnp.ndarray, 
            conf: float = None
    ) -> jnp.ndarray:
        """???

        Args:
            ???

        Returns:
            jnp.ndarray: rendered latent objects
            jnp.ndarray: rendered instance segmentation
        """
        if conf is not None and obj.outer_shape[0] != 1:
            conf_th = self.conf_threshold
            obj = obj.replace(pos=jnp.where(conf[...,None]>=conf_th, obj.pos, jnp.array([0,0,10.0])))
            obj = obj.replace(dc_rel_centers=jnp.where(conf[...,None,None]>=conf_th, obj.dc_rel_centers, 0))
        vis_cam_pos, vis_cam_quat = vis_cam_posquat[:3], vis_cam_posquat[3:]
        rgb_, seg = self.latent_render_func(obj, intrinsic=vis_intrinsic, camera_pos=vis_cam_pos, camera_quat=vis_cam_quat)
        # pixel_coord, out_ = cutil.global_pnts_to_pixel(vis_intrinsic, vis_cam_posquat, obj.dc_centers_tf)
        # pixel_coord = np.array(pixel_coord).astype(np.int32).clip(0, vis_intrinsic[:2][::-1]-1).astype(np.int32)
        # pixel_coord = np.where(np.isnan(pixel_coord), 0, pixel_coord)
        # pixel_coord = np.where(pixel_coord<0, 0, pixel_coord)
        # pixel_coord = np.where(pixel_coord>10000, 0, pixel_coord)
        # rgb_ = np.array(rgb_)
        # for i in range(pixel_coord.shape[0]):
        #     rgb_[pixel_coord[i,:,0], pixel_coord[i,:,1]] = np.ones(3)
        return rgb_, seg

    
    def estimation(
            self, 
            jkey: jax.Array, 
            pixel_size: Tuple[int, int], 
            rgbs_origin: jnp.ndarray, 
            cam_intrinsics_origin: jnp.ndarray, 
            cam_posquats: jnp.ndarray, 
            plane_params: jnp.ndarray = None,
            previous_obj_pred: cxutil.LatentObjects = None,
            out_valid_obj_only: bool = False, 
            apply_conf_filter: bool = True,
            debug_seg_obs: jnp.ndarray = None, 
            filter_before_opt: bool=False,
            verbose = 0,
    ):
        """_summary_

        Args:
            jkey (jax.Array): RNG
            pixel_size (Tuple[int, int]): ???
            rgbs_origin (jnp.ndarray): RGB with shape of [NB, NV, H, W, 3] or [NV, H, W, 3]
            cam_intrinsics_origin (jnp.ndarray): _description_
            cam_posquats (jnp.ndarray): _description_
            out_valid_obj_only (bool, optional): _description_. Defaults to False.
            apply_conf_filter (bool, optional): _description_. Defaults to True.
            debug_seg_obs (jnp.ndarray): Predefined segmentation for debug purpose. Defaults to None.
            verbose (int): Logging verbose level

        Raises:
            ValueError: _description_

        Returns:
            _type_: _description_
        """

        with ioutil.context_profiler("est_time", print_log=False) as est_time:
            # Preprocessing
            #   Batchify [NV, H, W, 3] -> [NB, NV, H, W, 3]
            if len(rgbs_origin.shape) == 4:
                rgbs_origin, cam_intrinsics_origin, cam_posquats = map(lambda x:x[None], (rgbs_origin, cam_intrinsics_origin, cam_posquats))
            #   Resize (space to perform seg optimization)
            origin_pixel_size = rgbs_origin.shape[-3:-1]
            rgbs = self.resize_img_fn_batched(rgbs_origin, pixel_size)
            cam_intrinsics = self.resize_intrinsic_fn_batched(cam_intrinsics_origin, origin_pixel_size, pixel_size)

            #   Re-center OOD absolute position
            cam_posquats_offset = copy.deepcopy(cam_posquats)
            cam_posquats_offset[...,:3] = cam_posquats_offset[...,:3] - self.cam_offset

            # Sampling
            cond = self.get_img_feat(rgbs, cam_intrinsics, cam_posquats_offset)

            # Get seg_mask
            seg_obs_size = (pixel_size[0]//self.pixel_reduce_ratio, pixel_size[1]//self.pixel_reduce_ratio)
            if debug_seg_obs is not None:
                seg_obs = cutil.resize_img(debug_seg_obs[...,None], seg_obs_size, 'nearest').squeeze(-1).astype(jnp.float32)
            else:
                seg_obs_original_size = self.seg_pred_func(cond).squeeze(0)
                valid_obs_mask = jnp.sum(rgbs[...,:5,:5,:].squeeze(0).astype(jnp.int32),(-1,-2,-3)) != 0
                seg_obs_original_size = jnp.where(valid_obs_mask[...,None,None,None], seg_obs_original_size, 0)
                seg_obs = cutil.resize_img(seg_obs_original_size, seg_obs_size, 'nearest').squeeze(-1).astype(jnp.float32)

            # w_dist = jnp.where(jax.random.uniform(jkey, shape=(ns,))<0.7, 1.0, jax.random.uniform(jkey, shape=(ns,), minval=1.5, maxval=7.0))
            w_dist = jnp.ones((self.ns,))
            jkey, subkey = jax.random.split(jkey)
            det_mask = jax.random.uniform(subkey, shape=(self.ns,))<0.5
            jkey, subkey = jax.random.split(jkey)
            if previous_obj_pred is None:
                x_pred, dif_process_aux_info = self.euler_sampler_obj_jit(
                    cond=cond, jkey=subkey, deterministic=det_mask, w=w_dist, guidance_func_args=(cond, seg_obs, valid_obs_mask, plane_params))
                
                # Confidence averaging along last a few denoising steps
                k = 1 
                conf = jnp.mean(jnp.concatenate(dif_process_aux_info[-1][-k:], -1), axis=-1)

            else:
                # previous_obj_pred_extended = previous_obj_pred.extend_and_repeat_outer_shape(self.ns, 0)
                # x_pred = dfutil.forward_process_obj(previous_obj_pred_extended, 0.02, subkey, 'ddpm', None, 1, self.models.rot_configs)
                # conf = jnp.ones(previous_obj_pred_extended.outer_shape, dtype=bool)
                # dif_process_aux_info = None

                # padding first
                nparticles = self.models.dif_args.nparticles
                previous_nobj = previous_obj_pred.outer_shape[-1]
                jkey, subkey = jax.random.split(jkey)
                x = jax.random.normal(subkey, shape=(nparticles,previous_obj_pred.h.shape[-1]))
                jkey, subkey = jax.random.split(jkey)
                x = dfutil.noise_FER_projection(x, self.models.latent_shape, self.models.rot_configs)
                previous_obj_pred_padded = x.at[:previous_nobj].set(previous_obj_pred.h)
                previous_obj_pred_padded = previous_obj_pred.set_h(previous_obj_pred_padded)

                # apply refinement
                x_pred, dif_process_aux_info = self.euler_sampler_obj_refinement_jit(
                    cond=cond, jkey=subkey, deterministic=det_mask, w=w_dist, previous_x_pred=previous_obj_pred_padded, guidance_func_args=(cond, seg_obs, valid_obs_mask, plane_params))
                k = 1
                conf = jnp.mean(jnp.concatenate(dif_process_aux_info[-1][-k:], -1), axis=-1)

            if self.max_obj_no==1:
                conf = jnp.ones_like(conf)  
            
            # Send inconfident objects far far away until not visible
            if apply_conf_filter:
                if self.within_range_func is not None:
                    in_range = self.within_range_func(x_pred.pos).squeeze(-1)
                    keep_mask = jnp.logical_and(conf>self.conf_threshold, in_range)
                else:
                    keep_mask = conf>self.conf_threshold
                x_pred:cxutil.LatentObjects = x_pred.replace(pos=jnp.where(keep_mask[...,None], x_pred.pos, jnp.array([0,0,10.])))
            x_pred = jax.block_until_ready(x_pred)

        # Filter out object in invalid region
        if filter_before_opt:
            # Compose mask
            # sort by conf
            # sort_idx = jnp.argsort(-conf + 100*self.within_range_func(x_pred.pos).squeeze(-1), axis=-1)
            conf_mask = conf>self.conf_threshold
            in_range_mask = self.within_range_func(x_pred.pos).squeeze(-1)
            keep_mask = jnp.ones_like(conf_mask, dtype=bool)
            sort_value = 0
            if apply_conf_filter:
                keep_mask = jnp.logical_and(keep_mask, conf_mask)
                sort_value = sort_value-conf
            if self.within_range_func:
                keep_mask = jnp.logical_and(keep_mask, in_range_mask)
                sort_value = sort_value-100*in_range_mask
            valid_obj_no = jnp.max(jnp.sum(keep_mask, axis=-1))
            valid_obj_no = int(valid_obj_no)
            if self.scene_obj_no is not None:
                valid_obj_no = self.scene_obj_no
        else:
            sort_value = -conf
            valid_obj_no = x_pred.outer_shape[1]

        sort_idx = jnp.argsort(sort_value, axis=-1)
        valid_obj_idx = sort_idx[...,:valid_obj_no]

        def take_with_outer_filling(x_, idx_):
            for _ in range(x_.ndim - idx_.ndim):
                idx_ = idx_[...,None]
            return jnp.take_along_axis(x_, idx_, axis=1)
        x_pred, conf = jax.tree_map(lambda x: take_with_outer_filling(x, valid_obj_idx), (x_pred, conf))

        if verbose >= 1:
            logging.info(f'valid obj no before optimization: {valid_obj_no}')
        # else:
        #     valid_obj_no = x_pred.outer_shape[1]

        with ioutil.context_profiler("opt_time", print_log=False) as opt_time:

            # if self.early_reduce_samples:
            if self.apply_nms or (not (self.early_reduce_sample_size == 0 or self.early_reduce_sample_size is None)):
                # Observation likelihood measurement
                seg_per_obj = self.get_obj_seg_jit(valid_obj_no, x_pred.outer_shape[0])(x_pred, cond) # (NS, NO, NC, NI, NJ)
                seg_global = jnp.max(seg_per_obj, axis=-4) # (NS, NC, NI, NJ)
                seg_global = jnp.where(valid_obs_mask[None,:,None,None],seg_global,0)
                obs_preds = jnp.sum(seg_obs*seg_global, (-1,-2,-3))/(jnp.sum(jnp.maximum(seg_obs,seg_global), (-1,-2,-3)) + 1e-6)

                if not (self.early_reduce_sample_size == 0 or self.early_reduce_sample_size is None):
                    obs_max_idx = jnp.argsort(-obs_preds, axis=-1)[:self.early_reduce_sample_size]
                    x_pred, conf, seg_per_obj = jax.tree_map(lambda x:x[obs_max_idx], (x_pred, conf, seg_per_obj))
            
            # apply NMS
            if self.apply_nms:
                nobj = x_pred.outer_shape[1]
                nms_iou_threshold = 0.6
                valid_mask = jnp.ones(x_pred.outer_shape, dtype=jnp.bool_)
                for i in range(nobj-1):
                    target_seg = seg_per_obj[:,i:i+1]
                    other_seg = seg_per_obj[:,i+1:]
                    iou = jnp.sum(target_seg * other_seg, axis=(-1,-2,-3))
                    iou = iou/(jnp.sum(target_seg + other_seg, axis=(-1,-2,-3))-iou + 1e-5)
                    valid_mask_current = iou<nms_iou_threshold
                    valid_mask = valid_mask.at[:,i+1:].set(jnp.logical_and(valid_mask[:,i+1:], valid_mask_current))
                x_pred = x_pred.replace(pos=jnp.where(valid_mask[...,None], x_pred.pos, jnp.array([0,0,10.0])))
                conf = jnp.where(valid_mask, conf, -100)
            
            # Optimization steps
            cur_h = x_pred.h
            # optimizer = optax.adamw(self.optim_lr)
            optimizer = optax.adam(self.optim_lr)
            # optimizer = optax.adam(self.optim_lr if previous_obj_pred is None else self.optim_lr/10.)
            opt_state = optimizer.init(cur_h)
            for i in range(self.optimization_step):
                grad, loss_per_obj = self.get_obs_loss_jit(valid_obj_no)(cur_h, cond, seg_obs, valid_obs_mask, plane_params)
                updates, opt_state = optimizer.update(grad, opt_state, cur_h)
                cur_h = optax.apply_updates(cur_h, updates)
                # print(loss_per_obj)
            x_pred = x_pred.set_h(cur_h)
            
            # Observation likelihood measurement
            seg_per_obj = self.get_obj_seg_jit(valid_obj_no, x_pred.outer_shape[0])(x_pred, cond) # (NS, NO, NC, NI, NJ)

            # object filtering by per-object observation
            seg_pixel_threshold = 5
            keep_mask_all_smaples = jnp.sum(seg_per_obj*seg_obs, (-1,-2,-3)) # (NS, NO)
            keep_mask_all_smaples = keep_mask_all_smaples > seg_pixel_threshold * seg_per_obj.shape[-3]
            x_pred:cxutil.LatentObjects = x_pred.replace(pos=jnp.where(keep_mask_all_smaples[...,None], x_pred.pos, jnp.array([0,0,10.0])))
            conf = jnp.where(keep_mask_all_smaples, conf, -100)
            seg_per_obj = jnp.where(keep_mask_all_smaples[...,None,None,None], seg_per_obj, 0)

            # apply NMS
            if self.apply_nms:
                nobj = x_pred.outer_shape[1]
                nms_iou_threshold = 0.6
                valid_mask = jnp.ones(x_pred.outer_shape, dtype=jnp.bool_)
                for i in range(nobj-1):
                    target_seg = seg_per_obj[:,i:i+1]
                    other_seg = seg_per_obj[:,i+1:]
                    iou = jnp.sum(target_seg * other_seg, axis=(-1,-2,-3))
                    iou = iou/(jnp.sum(target_seg + other_seg, axis=(-1,-2,-3))-iou + 1e-5)
                    valid_mask_current = iou<nms_iou_threshold
                    valid_mask = valid_mask.at[:,i+1:].set(jnp.logical_and(valid_mask[:,i+1:], valid_mask_current))
                x_pred = x_pred.replace(pos=jnp.where(valid_mask[...,None], x_pred.pos, jnp.array([0,0,10.0])))
                conf = jnp.where(valid_mask, conf, -100)
                seg_per_obj = jnp.where(valid_mask[...,None,None,None], seg_per_obj, 0)

            seg_global = jnp.max(seg_per_obj, axis=-4) # (NS, NC, NI, NJ)
            seg_global = jnp.where(valid_obs_mask[None,:,None,None],seg_global,0)
            obs_preds = jnp.sum(seg_obs*seg_global, (-1,-2,-3))/(jnp.sum(jnp.maximum(seg_obs,seg_global), (-1,-2,-3)) + 1e-6)

            if hasattr(self.models.dif_args, 'implicit_baseline') and self.models.dif_args.implicit_baseline:
                jkey, subkey = jax.random.split(jkey)
                x_pred = x_pred.register_pcd_from_latent(self.models, self.models.args.npoint, subkey, self.pcd_sample_func)
            
            obs_max_idx = jnp.argsort(-obs_preds, axis=-1)[:self.overay_obj_no]
            obj_pred, conf_select, seg_per_obj_select = jax.tree_map(lambda x:einops.rearrange(x[obs_max_idx], 'i j ... -> (i j) ...'), (x_pred, conf, seg_per_obj))
            if verbose > 1:
                print(f'max_obs after opt: {jnp.max(obs_preds)}')
                print(f'Select objects with highest observation value {obs_preds[obs_max_idx]}')
            
            # # Filter out object in invalid region
            # Compose mask
            # seg_per_obj_select (NO NC NI NJ)
            obs_measurement_per_obj_in_pixel = jnp.sum(seg_per_obj_select*seg_obs[None], (-1,-2,-3)) # (NO, )
            obs_measurement_per_obj_normalized = obs_measurement_per_obj_in_pixel / (jnp.max(obs_measurement_per_obj_in_pixel)+1e-5)
            if out_valid_obj_only:
                keep_mask = obs_measurement_per_obj_in_pixel > seg_pixel_threshold * seg_per_obj_select.shape[1]
                keep_mask = jnp.logical_and(keep_mask, jnp.all(jnp.abs(obj_pred.pos) < self.valid_range, -1))
                # Filtering... Not jit-able!
                valid_obj_idx = jnp.where(keep_mask)
                obj_pred = obj_pred[valid_obj_idx]
                conf_select = conf_select[valid_obj_idx]
                # Logging...
                # assert jnp.all(jnp.abs(obj_pred.pos) < self.valid_range)
                if verbose >= 1:
                    logging.info(f'valid obj no: {obj_pred.outer_shape[0]}')

            # Sort by confidence
            conf_sorted_idx = jnp.argsort(conf_select)[::-1]
            obj_pred_sorted = obj_pred[conf_sorted_idx]
            conf_sorted = conf_select[conf_sorted_idx]
            obs_measurement_per_obj_sorted = obs_measurement_per_obj_normalized[conf_sorted_idx]

            # obj_pred_sorted = obj_pred
            # conf_sorted = conf_select
            # obs_measurement_per_obj_sorted = obs_measurement_per_obj_normalized
            
            # Restore the position from offset.
            obj_pred_sorted = obj_pred_sorted.translate(jnp.array(self.cam_offset))
            obj_pred = obj_pred.translate(jnp.array(self.cam_offset))
            x_pred = x_pred.translate(jnp.array(self.cam_offset))
            obj_pred_sorted = jax.block_until_ready(obj_pred_sorted)

        # Logging
        self.__log_time(est_time.duration, opt_time.duration, verbose)
        if self.save_images:    
            # NOTE(ssh): This is super slow... > 1s
            self.__save_images(
                rgbs_origin, cam_intrinsics_origin, cam_posquats_offset,
                x_pred, conf, obs_preds, seg_obs_original_size, seg_global,
                det_mask, w_dist, verbose)
        if self.save_videos:
            self.__save_videos()
        
        self.unique_id += 1
        aux_inf = self.InferenceAux(x_pred, obs_preds, conf, obs_max_idx, seg_obs_original_size, det_mask, w_dist, seg_per_obj)
        aux_time = (est_time.duration, opt_time.duration)

        return obj_pred_sorted, conf_sorted, aux_inf, aux_time, dif_process_aux_info


    def __log_time(self, est_time: float, opt_time: float, verbose: int):
        """Log inference times"""

        # Log time
        total_time = est_time + opt_time
        self.save_time_intervals['est'].append(est_time)
        self.save_time_intervals['opt'].append(opt_time)
        self.save_time_intervals['total'].append(total_time)
        if verbose > 0:
            print(f'total time: {total_time:.3f} // estimation time: {est_time:.3f} // optimization time: {opt_time:.3f}')
            logging.info(f'total time: {total_time:.3f} // estimation time: {est_time:.3f} // optimization time: {opt_time:.3f}')
        
        # Log averaged time
        if len(self.save_time_intervals['total']) > 1:
            time_summary = [
                f"{k} avg={np.mean(self.save_time_intervals[k][1:]):.3f}, std={np.std(self.save_time_intervals[k][1:]):.3f} "
                for k in ['total', 'est', 'opt']
            ]
            time_summary = f"cummulative time: {time_summary[0]} // {time_summary[1]} // {time_summary[2]}"
            if verbose > 0:
                print(time_summary)
                logging.info(time_summary)


    def save_images2(
            self, 
            save_path: Path,
            rgbs_origin: jnp.ndarray, 
            cam_intrinsics_origin: jnp.ndarray, 
            cam_posquats_origin: jnp.ndarray,
            aux_inf: InferenceAux,
    ):
        """ Recover rendering from previous prediction
        
        NOTE(ssh): Refactor this later..."""
        #   Resize (space to perform seg optimization)
        origin_pixel_size = rgbs_origin.shape[-3:-1]

        #   Re-center OOD absolute position
        cam_posquats_offset = copy.deepcopy(cam_posquats_origin)
        cam_posquats_offset[...,:3] = cam_posquats_offset[...,:3] - self.cam_offset

        # Resize
        origin_pixel_size = rgbs_origin.shape[-3:-1]
        vis_rgbs_gt = self.resize_img_fn_batched(rgbs_origin, self.vis_pixel_size)
        vis_intrinsic_entire = self.resize_intrinsic_fn_batched(
            cam_intrinsics_origin, origin_pixel_size, self.vis_pixel_size)

        # Select top 4 obs score pred
        pick_top_4_idx = jnp.argsort(-aux_inf.obs_preds)[:4]
        x_pred_vis, conf_vis = jax.tree_map(lambda x: x[pick_top_4_idx], (aux_inf.x_pred, aux_inf.conf))

        def rgb_mix(rgb, pred, instance_seg=None):
            alpha = 0.6
            if instance_seg is not None:
                pred_ = np.where(instance_seg[...,None]>=0, pred, 0)
            return (1-alpha)*rgb/255. + pred_*alpha
        
        plt.figure(figsize=(12,10))
        nv = rgbs_origin.shape[1]
        for ri in range(nv):
            vis_intrinsic = vis_intrinsic_entire[0,ri]
            vis_cam_posquat = cam_posquats_offset[0,ri]
            rgb_gt = vis_rgbs_gt[0][ri]
            rgb_debug, instant_seg_pred = jax.vmap(self.render_img, (0,None,None,0))(x_pred_vis, vis_intrinsic, vis_cam_posquat, conf_vis)
            rgb_debug = np.array(rgb_debug)
            instant_seg_pred = np.array(instant_seg_pred)

            plt.subplot(4,nv,nv*0+ri+1)
            plt.imshow(rgb_gt)
            plt.axis('off')
            plt.subplot(4,nv,nv*1+ri+1)
            plt.axis('off')

            loop = range(1) if self.models.dif_args.one_step_test else range(3)
            for i in loop:
                plt.subplot(5,nv,nv*(i+2)+ri+1)
                plt.imshow(rgb_mix(rgb_gt, rgb_debug[i], instant_seg_pred[i]))
                plt.axis('off')
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()


    def __save_images(
            self, 
            rgbs_origin: jnp.ndarray, 
            cam_intrinsics_origin: jnp.ndarray, 
            cam_posquats_offset: jnp.ndarray,
            x_pred: cxutil.LatentObjects,
            conf: jnp.ndarray,
            obs_preds: jnp.ndarray,
            seg_obs_original_size: jnp.ndarray,
            seg_pred_from_obj: jnp.ndarray,
            det_mask: jnp.ndarray,
            w_dist: jnp.ndarray,
            verbose: int,
            override_path: Path = None
    ):
        # Resize
        origin_pixel_size = rgbs_origin.shape[-3:-1]
        vis_rgbs_gt = self.resize_img_fn_batched(rgbs_origin, self.vis_pixel_size)
        vis_intrinsic_entire = self.resize_intrinsic_fn_batched(
            cam_intrinsics_origin, origin_pixel_size, self.vis_pixel_size)
        seg_obs_resized = cutil.resize_img(seg_obs_original_size, self.vis_pixel_size)
        seg_pred_from_obj_resized = cutil.resize_img(seg_pred_from_obj, self.vis_pixel_size)

        # Select top 4 obs score pred
        pick_top_4_idx = jnp.argsort(-obs_preds)[:4]
        x_pred_vis, conf_vis, seg_pred_from_obj_resized_vis = jax.tree_map(lambda x: x[pick_top_4_idx], (x_pred, conf, seg_pred_from_obj_resized))

        def rgb_mix(rgb, pred, instance_seg=None):
            alpha = 0.6
            if instance_seg is not None:
                pred_ = np.where(instance_seg[...,None]>=0, pred, 0)
            return (1-alpha)*rgb/255. + pred_*alpha
        
        plt.figure(figsize=(12,10))
        nv = rgbs_origin.shape[1]
        for ri in range(nv):
            vis_intrinsic = vis_intrinsic_entire[0,ri]
            vis_cam_posquat = cam_posquats_offset[0,ri]
            rgb_gt = vis_rgbs_gt[0][ri]
            rgb_debug, instant_seg_pred = jax.vmap(self.render_img, (0,None,None,0))(x_pred_vis, vis_intrinsic, vis_cam_posquat, conf_vis)
            rgb_debug = np.array(rgb_debug)
            instant_seg_pred = np.array(instant_seg_pred)

            plt.subplot(5,nv,nv*0+ri+1)
            plt.imshow(rgb_gt)
            plt.axis('off')
            plt.subplot(5,nv,nv*1+ri+1)
            plt.imshow(seg_obs_resized[ri])
            plt.axis('off')
            plt.subplot(5,nv,nv*2+ri+1)
            plt.imshow(seg_pred_from_obj_resized_vis[0][ri])
            plt.axis('off')

            loop = range(1) if (self.models.dif_args.one_step_test or self.early_reduce_sample_size==1) else range(2)
            for i in loop:
                plt.subplot(5,nv,nv*(i+3)+ri+1)
                plt.imshow(rgb_mix(rgb_gt, rgb_debug[i], instant_seg_pred[i]))
                plt.axis('off')
                if self.models.dif_args.one_step_test:
                    plt.title(f'{obs_preds[pick_top_4_idx[i]]:0.3f}')
                else:
                    plt.title(f'{obs_preds[pick_top_4_idx[i]]:0.3f}/{det_mask[pick_top_4_idx[i]]}/{w_dist[pick_top_4_idx[i]]}')
        plt.tight_layout()
        if override_path:
            plt.savefig(override_path)
        else:
            plt.savefig(os.path.join(self.logs_dir, f'inf_{self.unique_id}.png'))
        # plt.show()
        plt.close()

        if verbose > 1:
            print('saved images')


    def __save_videos(self):

        raise NotImplementedError()

        view_idx = 0
        x_dif_list_vis = seq[0]
        x_dif_list_vis = jax.tree_map(lambda x: x[obs_max_idx], x_dif_list_vis)
        x_pred_list_vis = seq[1]
        x_pred_list_vis = jax.tree_map(lambda x: x[obs_max_idx], x_pred_list_vis)

        vis_intrinsic = vis_intrinsic_entire[0,view_idx]
        vis_cam_posquat = cam_posquats_offset[0,view_idx]
        rgb_gt = vis_rgbs_gt[0][view_idx]

        print('create videos')
        video_logs_dir = os.path.join(self.logs_dir, 'video')
        os.makedirs(video_logs_dir, exist_ok=True)
        video_arr = []
        with rutil.VideoWriter(os.path.join(video_logs_dir, f'dif_{self.unique_id}.mp4'), fps=15) as vid:
            for t in x_dif_list_vis:
                video_arr.append(rgb_mix(rgb_gt, self.render_img(t, vis_intrinsic, vis_cam_posquat)))
                vid(video_arr[-1])
        video_arr = []
        with rutil.VideoWriter(os.path.join(video_logs_dir, f'pred_{self.unique_id}.mp4'), fps=15) as vid:
            for t in x_pred_list_vis:
                video_arr.append(rgb_mix(rgb_gt, self.render_img(t, vis_intrinsic, vis_cam_posquat)))
                vid(video_arr[-1])


    def summary(self):
        est_time = np.array(self.save_time_intervals['est'])
        opt_time = np.array(self.save_time_intervals['opt'])
        total_time = est_time + opt_time

        # NOTE: exclude jit time
        print(f'avg est: {np.mean(est_time[1:]):.3f} opt: {np.mean(opt_time[1:]):.3f}')
        print(f'std est: {np.std(est_time[1:]):.3f} opt: {np.std(opt_time[1:]):.3f}')
        print(f'total avg: {np.mean(total_time[1:]):.3f}')
        print(f'total std: {np.std(total_time[1:]):.3f}')




class SimpleCollisionChecker(envutil.BaseCollisionChecker):
    """Path continous collision detection(CCD)"""
    def __init__(
            self, 
            models: mutil.Models, 
            panda_objs: cxutil.LatentObjects, 
            env_objs: cxutil.LatentObjects, 
            plane_params: jnp.ndarray,
            robot_base_pos: np.ndarray,
            robot_base_quat: np.ndarray,
            gripper_width: float=0.05,
            camera_body: cxutil.LatentObjects = None,
            robot_attached_obj: cxutil.LatentObjects = None,
            pos_quat_eo:jnp.ndarray=None,
            ):
        """Init params

        Args:
            models (mutil.Models): Collision predictor.
            gripper_width (float): open-0.05 / close-0
        """
        super().__init__(models)
        self.panda_objs = panda_objs
        self.env_objs = env_objs
        self.gripper_width = gripper_width
        self.plane_params = plane_params
        if robot_attached_obj is not None:
            self.pos_quat_eo = pos_quat_eo
            self.robot_attached_obj = robot_attached_obj.replace(pos=jnp.zeros_like(robot_attached_obj.pos))
            self.robot_attached_obj = self.robot_attached_obj.apply_pq_z(*pos_quat_eo, self.models.rot_configs) # pose w.r.t ee-pose (wrist frame)
        else:
            self.robot_attached_obj = None
        self.robot_base_pos = jnp.array(robot_base_pos)
        self.robot_base_quat = jnp.array(robot_base_quat)

        if camera_body is None:
            self.camera_body = cxutil.create_box(np.array([0.025, 0.14, 0.03]), 32, 64)
            self.camera_body = self.camera_body.set_z_with_models(jax.random.PRNGKey(0), models)
        else:
            self.camera_body = camera_body

    def check_q(
            self,
            jkey: jax.Array,
            q: jnp.ndarray,
            visualize=False
    ) -> jnp.ndarray:
        """Check collision at a configuration

        Try using vmap with this function for batched inference.

        Args:
            jkey (jax.Array): Random generator
            q (jnp.ndarray): A configuration to check. (7)

        Returns:
            jnp.ndarray: True when in collision. (0-d array)
        """
        # Shape check
        if q.ndim != 1:
            raise ValueError("Use vmap.")
        
        env_obj_valid_mask = jnp.all(self.env_objs.pos < 8.0, axis=-1)
        
        # Transform grasped object and panda links
        panda_link_obj_tf = fkutil.transform_panda_from_q(q, self.panda_objs, self.models.rot_configs, gripper_width=self.gripper_width)

        # collision check obj
        panda_link_obj_tf:cxutil.LatentObjects = jax.tree_map(lambda x: x[5:], panda_link_obj_tf)
        # panda_link_obj_tf:cxutil.LatentObjects = jax.tree_map(lambda x: x[7:], panda_link_obj_tf)
        franka_link_pq, grasp_center_pq = fkutil.Franka_FK(q, self.gripper_width) # DH Link frame

        # add obj
        if self.robot_attached_obj is not None:
            robot_attached_obj_tf = self.robot_attached_obj.apply_pq_z(*franka_link_pq[-3], self.models.rot_configs)
            panda_link_obj_tf = panda_link_obj_tf.concat(robot_attached_obj_tf, axis=0)

        # add camera
        camera_offset = np.array([0.057373, 0.0, -0.073])
        cam_body_tf = self.camera_body.apply_pq_z(grasp_center_pq[0] + tutil.qaction(grasp_center_pq[1], camera_offset), grasp_center_pq[1], self.models.rot_configs)
        panda_link_obj_tf = jax.tree_map(lambda x,y: jnp.concatenate([x,y[None]], 0), panda_link_obj_tf, cam_body_tf)

        panda_link_obj_tf = panda_link_obj_tf.apply_pq_z(self.robot_base_pos, self.robot_base_quat, self.models.rot_configs)
        
        # if self.robot_attached_obj is not None:
        if visualize:
            import open3d as o3d
            from examples.visualize_occ import create_mesh_from_latent

            franka_mesh_o3d = create_mesh_from_latent(jkey, self.models, panda_link_obj_tf, density=250, qp_bound=0.5, ndiv=800, visualize=False)
            franka_mesh_o3d.compute_vertex_normals()
            obj_mesh_o3d = create_mesh_from_latent(jkey, self.models, self.env_objs, density=250, qp_bound=0.5, ndiv=800, visualize=False)
            obj_mesh_o3d.compute_vertex_normals()

            link_pq = tutil.pq_multi(self.robot_base_pos, self.robot_base_quat, *franka_link_pq[-3])
            # obj_T = np.array(tutil.pq2H(*tutil.pq_multi(*link_pq, *self.pos_quat_eo)))
            mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, origin=np.array([0., 0., 0.]))
            link_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, origin=np.array([0., 0., 0.]))
            link_frame = link_frame.transform(np.array(tutil.pq2H(*link_pq)))
            # obj_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, origin=np.array([0., 0., 0.]))
            # obj_frame = obj_frame.transform(obj_T)
            o3d.visualization.draw_geometries([franka_mesh_o3d, obj_mesh_o3d, mesh_frame, link_frame])

        if isinstance(self.plane_params, tuple) or isinstance(self.plane_params, list):
            consider_open_space = True
            plane_params = jnp.concatenate(self.plane_params, axis=0)
        else:
            consider_open_space = False
            plane_params = self.plane_params
        pln_p = plane_params[:,:3]*plane_params[:,3:]
        pln_n =  plane_params[:,:3]

        plane_col = self.models.apply("pln_predictor", panda_link_obj_tf, pln_p, pln_n)   # (1)
        if consider_open_space:
            plane_col_open_space = plane_col[...,-1]
            plane_col = plane_col[...,:-1]
        # plane_col_cost = jnp.sum(jax.nn.relu(-plane_col+0.5), axis=-1)
        # plane_col_cost = jnp.sum(jax.nn.relu(-plane_col+1.0), axis=-1)
        plane_col_cost = 10*jnp.sum(jax.nn.relu(-plane_col+1.5), axis=-1)
        plane_col = jnp.any(plane_col < -0.2, axis=-1)
        # plane_col = jnp.any(plane_col < 0., axis=-1)
        # plane_col = jnp.any(plane_col < 0.2, axis=-1)

        if consider_open_space:
            plane_col = jnp.where(plane_col_open_space>0, False, plane_col)
            plane_col_cost = jnp.where(plane_col_open_space>0, 0, plane_col_cost)
        
        # Grasped object <-> env objects + panda arm self collision check
        # Positive when collided
        jkey, qobj_key = jax.random.split(jkey)
        pair_a_obj_col = self.models.pairwise_collision_prediction(panda_link_obj_tf, self.env_objs, jkey=qobj_key, train=False)
        pair_a_obj_col = pair_a_obj_col*env_obj_valid_mask[None].astype(jnp.float32)
        
        if visualize:
            print('hold')
        pair_a_obj_col_cost = jnp.sum(jax.nn.relu(pair_a_obj_col+7.0)*env_obj_valid_mask[None].astype(jnp.float32), axis=-1)
        # pair_a_obj_col_cost = jnp.sum(jax.nn.relu(pair_a_obj_col+20.0), axis=-1)
        # pair_a_obj_col = jnp.any(pair_a_obj_col > 1.5, axis=-1)
        pair_a_obj_col = jnp.any(pair_a_obj_col > 1.0, axis=-1)
        # pair_a_obj_col = jnp.any(pair_a_obj_col > 0.0, axis=-1)
        # pair_a_obj_col = jnp.any(pair_a_obj_col > -0.5, axis=-1)
        # pair_a_obj_col = jnp.any(pair_a_obj_col > -1.5, axis=-1)
        # pair_a_obj_col = jnp.any(pair_a_obj_col > -2.0, axis=-1)


        # joint limit
        # # limit cost
        # upper_limit = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])
        # lower_limit = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
        # joint_limit_cost = jax.nn.relu(q - upper_limit + 0.3) + jax.nn.relu(lower_limit - q + 0.3)
        # joint_limit_cost = jnp.sum(joint_limit_cost, axis=-1)
        # joint_limit_col = jnp.logical_or(q > upper_limit - 0.1, q < lower_limit + 0.1)
        # joint_limit_col = jnp.any(joint_limit_col)

        # Aggregate collisions
        cols = jnp.concatenate((plane_col, pair_a_obj_col))
        col_res = jnp.any(cols)
        col_cost = jnp.sum(plane_col_cost + pair_a_obj_col_cost)

        if visualize:
            print('hold!')

        # col_res = jnp.where(grasp_center_pq[0][0] < 0.35, False, col_res)
        # col_cost = jnp.where(grasp_center_pq[0][0] < 0.35, 0, col_cost)
        # cols = pair_a_obj_col
        return col_res, col_cost


def draw_RRT_state(
        q: jnp.ndarray, 
        panda_link_obj,
        env_objs:cxutil.LatentObjects,
        robot_base_pos: jnp.ndarray,
        robot_base_quat: jnp.ndarray,
        models: mutil.Models,
        gripper_width=0.05, # open : 0.05
) -> jnp.ndarray:
    """???"""
    # Robot
    # Transform grasped object and panda links
    panda_link_obj_tf = fkutil.transform_panda_from_q(q, panda_link_obj, models.rot_configs, gripper_width=gripper_width)
    panda_link_obj_tf = panda_link_obj_tf.apply_pq_z(robot_base_pos, robot_base_quat, models.rot_configs)
    # panda_link_obj_tf = panda_link_obj_tf.replace(color=jnp.ones_like(panda_link_obj_tf.pos)*0.9)
    panda_link_obj_tf = panda_link_obj_tf.random_color(jax.random.PRNGKey(0))
    panda_link_obj_tf = panda_link_obj_tf.replace(color=panda_link_obj_tf.color.at[...,:3].set(jnp.ones_like(panda_link_obj_tf.pos)*0.9))
    env_objs = env_objs.random_color(jax.random.PRNGKey(0))
    # Render
    vis_obj = jax.tree_map(lambda *x: jnp.concatenate(x, axis=0), panda_link_obj_tf, env_objs)

    return rutil.cvx_render_scene(vis_obj, models, sdf=partial(rutil.scene_sdf, models=models), 
                                  floor_offset=0.0,
                                  pixel_size=(500, 500),
                                  target_pos=jnp.array([0,0,0.6]), 
                                  camera_pos=jnp.array([1.5,0.7,1.4]))

def Bezier_curve_3points(p1, p2, p3, t):
    '''
    p1, p2, p3 -> (f,)
    t -> (s,)
    '''
    t_extended = t[...,None]
    res = (1-t_extended)**2*p1[None] + 2*(1-t_extended)*t_extended*p2[None] + t_extended**2*p3[None]
    return res

def interval_based_interpolations(waypnts, gap):
    wp_len = jnp.linalg.norm(waypnts[1:] - waypnts[:-1], axis=-1)
    entire_len = jnp.sum(wp_len)
    assert entire_len > gap
    int_no = int(entire_len/gap)+1
    return way_points_to_trajectory(waypnts, int_no, cos_transition=True)



# @partial(jax.jit, static_argnums=[1,2])
def way_points_to_trajectory(waypnts, resolution, smoothing=False, cos_transition=False, window_size=5):
    """???"""
    wp_len = jnp.linalg.norm(waypnts[1:] - waypnts[:-1], axis=-1)
    wp_len = wp_len/jnp.sum(wp_len).clip(1e-5)
    wp_len = jnp.where(wp_len<1e-4, 0, wp_len)
    wp_len = wp_len/jnp.sum(wp_len)
    wp_len_cumsum = jnp.cumsum(wp_len)
    wp_len_cumsum = jnp.concatenate([jnp.array([0]),wp_len_cumsum], 0)
    wp_len_cumsum = wp_len_cumsum.at[-1].set(1.0)
    indicator = jnp.linspace(0, 1, resolution)
    if smoothing or cos_transition:
        indicator = (-jnp.cos(indicator*jnp.pi)+1)/2.
    included_idx = jnp.sum(indicator[...,None] > wp_len_cumsum[1:], axis=-1)
    
    upper_residual = (wp_len_cumsum[included_idx+1] - indicator)/wp_len[included_idx].clip(1e-5)
    upper_residual = upper_residual.clip(0.,1.)
    bottom_residual = 1.-upper_residual
    
    traj = waypnts[included_idx] * upper_residual[...,None] + waypnts[included_idx+1] * bottom_residual[...,None]
    traj = jnp.where(wp_len[included_idx][...,None] < 1e-4, waypnts[included_idx], traj)
    traj = traj.at[0].set(waypnts[0])
    traj = traj.at[-1].set(waypnts[-1])
    
    if smoothing:
        window = jnp.ones(window_size) / window_size
        # window = jnp.array([1,2,5.,2,1])
        window = window/jnp.sum(window)
        traj = jnp.concatenate([waypnts[0:1],waypnts[0:1],waypnts[0:1],waypnts[0:1],waypnts[0:1],waypnts[0:1], traj, waypnts[-1:],waypnts[-1:],waypnts[-1:],waypnts[-1:],waypnts[-1:],waypnts[-1:]], 0)
        traj = jax.vmap(partial(jnp.convolve, mode='same'), (0,None))(traj.swapaxes(0,1), window).swapaxes(0,1)
        traj = traj[6:-6]
        # traj = traj.at[0].set(waypnts[0])
        # traj = traj.at[-1].set(waypnts[-1])

    return traj

def two_trajectory_smoothing(traj1, traj2, window_size=5, output_resolution=100):
    mid_traj = Bezier_curve_3points(traj1[-window_size], traj1[-1], traj2[window_size], jnp.linspace(0,1,int(2.5*window_size),True))
    grasp_traj = way_points_to_trajectory(jnp.concatenate([traj1[:-window_size], mid_traj, traj2[window_size:]], axis=0), resolution=output_resolution)
    return grasp_traj

def traj_gaol_clip(cur_traj, epsilon=0.04):
    # # goal reaching traj
    goal = cur_traj[-1]
    traj_norm = jnp.linalg.norm(goal - cur_traj, axis=-1)
    traj_min_idx = jnp.min(jnp.arange(traj_norm.shape[0], dtype=jnp.int32)*(traj_norm<epsilon).astype(jnp.int32) + 100000*(traj_norm>=epsilon).astype(jnp.int32))
    if traj_min_idx==cur_traj.shape[0]-1:
        return cur_traj
    cur_traj = cur_traj[:traj_min_idx+1]
    cur_traj = jnp.concatenate([cur_traj, goal[None]], axis=0)
    return cur_traj

class FrankaRRT(object):
    def __init__(
            self, 
            models, 
            robot_base_pos,
            robot_base_quat,
            logs_dir,
            nb = 2, 
            node_size = 4000,
            node_step_list = None,
    ):
        # Create latent panda
        build = ioutil.BuildMetadata.from_str("32_64_1_v4")
        panda_dir_path = BASEDIR/"data"/"PANDA"/str(build)
        panda_key = jax.random.PRNGKey(0)
        panda_link_obj = envutil.create_panda(panda_key, panda_dir_path, build, models)
        self.robot_base_pos = robot_base_pos
        self.robot_base_quat = robot_base_quat
        self.logs_dir = logs_dir
        self.models = models
        self.panda_link_obj = panda_link_obj
        self.nb = nb
        self.npd = node_size
        self.unique_id = 0
        if node_step_list is None:
            # self.npd_step_list = [250, self.npd]
            self.npd_step_list = [500, self.npd]
        else:
            self.npd_step_list = node_step_list
        self.time_tracker = TimeTracker()
        logging.info(f'Franka RRT initialized. using nb={nb}, node size={node_size}')

        self.way_points_to_trajectory_nonsmooth = jax.jit(partial(way_points_to_trajectory, smoothing=False), static_argnums=[1])
        self.way_points_to_trajectory_smooth = jax.jit(partial(way_points_to_trajectory, smoothing=True), static_argnums=[1])

    def comfile_jit(self):
        print('start jit compile for motion planning')
        
        camera_body = cxutil.create_box(np.array([0.025, 0.14, 0.03]), 32, 64)
        camera_body = camera_body.set_z_with_models(jax.random.PRNGKey(0), self.models)
        def one_itr_rrt_jit(jkey, points_list, parent_id, cost_list, csq, start_q, goal_q, env_objs, 
                            plane_params, obj_in_hand, pos_quat_eo, gripper_width, nn_max_len):
            cchecker = SimpleCollisionChecker(self.models, self.panda_link_obj.drop_gt_info(color=True), env_objs.drop_gt_info(color=True), 
                                              plane_params, self.robot_base_pos, self.robot_base_quat, gripper_width, camera_body, obj_in_hand, pos_quat_eo)
            return rrtutil.one_itr_rrt(jkey, points_list, parent_id, cost_list, csq, start_q, goal_q,
                # expand_length = 0.35,
                expand_length = 0.10,
                sampler       = partial(fkutil.sample_biased_random_configuration, backward_sample_ratio = 0.15, backward_sample_std=0.05), 
                path_check    = partial(cchecker.check_path, col_res_no=40), 
                star          = True,
                nn_max_len    = nn_max_len,
                )

        
        self.one_itr_rrt_jit_list = {True:{},False:{}}
        def get_rrt_jit_func(obj_no, grasp_obj):
            if obj_no in self.one_itr_rrt_jit_list[grasp_obj]:
                return self.one_itr_rrt_jit_list[grasp_obj][obj_no]
            else:
                print(f'rrt jit compile obj_no={obj_no} / grasp {grasp_obj}')
                self.one_itr_rrt_jit_list[grasp_obj][obj_no] = []
                for nsl in self.npd_step_list:
                    self.one_itr_rrt_jit_list[grasp_obj][obj_no].append(jax.jit(jax.vmap(partial(one_itr_rrt_jit, nn_max_len=nsl), (0,0,0,0,0,None,None,None,None, None, None,None))))
                return self.one_itr_rrt_jit_list[grasp_obj][obj_no]
        self.get_rrt_jit_func = get_rrt_jit_func

        # dummy input for jit compile
        points_list = jnp.zeros((self.npd, 7))                   # ?
        parent_id = -2*jnp.ones((self.npd,), dtype=jnp.int32)    # ?
        cost_list = 1e5*jnp.ones((parent_id.shape[0],))     # ?
        
        jkey = jax.random.PRNGKey(0)

        def col_cost(q_, jkey, env_objs, plane_params, obj_in_hand, pos_quat_eo, gripper_width):
            cchecker = SimpleCollisionChecker(self.models, self.panda_link_obj.drop_gt_info(color=True), 
                                              env_objs.drop_gt_info(color=True), plane_params, self.robot_base_pos, self.robot_base_quat, 
                                              gripper_width, camera_body, obj_in_hand, pos_quat_eo)
            col_res, cost = cchecker.check_q(jkey, q_)
            return jnp.where(col_res, cost, 0), cost
        self.col_cost_func = jax.jit(col_cost)
        grad_func = jax.jit(jax.grad(col_cost, has_aux=True))

        def col_cost_vmap(q_, jkey, env_objs, plane_params, obj_in_hand, pos_quat_eo, gripper_width):
            cchecker = SimpleCollisionChecker(self.models, self.panda_link_obj.drop_gt_info(color=True), 
                                              env_objs.drop_gt_info(color=True), plane_params, self.robot_base_pos, self.robot_base_quat, 
                                              gripper_width, camera_body, obj_in_hand, pos_quat_eo)
            col_res, cost = jax.vmap(cchecker.check_q)(jax.random.split(jkey, q_.shape[0]), q_)
            return jnp.sum(cost), (col_res, cost)
        self.col_cost_vmap_func = jax.jit(col_cost_vmap)
        self.col_cost_vmap_grad = jax.jit(jax.grad(col_cost_vmap, has_aux=True))

        def col_optimization(q_, jkey, env_objs, plane_params, obj_in_hand, pos_quat_eo, gripper_width):
            optimizer = optax.adam(1e-2)
            opt_state = optimizer.init(q_)
            initial_q_res = q_
            for i in range(3):
                grad, (col_res, cost) = self.col_cost_vmap_grad(initial_q_res, jkey, env_objs, plane_params, obj_in_hand, pos_quat_eo, gripper_width)
                _, jkey = jax.random.split(jkey)
                updates, opt_state = optimizer.update(grad, opt_state, initial_q_res)
                initial_q_res = optax.apply_updates(initial_q_res, updates)
                # if i%10==0:
                #     print(i, jnp.mean(cost))
            _, (col_res, cost) = self.col_cost_vmap_func(initial_q_res, jkey, env_objs, plane_params, obj_in_hand, pos_quat_eo, gripper_width)
            return initial_q_res, col_res, cost

        self.col_optimization = col_optimization

        self.col_grad_func_dict = {True:{}, False:{}}
        def get_col_grad_func(obj_no, grasp_obj):
            if obj_no in self.col_grad_func_dict[grasp_obj].keys():
                # print(f'bring registered func {obj_no}')
                return self.col_grad_func_dict[grasp_obj][obj_no]
            else:
                print(f'col grad jit {obj_no} / grasp {grasp_obj}')
                self.col_grad_func_dict[grasp_obj][obj_no] = grad_func
                return grad_func
        self.get_col_grad_func = get_col_grad_func


        def traj_opt_cost(traj, env_objs, plane_params, obj_in_hand, pos_quat_eo, gripper_width, jkey):
            traj = traj.at[:1].set(jax.lax.stop_gradient(traj[:1]))
            traj = traj.at[-1:].set(jax.lax.stop_gradient(traj[-1:]))
            _, col_cost_per_traj = jax.vmap(col_cost, (0,None,None,None,None,None,None))(traj, jkey, env_objs, plane_params, obj_in_hand, pos_quat_eo, gripper_width)
            # col_cost_mean = jnp.mean(col_cost_per_traj.sort()[-traj.shape[0]//10:])
            
            reducer = 2.
            col_cost_per_traj = col_cost_per_traj.at[:5].set(col_cost_per_traj[:5]/reducer)
            col_cost_per_traj = col_cost_per_traj.at[-5:].set(col_cost_per_traj[-5:]/reducer)

            col_cost_mean = jnp.mean(col_cost_per_traj)
            dif = (traj[1:] - traj[:-1])
            # dif_dif = (dif[1:] - dif[:-1])*10
            # dif_dif_dif = (dif_dif[1:] - dif_dif[:-1])*200
            vel_norm = jnp.linalg.norm(dif, axis=-1)
            cost = col_cost_mean + 2*jnp.mean(vel_norm)
            return cost, col_cost_per_traj
        traj_col_func_grad = jax.jit(jax.grad(traj_opt_cost, has_aux=True))

        self.traj_col_grad_func_dict = {True:{}, False:{}}
        def get_traj_col_grad_func(obj_no, grasp_obj):
            if obj_no in self.traj_col_grad_func_dict[grasp_obj]:
                return self.traj_col_grad_func_dict[grasp_obj][obj_no]
            else:
                print(f'traj col grad jit {obj_no} / grasp {grasp_obj}')
                self.traj_col_grad_func_dict[grasp_obj][obj_no] = traj_col_func_grad
                return self.traj_col_grad_func_dict[grasp_obj][obj_no]
        self.get_traj_col_grad_func = get_traj_col_grad_func
        
        print('end jit compile')


    def execution(
            self, 
            jkey, 
            env_objs, 
            initial_q, 
            goal_pq, 
            goal_q, 
            plane_params,
            gripper_width, # 0.05 open / 0.00 close
            refinement = False, 
            pb_robot = None, 
            video_fn = None,
            early_stop=False,
            obj_in_hand:cxutil.LatentObjects=None,
            pos_quat_eo=None,
            compile=False,
            non_selected_obj_pred=None,
            verbose: int = 0
    ):
        self.time_tracker.set('total')

        nobj = env_objs.outer_shape[0]
        # Goal pose generation
        if goal_pq is None:
            assert goal_q is not None

        if non_selected_obj_pred is not None:
            nobj = non_selected_obj_pred.outer_shape[-1]
            if obj_in_hand is not None:
                target_idx_in_non_selected_obj_pred = jnp.argmin(jnp.linalg.norm(non_selected_obj_pred.pos - obj_in_hand.pos, axis=-1), axis=-1) # (NS,)
                non_selected_target_objs = non_selected_obj_pred.take_along_outer_axis(target_idx_in_non_selected_obj_pred, 1) # (NS, 1)
                non_selected_target_objs = non_selected_target_objs.squeeze_outer_shape(1)
                non_selected_env_idx = jnp.where(jnp.arange(nobj-1) >= target_idx_in_non_selected_obj_pred[...,None], jnp.arange(nobj-1)+1, jnp.arange(nobj-1))
                non_selected_env_objs = non_selected_obj_pred.take_along_outer_axis(non_selected_env_idx, 1)
            else:
                non_selected_env_objs = non_selected_obj_pred
            non_selected_env_objs = jax.tree_util.tree_map(lambda x: jnp.reshape(x, (-1,)+x.shape[2:]), non_selected_env_objs)
            env_objs = env_objs.concat(non_selected_env_objs, axis=0)


        goal_pq_rel = fkutil.Franka_FK(goal_q, gripper_width)[0][-3]
        if goal_q is None:
            goal_q, ik_cost = fkutil.Franka_IK_numetrical(initial_q, goal_pq_rel, itr_no=300, output_cost=True, grasp_basis=False)
            print(f'goal ik cost{ik_cost[0]}, {ik_cost[1]}')
        if pb_robot is not None:
            for i in pb_robot.nonfixed_joint_indices_arm:
                pb_robot.bc.resetJointState(pb_robot.uid, i, goal_q[i])

        # initial q collision resolusion
        self.time_tracker.set('initial_optimization')
        optimizer = optax.adam(1e-2)
        opt_state = optimizer.init(initial_q)
        initial_q_res = initial_q
        for i in range(80):
            col_grad_func = self.get_col_grad_func(nobj, obj_in_hand is not None)
            grad, cost = col_grad_func(initial_q_res, jkey, env_objs, plane_params, obj_in_hand, pos_quat_eo, gripper_width)
            _, jkey = jax.random.split(jkey)
            updates, opt_state = optimizer.update(grad, opt_state, initial_q_res)
            initial_q_res = optax.apply_updates(initial_q_res, updates)
            if i%10==0 and verbose >= 2:
                print(f'init collision resolution {i}, {cost}')
            # pb_robot.reset(np.array(initial_q_res))
        self.time_tracker.set('initial_optimization')

        # print('goal collision resolusion')
        # t_io_s = time.time()
        self.time_tracker.set('goal_optimization')
        optimizer = optax.adam(1e-2)
        opt_state = optimizer.init(goal_q)
        goal_q_res = goal_q
        # pb_robot.reset(np.array(initial_q_res))
        for i in range(80):
            col_grad_func = self.get_col_grad_func(nobj, obj_in_hand is not None)
            grad, cost = col_grad_func(initial_q_res, jkey, env_objs, plane_params, obj_in_hand, pos_quat_eo, gripper_width)
            _, jkey = jax.random.split(jkey)
            updates, opt_state = optimizer.update(grad, opt_state, goal_q_res)
            goal_q_res = optax.apply_updates(goal_q_res, updates)
            if i%10==0 and verbose >= 2:
                print(f'goal collision resolution {i}, {cost}')
        self.time_tracker.set('goal_optimization')

        # PRM
        # cchecker = SimpleCollisionChecker(self.models, self.panda_link_obj.drop_gt_info(color=True), env_objs.drop_gt_info(color=True), 
        #                                       plane_params, self.robot_base_posquat, None)

        # trajectory, nodes = PRM.PRM_node_only(jkey, initial_q_res, goal_q_res, 6_000_000, 50,
        #         fkutil.sample_random_configuration,
        #         self.col_cost_vmap_func, self.col_optimization, col_aug=(env_objs, plane_params), nn_dist_limit=0.6, one_batch_size=10000)
        
        # trajectory, nodes = PRM.PRM_node_only(jkey, initial_q_res, goal_q_res, 600_000, 5,
        #         fkutil.sample_random_configuration,
        #         self.col_cost_vmap_func, self.col_optimization, col_aug=(env_objs, plane_params), nn_dist_limit=None, one_batch_size=1000)

        # # RRT
        self.time_tracker.set('rrt')
        points_list = jnp.zeros((self.nb, self.npd, 7), dtype=jnp.float32)                   # ?
        parent_id = -2*jnp.ones((self.nb, self.npd,), dtype=jnp.int32)    # ?
        cost_list = 1e5*jnp.ones((self.nb, self.npd,), dtype=jnp.float32)     # ?
        
        points_list = points_list.at[:,0].set(initial_q_res)
        parent_id = parent_id.at[:,0].set(-1)
        cost_list = cost_list.at[:,0].set(0.0)
        
        # RRT loop
        goal_reached_threshold = 0.05
        csq = jnp.ones((self.nb,), dtype=jnp.int32)
        # print('start RRT')
        step_idx = 0
        jkey_batch = jax.random.split(jkey, self.nb)
        for i in range(self.npd):
            if jnp.max(csq) >= self.npd_step_list[step_idx]:
                # goal reaching test
                cost_to_goal = jnp.linalg.norm(points_list - goal_q_res, axis=-1)
                cost_to_goal = jnp.min(jnp.where(jnp.arange(points_list.shape[-2])<csq[...,None], cost_to_goal, 1e5), -1)
                # Logging...?
                if verbose >= 1:
                    print(f'total itr {i+1} // cur node func: {self.npd_step_list[step_idx]} // dist to goal: {cost_to_goal}')
                    logging.info(f'total itr {i+1} // cur node func: {self.npd_step_list[step_idx]} // dist to goal: {cost_to_goal}')

                if early_stop and jnp.min(cost_to_goal) < goal_reached_threshold:
                    print('goal reached break')
                    break
                step_idx += 1
            if compile and i==0:
                for cp_n, rrt_jit_func in enumerate(self.get_rrt_jit_func(nobj, obj_in_hand is not None)):
                    print(f'compiling {nobj} / {obj_in_hand} / {cp_n}')
                    qpt, jkey_batch, points_list, parent_id, cost_list, csq \
                        = rrt_jit_func(jkey_batch, points_list, parent_id, cost_list, csq, 
                                                                initial_q_res, goal_q_res, env_objs, plane_params, obj_in_hand, pos_quat_eo, gripper_width)
            else:
                qpt, jkey_batch, points_list, parent_id, cost_list, csq \
                    = self.get_rrt_jit_func(nobj, obj_in_hand is not None)[step_idx](jkey_batch, points_list, parent_id, cost_list, csq, 
                                                                initial_q_res, goal_q_res, env_objs, plane_params, obj_in_hand, pos_quat_eo, gripper_width)
        jkey = jkey_batch[0]
        _, jkey = jax.random.split(jkey)

        if verbose >= 1:
            print(f'final node size: {csq}')
            logging.info(f'final node size: {csq}')

        # select best trajectory
        dist_to_goal = jnp.linalg.norm(points_list - goal_q_res, axis=-1)
        nn_idx = jnp.argmin(jnp.where(jnp.arange(points_list.shape[-2])<csq[...,None], dist_to_goal, 1e5), -1)
        dist_to_goal = jnp.take_along_axis(dist_to_goal, nn_idx[...,None], axis=-1).squeeze(-1)
        goal_reached = dist_to_goal < goal_reached_threshold
        cost_batch = jnp.take_along_axis(cost_list, nn_idx[...,None], axis=-1).squeeze(-1)
        batch_idx = jnp.argmin(jnp.where(goal_reached, cost_batch, jnp.inf))
        batch_idx = jnp.where(jnp.any(goal_reached), batch_idx, jnp.argmin(dist_to_goal))
        if verbose >= 1:
            print(f'select idx {batch_idx} with cost {cost_batch} dist to goal {dist_to_goal} and goal reached {goal_reached}')
            logging.info(f'select idx {batch_idx} with cost {cost_batch} dist to goal {dist_to_goal} and goal reached {goal_reached}')

        points_list = points_list[batch_idx]
        parent_id = parent_id[batch_idx]
        dist_to_goal = dist_to_goal[batch_idx]
        goal_reached = goal_reached[batch_idx]
        rrt_cost = cost_batch[batch_idx]
        nn_idx = nn_idx[batch_idx]

        # ????????
        # cost_to_goal, _ = jax.vmap(fkutil.cost_func, (0,None))(points_list, goal_pq_rel)
        # cost_to_goal = jnp.linalg.norm(points_list - goal_q_res, axis=-1)
        # # nn_idx = jnp.argmin(jnp.where(jnp.arange(points_list.shape[0])<csq, cost_to_goal, 1e5))
        # nn_idx = jnp.argmin(jnp.where(jnp.arange(points_list.shape[-2])<csq[...,None], cost_to_goal, 1e5), -1)
        if verbose >= 1:
            logging.info(f'dist to goal: {dist_to_goal} // goal reached: {goal_reached} // cost: {rrt_cost}')
            print(f'dist to goal: {dist_to_goal} // goal reached: {goal_reached} // cost: {rrt_cost}')
        cid = nn_idx
        trajectory = [goal_q]
        cnt = 0
        while parent_id[cid] != -1:
            cur_state = points_list[cid]
            trajectory.append(cur_state)
            if cid == parent_id[cid]:
                trajectory.append(points_list[parent_id[cid]])
                print('duplicated end -> caused by collision in initial state')
                break
            cid = parent_id[cid]
            cnt += 1
            #     invalid_solution=True
        trajectory.append(initial_q_res)
        trajectory.append(initial_q)
        trajectory = list(reversed(trajectory))

        # print('interpolation')
        trajectory = jnp.stack(trajectory)
        trajectory = traj_gaol_clip(trajectory, epsilon=0.04)
        # trajectory_interporated = trajectory
        if refinement:
            trajectory_interporated = self.way_points_to_trajectory_smooth(trajectory, resolution=200)
        else:
            trajectory_interporated = self.way_points_to_trajectory_nonsmooth(trajectory, resolution=200)
        self.time_tracker.set('rrt')
        origin_jraj = trajectory_interporated

        self.time_tracker.set('final_optimization')
        # print('final optimization')
        if refinement:
            optimizer_fn = optax.adam(1e-2)
            opt_state = optimizer_fn.init(trajectory_interporated)
            traj_rf = trajectory_interporated
            for i in range(60):
                grad, cost = self.get_traj_col_grad_func(nobj, obj_in_hand is not None)(traj_rf, env_objs, 
                                                                                        plane_params, obj_in_hand, pos_quat_eo, gripper_width, jkey)
                grad = jax.tree_map(lambda x: jnp.where(jnp.isnan(x), 0, x), grad)
                _, jkey = jax.random.split(jkey)
                updates, opt_state = optimizer_fn.update(grad, opt_state, traj_rf)
                traj_rf = optax.apply_updates(traj_rf, updates)
                traj_rf = self.way_points_to_trajectory_nonsmooth(traj_rf, resolution=traj_rf.shape[0])
                if i%5==0 and verbose>=2:
                    print(f'motion refinement {i}, {jnp.mean(cost)}')
            traj_rf = self.way_points_to_trajectory_smooth(traj_rf, resolution=traj_rf.shape[0])
            trajectory_interporated = traj_rf
        self.time_tracker.set('final_optimization')
        self.time_tracker.set('total')

        if verbose >= 0:
            print(self.time_tracker.str_dt_all())
            logging.info(self.time_tracker.str_dt_all())
        if video_fn is not None:
            # # Write video
            print('create videos')
            with rutil.VideoWriter(filename=str(Path(self.logs_dir)/f"{video_fn}_{self.unique_id}.mp4"), fps=25) as video_writer:
                drawer = partial(draw_RRT_state, panda_link_obj=self.panda_link_obj.drop_gt_info(color=True), env_objs=env_objs, robot_base_pos=self.robot_base_pos, robot_base_quat=self.robot_base_quat, models=self.models)
                drawer = jax.jit(drawer)
                for i, stt in enumerate(trajectory_interporated):
                    rgb = drawer(stt)
                    video_writer(rgb)
        self.unique_id += 1
        return trajectory_interporated, origin_jraj, goal_reached, (points_list, nn_idx)
    
if __name__ == '__main__':
    wpnts = np.array([[1.2,1], [0,1], [0,1+1e-5], [0, 2e-5], [0, 2e-5], [0, 2e-5]])
    res = way_points_to_trajectory(wpnts, resolution=10, smoothing=False)
    res = way_points_to_trajectory(res, resolution=10, smoothing=False)
    res = way_points_to_trajectory(res, resolution=10, smoothing=False)

    print(1)