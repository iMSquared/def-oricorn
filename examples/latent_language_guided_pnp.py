import os, sys
# os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"   
# os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

from pathlib import Path
import glob
import jax.numpy as jnp
import jax
import datetime
import logging
import pickle
from PIL import Image
import torch
import torch.utils.dlpack
import numpy as np
import matplotlib.pyplot as plt
import copy
import yaml
import pybullet as pb
import open3d as o3d
import time
from functools import partial
from scipy.spatial.transform import Rotation as sciR
from typing import List
import einops

BASEDIR = Path(__file__).parent.parent
if BASEDIR not in sys.path:
    sys.path.insert(0, str(BASEDIR))

import util.inference_util as ifutil
import util.camera_util as cutil
import examples.latent_grasp as latent_grasp
import util.io_util as ioutil
from distill_features import api as clip_api
import util.structs as structs
import util.render_util as rutil
import util.transform_util as tutil
import util.cvx_util as cxutil
import util.obb_util as obbutil
import util.franka_util as fkutil
from experiment.exp_utils import RRT_INITIAL_Q, END_Q, TimeTracker, GRASP_MID_Q
from imm_pb_util.bullet_envs.robot import PandaGripper, FrankaPanda, PandaGripperVisual
from imm_pb_util.bullet_envs.manipulation import FrankaManipulation
from examples.visualize_occ import create_mesh_from_latent
from pybullet_utils.bullet_client import BulletClient
from imm_pb_util.bullet_envs.env import SimpleEnv

class LanguageGuidedPnP(object):
    
    def __init__(self, robot_base_pos, robot_base_quat, env_type='table', scene_obj_no=5, logs_dir=None, cam_posquats=None, save_inf_images=False, rrt_node_size=1000, rrt_node_step_list=None):
        self.robot_base_pos_quat = (robot_base_pos, robot_base_quat)

        self.unique_id = 0
        # self.interpolation_gap = 0.002
        self.interpolation_gap = 0.004
        self.place_z_offset = 0.020
        self.scene_obj_no = scene_obj_no
        self.grasp_deeper_depth = 0.010
        
        self.env_type = env_type
        if logs_dir is None:
            now = datetime.datetime.now() # current date and time
            date_time = now.strftime("%m%d%Y-%H%M%S")
            self.logs_dir = os.path.join('logs_pnp', date_time)
        else:
            self.logs_dir = logs_dir

        os.makedirs(self.logs_dir, exist_ok=True)
        if self.env_type == 'shelf':
            # self.plane_params = jnp.array([[0,0,1,0.], [0,0,-1,-0.38], [-1,0,0,-0.21], [0,1,0,-0.415], [0,-1,0,-0.415]]) # real shelf params
            # self.plane_params = (jnp.array([[0,0,1,0.], [0,0,-1,-0.38], [-1,0,0,-0.21], [0,1,0,-0.415], [0,-1,0,-0.415]]), jnp.array([[-1,0,0,0.23]])) # real shelf params
            self.plane_params = (jnp.array([[0,0,1,0.], [-1,0,0,-0.21], [0,1,0,-0.415], [0,-1,0,-0.415]]), jnp.array([[-1,0,0,0.23]])) # no upper part shelf params
        else:
            self.plane_params = jnp.array([[0,0,1,0.]]) # real shelf params
        base_dir = 'checkpoints/dif_model/02262024-180216_final'
        # base_dir = 'logs_dif_final/parq'
        # estimation_batch_size = 16
        estimation_batch_size = 32
        early_reduce_sample_size = 4
        self.valid_range_len = 0.35
        # self.inf_cls = ifutil.InfCls(base_dir, ns=estimation_batch_size, conf_threshold=-0.5, overay_obj_no=1, max_time_steps=15,
        #                         apply_in_range=True, save_images=True, save_videos=False, scene_obj_no=self.scene_obj_no,
        #                         gradient_guidance=True, optimization_step=7, guidance_grad_step=100, shape_optimization=True, 
        #                         valid_range=self.valid_range_len, log_dir=self.logs_dir)
        # self.inf_cls = ifutil.InfCls(base_dir, ns=estimation_batch_size, conf_threshold=-0.5, overay_obj_no=1, max_time_steps=15,
        #                         apply_in_range=True, save_images=True, save_videos=False, scene_obj_no=self.scene_obj_no,
        #                         gradient_guidance=False, optimization_step=500, optim_lr=4e-3, guidance_grad_step=100, shape_optimization=True, 
        #                         valid_range=self.valid_range_len, log_dir=self.logs_dir)
        self.inf_cls = ifutil.InfCls(base_dir, ns=estimation_batch_size, conf_threshold=-0.5, overay_obj_no=1, max_time_steps=15,
                                apply_in_range=True, save_images=save_inf_images, save_videos=False, scene_obj_no=self.scene_obj_no,
                                gradient_guidance=False, optimization_step=200, optim_lr=4e-3, guidance_grad_step=5, 
                                early_reduce_sample_size=early_reduce_sample_size, shape_optimization=True, 
                                valid_range=self.valid_range_len, log_dir=self.logs_dir)
        # self.inf_cls = ifutil.InfCls(base_dir, ns=estimation_batch_size, conf_threshold=-0.5, overay_obj_no=1, max_time_steps=50,
        #                         apply_in_range=True, save_images=True, save_videos=False, scene_obj_no=self.scene_obj_no,
        #                         gradient_guidance=False, optimization_step=7, guidance_grad_step=10, shape_optimization=False, 
        #                         valid_range=self.valid_range_len, log_dir=self.logs_dir)
        self.models = self.inf_cls.models

        vis_img_size = (240, 424)
        self.inf_cls.init_renderer(vis_img_size, sdf_ratio=2200)

        self.franka_rrt = ifutil.FrankaRRT(self.models, robot_base_pos=robot_base_pos, 
                                           robot_base_quat=robot_base_quat, logs_dir=self.logs_dir, nb=1, node_size=rrt_node_size, node_step_list=rrt_node_step_list)
        self.franka_rrt.comfile_jit()

        self.clip_model, self.clip_preprocess = clip_api.get_clip_model()

        self.obj_segmentation_jit = jax.jit(partial(rutil.obj_segmentation, self.inf_cls.models, self.inf_cls.models.pixel_size, output_depth=True))

        grasp_prediction_lambda = lambda : partial(latent_grasp.grasp_pose_from_latent_considering_collision, self.models, 
                                                   n_grasps_parall_computation=800, n_grasps_output=500, n_grasps_reachability_check=200,
                                                   plane_params=self.plane_params, env_type=self.env_type, deeper_depth=self.grasp_deeper_depth, vis=False)

        grasp_prediction_side_lambda = lambda : partial(latent_grasp.grasp_pose_from_latent_considering_collision, self.models, 
                                                   n_grasps_parall_computation=800, n_grasps_output=500, n_grasps_reachability_check=200,
                                                   plane_params=self.plane_params, env_type='side', deeper_depth=self.grasp_deeper_depth, vis=False)

        self.grasp_prediction_side_func = jax.jit(grasp_prediction_side_lambda())

        self.grasp_prediction_approach_func = jax.jit(partial(latent_grasp.grasp_pose_from_latent_considering_collision, self.models, 
                                                   n_grasps_parall_computation=800, n_grasps_output=500, n_grasps_reachability_check=200,
                                                   plane_params=self.plane_params, env_type='approach', deeper_depth=self.grasp_deeper_depth, vis=False))

        self.grasp_prediction_func = grasp_prediction_lambda()
        self.grasp_prediction_dict = {}
        def get_grasp_prediction_jit(obj_no):
            if obj_no not in self.grasp_prediction_dict:
                print(f'compile grasp prediction jit with obj no: {obj_no}')
                self.grasp_prediction_dict[obj_no] = jax.jit(grasp_prediction_lambda())
            return self.grasp_prediction_dict[obj_no]
        self.get_grasp_prediction_jit = get_grasp_prediction_jit

        self.get_obb_func_dict = {}
        def get_obb_func_jit(obj_no):
            if obj_no not in self.get_obb_func_dict:
                print(f'compile obj obb jit with obj no: {obj_no}')
                self.get_obb_func_dict[obj_no] = jax.jit(jax.vmap(partial(obbutil.get_obb_from_lying_latent_object_2, self.models)))
            return self.get_obb_func_dict[obj_no]
        self.get_obb_func_jit = get_obb_func_jit

        self.get_ik_dict = {}
        def get_ik_func_jit(sample_no):
            if sample_no not in self.get_ik_dict:
                print(f'compile ik jit with sample no: {sample_no}')
                self.get_ik_dict[sample_no] = jax.jit(partial(fkutil.Franka_IK_numetrical, robot_base_pos=self.robot_base_pos_quat[0], 
                                robot_base_quat=self.robot_base_pos_quat[1], itr_no=300, output_cost=False, grasp_basis=False, compensate_last_joint=True))
            return self.get_ik_dict[sample_no]
        self.get_ik_func_jit = get_ik_func_jit

        self.get_col_func_dict = {}
        def get_col_func_jit(obj_no):
            if obj_no not in self.get_col_func_dict:
                print(f'compile col func jit with sample no: {obj_no}')
                self.get_col_func_dict[obj_no] = jax.jit(self.evaluate_place_collision)
            return self.get_col_func_dict[obj_no]
        self.get_col_func_jit = get_col_func_jit

        if cam_posquats is not None:
            # assume that first camera is on the robot
            camera_body = cxutil.create_box(np.array([0.025, 0.14, 0.03]), 32, 64)
            camera_body = camera_body.set_z_with_models(jax.random.PRNGKey(0), self.models)
            self.camera_bodies = camera_body.stack(camera_body, axis=0)
            self.camera_bodies = self.camera_bodies.apply_pq_z(cam_posquats[1:3,:3], cam_posquats[1:3,3:], self.models.rot_configs)

        self.time_tracker = TimeTracker()
        logging.basicConfig(filename=os.path.join(self.logs_dir,'new_logs.log'), encoding='utf-8', level=logging.INFO)
    
    def clip_overlay(self, obj_pred, color, cam_posquat, cam_intrinsic, visualize=False):
        obj_no = obj_pred.outer_shape[0]

        # Segmentation
        original_pixel_size = color.shape[-3:-1]
        model_pixel_size = self.inf_cls.models.pixel_size
        cond = structs.ImgFeatures(cutil.resize_intrinsic(cam_intrinsic, original_pixel_size, model_pixel_size), cam_posquat, None)
        segs, depth = self.obj_segmentation_jit(obj_pred, cond)
        segs = (segs > 0.5) # (NO NC NI NJ)

        # considering instances
        depth = jnp.where(segs, depth, 10.0)
        instance_segs = jnp.where(jnp.any(segs, axis=0), jnp.argmin(depth, axis=0), -1) # (NC NI NJ)

        valid_rgb_mask = np.sum(color[...,:10,:10,:], axis=(-1,-2,-3)) != 0
        instance_segs_resized = cutil.resize_img(instance_segs, original_pixel_size, method='nearest')
        instance_segs_resized = jnp.where(valid_rgb_mask[...,None,None,None], instance_segs_resized, -1) # (NC NI NJ)
        
        # Clip process
        # text_query = "a white botle"
        color_pil_list = [Image.fromarray(rgb) for rgb in color]

        # Get patched CLIP feature
        with torch.no_grad():
            # Let's normalize after aggregation...
            clip_embs = clip_api.extract_clip_features(self.clip_model, self.clip_preprocess, color_pil_list, device="cuda:0", normalize=False)
            # Upsampling
            clip_embs = clip_embs.permute(0, 3, 1, 2)
            clip_embs = torch.nn.functional.interpolate(clip_embs, original_pixel_size, mode="nearest")
            clip_embs = clip_embs.permute(0, 2, 3, 1)

        torch.cuda.synchronize()
        clip_embs = jnp.array(clip_embs.cpu().numpy()) # (NC NI NJ NF)
        clip_embs = clip_embs.astype(jnp.float32) # (NC NI NJ NF)
        per_obj_mask = (jnp.arange(obj_no) == instance_segs_resized)[...,None] # (NC NI NJ NO)

        per_obj_embs_pixel = jnp.where(per_obj_mask, clip_embs[...,None,:], 0) # (NC NI NJ NO NF)
        language_aligned_obj_feat = jnp.sum(per_obj_embs_pixel, axis=(0,1,2))/(jnp.sum(per_obj_mask, axis=(0,1,2)) + 1e-5) # (NO NF)
        language_aligned_obj_feat /= (jnp.linalg.norm(language_aligned_obj_feat, axis=-1, keepdims=True)+1e-5) # (NO NF)
        language_aligned_obj_feat = jax.block_until_ready(language_aligned_obj_feat)
        masked_clip_embs = None

        # clip_embs: np.ndarray = clip_embs.cpu().numpy() # (NC NI NJ NF)
        # clip_embs = clip_embs.astype(np.float32)

        # # CLIP -> object
        # masked_clip_embs = []
        # language_aligned_obj_feat = []
        # for obj_idx in np.arange(obj_no):
        #     mask = np.broadcast_to((instance_segs_resized == obj_idx), shape=clip_embs.shape)
        #     obj_emb = clip_embs[mask].reshape(-1, 768)
        #     # if obj_emb.shape[0] == 0:
        #     #     raise ValueError("Size==0")
        #     mean = obj_emb.mean(axis=0)
        #     if np.any(np.isnan(mean)):
        #         mean = np.zeros(shape=(768,))
        #         print("nan mean detected")
        #     masked_clip_embs.append(mask)
        #     language_aligned_obj_feat.append(mean)
        # masked_clip_embs = np.array(masked_clip_embs)
        # language_aligned_obj_feat = np.array(language_aligned_obj_feat)
        # language_aligned_obj_feat /= (np.linalg.norm(language_aligned_obj_feat, axis=-1, keepdims=True)+1e-5)

        clip_aux_info = (clip_embs, masked_clip_embs, instance_segs_resized, color)

        if visualize:
            def apply_pca(x_, axes, resize_devider=2):
                x = copy.deepcopy(np.where(np.isnan(x_), 0, x_)).astype(np.float32)
                x = x-np.mean(x, axis=axes, keepdims=True)
                x = x/np.std(x, axis=axes, keepdims=True)
                if x.ndim >= 3:
                    # images
                    xz = cutil.resize_img(x, (x.shape[-3]//resize_devider, x.shape[-2]//resize_devider), method='nearest', outer_extend=False).squeeze(0)
                    z = np.sum(xz[...,None,:]*xz[...,None], axis=(-3,-4))
                else:
                    xz = x
                    z = np.sum(xz[...,None,:]*xz[...,None], axis=axes)
                eigenvalues, eigenvectors = np.linalg.eig(z)
                pricipal_axis = eigenvectors.real[:,:3]
                return np.einsum('...i,ij', x, pricipal_axis)
            
            clip_embs_pca = [apply_pca(clip_embs[0], (-2,-3)), apply_pca(clip_embs[1], (-2,-3)), apply_pca(clip_embs[2], (-2,-3))]
            obj_clip_feats_pca = apply_pca(language_aligned_obj_feat, 0)
            clip_emb_images = np.where(instance_segs_resized==-1, 0, obj_clip_feats_pca[instance_segs_resized.squeeze(-1)])
            plt.figure()
            # cmap = plt.get_cmap("turbo")
            for view_idx, rgb in enumerate(color):
                plt.subplot(4, len(color), view_idx + 1)
                plt.imshow(rgb)
                plt.axis("off")

                plt.subplot(4, len(color), view_idx + 1 + len(color))
                plt.imshow(instance_segs_resized[view_idx])
                plt.axis("off")

                plt.subplot(4, len(color), view_idx + 1 + 2*len(color))
                plt.imshow(clip_embs_pca[view_idx])
                plt.axis("off")

                plt.subplot(4, len(color), view_idx + 1 + 3*len(color))
                plt.imshow(clip_emb_images[view_idx])
                plt.axis("off")

            plt.tight_layout()
            plt.show()

        return language_aligned_obj_feat, clip_aux_info
    

    def clip_overlay_bbox(
            self, 
            obj_pred: cxutil.LatentObjects, 
            color: jnp.ndarray, 
            cam_posquat: jnp.ndarray, 
            cam_intrinsic: jnp.ndarray, 
            visualize: bool = True
    ):
        view_no = color.shape[0]
        obj_no = obj_pred.outer_shape[0]
        
        # Segmentation
        original_pixel_size = color.shape[-3:-1]
        model_pixel_size = self.inf_cls.models.pixel_size
        cond = structs.ImgFeatures(cutil.resize_intrinsic(cam_intrinsic, original_pixel_size, model_pixel_size), cam_posquat, None)
        segs, depth = self.obj_segmentation_jit(obj_pred, cond)
        segs = (segs > 0.5) # (NO NC NI NJ)

        # Considering instances
        depth = jnp.where(segs, depth, 10.0)
        instance_segs = jnp.where(jnp.any(segs, axis=0), jnp.argmin(depth, axis=0), -1) # (NC NI NJ)
        instance_segs_resized = cutil.resize_img(instance_segs, original_pixel_size, method='nearest')[..., 0]
        instance_ids_in_segs = jnp.unique(instance_segs_resized)
        instance_ids_in_segs = [val for val in instance_ids_in_segs if val >= 0]

        # Per instance mask
        masks = jnp.array([jnp.equal(instance_segs_resized, val) for val in instance_ids_in_segs])
        # Define bounding box
        def get_bounding_box(mask):
            rows = np.any(mask, axis=1)
            cols = np.any(mask, axis=0)
            rows_nonzero = np.where(rows)[0]
            cols_nonzero = np.where(cols)[0]
            if len(rows_nonzero) == 0 or len(cols_nonzero) == 0:
                return None
            y_min, y_max = rows_nonzero[[0, -1]]
            x_min, x_max = cols_nonzero[[0, -1]]
            return np.array((x_min, y_min, x_max, y_max))
        # Increase bounding box size
        def adjust_bbox(image, bbox, scale=0.2):
            if bbox is None:
                return None
            x_min, y_min, x_max, y_max = bbox
            width = x_max - x_min
            height = y_max - y_min
            # Calculate the increase
            increase_width = width * scale
            increase_height = height * scale
            # Adjust the bounding box
            x_min = max(0, x_min - increase_width / 2)
            y_min = max(0, y_min - increase_height / 2)
            x_max = min(image.shape[1], x_max + increase_width / 2)
            y_max = min(image.shape[0], y_max + increase_height / 2)
            # Ensure the bounding box is within the image boundaries
            x_min, y_min, x_max, y_max = map(int, [x_min, y_min, x_max, y_max])
            return (x_min, y_min, x_max, y_max)
        bboxes = [ [adjust_bbox(vm, get_bounding_box(vm)) for vm in view_masks] for view_masks in masks ]    # (NO, NC)

        # Crop RoI
        def crop_bbox_from_image(image, bbox):
            if bbox is None:
                return None
            x_min, y_min, x_max, y_max = bbox
            # Crop the image
            cropped_image = image[y_min:y_max, x_min:x_max]
            return cropped_image
        
        language_aligned_obj_feat = []
        for idx, instance_id in enumerate(instance_ids_in_segs):
            obj_bboxes = bboxes[idx]
            obj_clip_emb_list = []
            for view_idx, bbox in enumerate(obj_bboxes):
                patch = crop_bbox_from_image(color[view_idx], bbox)
                patch = Image.fromarray(patch)
                clip_emb = clip_api.extract_clip_features(self.clip_model, self.clip_preprocess, [patch], device="cuda:0", normalize=False, patch_output=False)
                obj_clip_emb_list.append(clip_emb.cpu().numpy())
            obj_clip_emb = np.squeeze(np.mean(obj_clip_emb_list, axis=0), axis=0)
            language_aligned_obj_feat.append(obj_clip_emb)
        language_aligned_obj_feat = np.array(language_aligned_obj_feat)
        # Normalize
        language_aligned_obj_feat /= np.linalg.norm(language_aligned_obj_feat, axis=-1, keepdims=True) # [#obj, 768]


        # if visualize:
        #     def draw_bounding_box(image, bbox, c=(255, 0, 0), thickness=2):
        #         x_min, y_min, x_max, y_max = bbox
        #         # Draw top and bottom lines
        #         image[y_min:y_min+thickness, x_min:x_max] = c
        #         image[y_max:y_max+thickness, x_min:x_max] = c
        #         # Draw left and right lines
        #         image[y_min:y_max, x_min:x_min+thickness] = c
        #         image[y_min:y_max, x_max:x_max+thickness] = c
        #         return image
            
        #     vis_color = np.copy(color)
        #     fig, axes = plt.subplots(1, 3)
        #     for i in range(view_no):
        #         view_bboxes = [bboxes[no][i] for no in range(obj_no)]
        #         for bb in view_bboxes:
        #             draw_bounding_box(vis_color[i], bb)
        #         axes[i].imshow(vis_color[i])
        #     fig.show()

        return language_aligned_obj_feat, None




    
    def category_overlay(self, geom_obj_feat, language_aligned_obj_feat):

        # category_list = ['cup', 'bottle', 'bowl', 'mug', 'dish', 'other']
        category_list = ['cup', 'bowl', 'bottle', 'can', 'other']
        category_text_embs = clip_api.get_text_embeddings(self.clip_model, category_list, "cuda:0", normalize=True)
        category_text_embs = jnp.array(category_text_embs)
        category_logit = jnp.einsum('if,jf->ij', language_aligned_obj_feat, category_text_embs)
        category_per_obj = np.array(category_list)[np.argmax(category_logit, axis=-1)]

        valid_obj = np.all(np.abs(geom_obj_feat.pos)<2.0, axis=-1)
        
        category_per_obj = category_per_obj[valid_obj]

        return category_per_obj



    def text_query(self, language_aligned_obj_feat, text_query, visualize=False, clip_aux_info=None, valid_target_mask=None):
        # Query text
        text_embs = clip_api.get_text_embeddings(self.clip_model, [text_query], "cuda:0", normalize=True)
        text_embs = jnp.array(text_embs.cpu().numpy())
        text_embs = jax.block_until_ready(text_embs)

        # Compute similarities
        sims = language_aligned_obj_feat @ text_embs.T
        sims = sims.squeeze(axis=-1)
        sims = jnp.where(jnp.isnan(sims), -jnp.inf, sims)
        if valid_target_mask is not None:
            sims = jnp.where(valid_target_mask, sims, -jnp.inf)


        target_idx = jnp.argmax(sims)

        if visualize:
            clip_embs, masked_clip_embs, instance_segs_resized, colors = clip_aux_info
            # Debug: Visualize similarity after segmentation
            # per pixel similarity
            sims_per_pixel = np.sum(clip_embs * text_embs, axis=-1) # (NC NI NJ)
            sims_per_pixel = sims_per_pixel.astype(np.float32)

            seg_sims_images = np.where(instance_segs_resized==-1, 0, sims[instance_segs_resized]).astype(np.float32)

            plt.figure()
            # cmap = plt.get_cmap("turbo")
            # plt.imshow(seg_sims_images[0])
            for view_idx, rgb in enumerate(colors):
                plt.subplot(4, len(colors), view_idx + 1)
                plt.imshow(rgb)
                plt.axis("off")

                plt.subplot(4, len(colors), view_idx + 1 + len(colors))
                plt.imshow(instance_segs_resized[view_idx])
                plt.axis("off")

                plt.subplot(4, len(colors), view_idx + 1 + 2*len(colors))
                plt.imshow(sims_per_pixel[view_idx])
                plt.axis("off")

                plt.subplot(4, len(colors), view_idx + 1 + 3*len(colors))
                plt.imshow(seg_sims_images[view_idx])
                plt.axis("off")

            plt.tight_layout()
            plt.suptitle(f'Similarity to language query "{text_query}"')
            plt.savefig(os.path.join(self.logs_dir, f'{text_query}_{self.unique_id}.png'))
            # plt.show()

        text_query_aux_info = (text_embs, sims)

        return target_idx, text_query_aux_info

    def estimation(self, images, camera_posquat, intrinsic, jkey):

        # estimation
        # try:
        #     with open(os.path.join(self.logs_dir,f'inf_outputs_{self.unique_id}.pkl'), 'rb') as f:
        #         inf_outputs = pickle.load(f)
        #     print('load estimation results from saved data')
        # except:
        print('start estimation')
        if isinstance(self.plane_params, tuple) or isinstance(self.plane_params, list):
            estimation_plane_params = self.plane_params[0]
        else:
            estimation_plane_params = self.plane_params
        inf_outputs = self.inf_cls.estimation(jkey, self.models.pixel_size, images, intrinsic, camera_posquat, 
                                                plane_params=estimation_plane_params[:1],
                                                out_valid_obj_only=False, 
                                                filter_before_opt=True, verbose=2)

        with open(os.path.join(self.logs_dir,f'inf_outputs_{self.unique_id}.pkl'), 'wb') as f:
            pickle.dump(inf_outputs, f)
        print('apply inferences')

        with open(os.path.join(self.logs_dir, f'observations_{self.unique_id}.pkl'), 'wb') as f:
            pickle.dump((images, camera_posquat, intrinsic), f)

        self.unique_id += 1

        obj_pred_select = inf_outputs[0]
        conf = inf_outputs[1]
        inf_aux_info = inf_outputs[-3]
        
        # clip overay
        self.time_tracker.set('clip overay')
        language_aligned_obj_feat, clip_aux_info = self.clip_overlay(obj_pred_select, images, camera_posquat, intrinsic, visualize=False)
        self.time_tracker.set('clip overay')
        dt = self.time_tracker.dt['clip overay']
        print(f'finish clip overay time: {dt}')

        return obj_pred_select, language_aligned_obj_feat, clip_aux_info, inf_aux_info


    def grasp_motion(self, geom_objrep, target_idx, start_q, jkey, inf_aux_info=None):
        
        # print('start text query')
        # self.time_tracker.set('text query')
        # target_idx = self.text_query(lan_aligned_objrep, target_obj_text, visualize=True, clip_aux_info=clip_aux_info)
        target_obj = jax.tree_util.tree_map(lambda x: x[target_idx:target_idx+1], geom_objrep) # predefine selections
        env_obj = jax.tree_util.tree_map(lambda x: x[:target_idx], geom_objrep) # predefine selections
        env_obj = env_obj.concat(jax.tree_util.tree_map(lambda x: x[target_idx+1:], geom_objrep), axis=0)
        env_obj = jax.block_until_ready(env_obj)
        # self.time_tracker.set('text query')
        # dt = self.time_tracker.dt['text query']
        # print(f'finish text query / time {dt}')

        print('start grasp prediction')
        self.time_tracker.set('grasp prediction')

        if inf_aux_info is not None:
            idx_mask = jnp.arange(inf_aux_info.x_pred.outer_shape[0]) != inf_aux_info.obs_max_idx
            idx_where = jnp.where(idx_mask)
            non_selected_obj_pred = jax.tree_util.tree_map(lambda x: x[idx_where], inf_aux_info.x_pred)
        else:
            non_selected_obj_pred = None

        approach_len=0.070
        cam_offset = jnp.array([0,0,-approach_len], dtype=jnp.float32)
        grasp_q, grasp_approaching_q, grasp_pq, grasp_approach_pq, grasp_aux_info = \
            self.get_grasp_prediction_jit(env_obj.outer_shape[0])(target_obj, *self.robot_base_pos_quat, start_q, jkey, env_obj, cam_offset, non_selected_obj_pred=non_selected_obj_pred)
        mask_info_select, best_joint_pos, approaching_joint_pos, best_pq, approaching_pq, max_idx, mask_info = grasp_aux_info
        _, jkey = jax.random.split(jkey)
        self.time_tracker.set('grasp prediction')
        dt = self.time_tracker.dt['grasp prediction']
        print(f'finish grasp prediction / time: {dt} / mask: {mask_info_select} / num reachable {np.sum(mask_info[1])} / num non-col {np.sum(mask_info[2])}')
        pq_oe = tutil.pq_multi(*tutil.pq_inv(target_obj.pos, jnp.array([0,0,0,1.])), *grasp_pq)

        # approach motion
        traj_rest_to_grasp = ifutil.way_points_to_trajectory(jnp.stack([start_q, grasp_approaching_q, grasp_q], axis=0), 100, smoothing=False)

        return traj_rest_to_grasp, pq_oe, grasp_q, grasp_pq, grasp_aux_info

    def place_to_target(self, target_idx, geom_objrep, lan_aligned_objrep, goal_obj_text, goal_obj_text2, goal_relation_type, pq_oe, grasp_q, rest_q, jkey, intermediate_traj, 
                        clip_aux_info=None, inf_aux_info=None, placement_collision=False, category_per_obj=None):

        print('start text query')
        self.time_tracker.set('text query')
        target_obj = jax.tree_util.tree_map(lambda x: x[target_idx:target_idx+1], geom_objrep) # predefine selections
        if goal_obj_text is not None:
            try:
                size_order = int(goal_obj_text[:1])
                goal_cat = goal_obj_text[1:]

                # get obb from objects
                self.time_tracker.set('obb')
                obbs = self.get_obb_func_jit(geom_objrep.outer_shape[0])(geom_objrep)
                obbs = jax.block_until_ready(obbs)
                self.time_tracker.set('obb')
                dt = self.time_tracker.dt['obb']
                print(f'obb cal dt: {dt}')
                
                # reorder by size
                obj_in_cat_indices = np.where(category_per_obj==goal_cat)[0]
                print(f'target object number: {len(obj_in_cat_indices)}')
                geom_objrep_cat, lan_aligned_objrep_cat, obbs = jax.tree_util.tree_map(lambda x: x[obj_in_cat_indices], (geom_objrep, lan_aligned_objrep, obbs))
                representative_lengths = jnp.max(obbs[-1], axis=-1)
                obj_idx_sorted = jnp.argsort(representative_lengths)
                goal_idx = obj_in_cat_indices[obj_idx_sorted[size_order]]
                goal_text_query_aux_info = None
            except:
                goal_idx, goal_text_query_aux_info = self.text_query(lan_aligned_objrep, goal_obj_text, visualize=False, clip_aux_info=clip_aux_info)
            goal_obj = jax.tree_util.tree_map(lambda x: x[goal_idx:goal_idx+1], geom_objrep) # predefine selections
        else:
            goal_text_query_aux_info = None
        if goal_obj_text2 is not None:
            goal_idx2, goal2_text_query_aux_info = self.text_query(lan_aligned_objrep, goal_obj_text2, visualize=False, clip_aux_info=clip_aux_info)
            goal_obj2 = jax.tree_util.tree_map(lambda x: x[goal_idx2:goal_idx2+1], geom_objrep) # predefine selections
        env_obj = jax.tree_util.tree_map(lambda x: x[:target_idx], geom_objrep) # predefine selections
        env_obj = env_obj.concat(jax.tree_util.tree_map(lambda x: x[target_idx+1:], geom_objrep), axis=0)
        # env_obj = env_obj.concat(self.camera_bodies, axis=0)
        env_obj = jax.block_until_ready(env_obj)
        self.time_tracker.set('text query')
        dt = self.time_tracker.dt['text query']
        print(f'finish text query / time {dt}')

        if inf_aux_info is not None:
            idx_mask = jnp.arange(inf_aux_info.x_pred.outer_shape[0]) != inf_aux_info.obs_max_idx
            idx_where = jnp.where(idx_mask)
            non_selected_obj_pred = jax.tree_util.tree_map(lambda x: x[idx_where], inf_aux_info.x_pred)
        else:
            non_selected_obj_pred = None

        # calculate place pose
        self.time_tracker.set('calculate place pose')
        if goal_obj_text is None or goal_relation_type is None:
            if self.env_type == 'table':
                # place_pos_candidates = jnp.array([0.4,0.4,0.2])
                place_pos_candidates = jnp.array([0.35,-0.35,0.2])
            elif self.env_type == 'shelf':
                # place_pos_candidates = jnp.array([-0.320,0.2,0.1])
                # place_pos_candidates = self.robot_base_pos_quat[0] + jnp.array([0.0,0.4,0.1])
                # place_pos_candidates = self.robot_base_pos_quat[0] + jnp.array([0.5,0.0,0.1])
                place_pos_candidates = self.robot_base_pos_quat[0] + jnp.array([0.22,0.32,0.5])
                # place_pos_candidates = self.robot_base_pos_quat[0] + jnp.array([0.0,0.4,0.1])
                # place_pos_candidates = self.robot_base_pos_quat[0] + jnp.array([0.5,0.0,0.97])
            thetas = jnp.linspace(0, np.pi, 8, endpoint=False)
            place_quat_candidates = tutil.aa2q(thetas[...,None]*jnp.array([0,0,1.]))
            place_pos_candidates += place_quat_candidates[...,:1]*0 # broadcasting
        
        # place pose candidates
        elif goal_relation_type in ['up', 'inside']:
            place_pos_candidates = goal_obj.pos + jnp.array([[0,0,0.06]], dtype=jnp.float32)
            thetas = jnp.linspace(0, np.pi, 8, endpoint=False)
            place_quat_candidates = tutil.aa2q(thetas[...,None]*jnp.array([0,0,1.]))
            place_pos_candidates += place_quat_candidates[...,:1]*0 # broadcasting
        elif goal_relation_type in ['left', 'right', 'front', 'behind']:
            place_pos_candidates = jnp.linspace(0.15, 0.25, 4, endpoint=True, dtype=jnp.float32)
            if goal_relation_type == 'left':
                base_axis = jnp.array([0,1.,0], dtype=jnp.float32)
            elif goal_relation_type == 'right':
                base_axis = jnp.array([0,-1.,0], dtype=jnp.float32)
            elif goal_relation_type == 'front':
                base_axis = jnp.array([-1.0,0,0], dtype=jnp.float32)
            elif goal_relation_type == 'behind':
                base_axis = jnp.array([1.0,0,0], dtype=jnp.float32)
            place_pos_candidates = goal_obj.pos + place_pos_candidates[...,None]*base_axis
            thetas = jnp.linspace(0, np.pi, 2, endpoint=False, dtype=jnp.float32)
            place_quat_candidates = tutil.aa2q(thetas[...,None]*jnp.array([0,0,1.]))
            place_quat_candidates = einops.repeat(place_quat_candidates, '... i j -> ... (r i) j', r=place_pos_candidates.shape[-2])
            place_pos_candidates = einops.repeat(place_pos_candidates, '... i j -> ... (i r) j', r=thetas.shape[-1])
            place_pos_candidates = place_pos_candidates.at[...,-1].set(target_obj.pos[...,-1])
        elif goal_relation_type in ['between']:
            place_pos_candidates = jnp.linspace(0.2, 0.8, 4, endpoint=True)
            place_pos_candidates = goal_obj.pos + place_pos_candidates[...,None]*(goal_obj2.pos - goal_obj.pos)
            thetas = jnp.linspace(0, np.pi, 2, endpoint=False, dtype=jnp.float32)
            place_quat_candidates = tutil.aa2q(thetas[...,None]*jnp.array([0,0,1.]))
            place_quat_candidates = einops.repeat(place_quat_candidates, '... i j -> ... (r i) j', r=place_pos_candidates.shape[-2])
            place_pos_candidates = einops.repeat(place_pos_candidates, '... i j -> ... (i r) j', r=thetas.shape[-1])
            place_pos_candidates = place_pos_candidates.at[...,-1].set(target_obj.pos[...,-1])
        place_pos_candidates = jax.block_until_ready(place_pos_candidates)
        self.time_tracker.set('calculate place pose')
        dt = self.time_tracker.dt['calculate place pose']
        print(f'finish calculate place pose / time {dt}')


        # evaluate rechability
        print('start eval reachability')
        self.time_tracker.set('eval reachability')
        place_pq_e, place_approach_pq_e, place_q, place_approach_q, reachability_mask = \
            self.calculate_place_q_and_reachability(place_pos_candidates, place_quat_candidates, pq_oe, rest_q, z_offset=self.place_z_offset)
        reachability_mask = jax.block_until_ready(reachability_mask)
        self.time_tracker.set('eval reachability')
        dt = self.time_tracker.dt['eval reachability']
        print(f'finish eval reachability / time {dt}')
        
        # evaluate collision
        print('start eval collision')
        self.time_tracker.set('eval collision')
        if placement_collision:
            col_mask = self.get_col_func_jit(env_obj.outer_shape[0])(place_q, place_approach_q, target_obj, pq_oe, env_obj, jkey)
        else:
            col_mask = jnp.ones_like(reachability_mask)
        _, jkey = jax.random.split(jkey)
        col_mask = jax.block_until_ready(col_mask)
        self.time_tracker.set('eval collision')
        dt = self.time_tracker.dt['eval collision']
        print(f'finish eval collision / time {dt}')

        total_mask = jnp.logical_and(reachability_mask, col_mask)

        # select place q
        q_pick_idx = jnp.argmin(jnp.linalg.norm(rest_q - place_approach_q, axis=-1) - 100*total_mask)
        place_q_select, place_approach_q_select, place_pq_select = jax.tree_util.tree_map(lambda x: x[q_pick_idx], (place_q, place_approach_q, place_pq_e))

        self.time_tracker.set('generate motion')
        if intermediate_traj is None:
                traj_grasp_to_place, _, goal_reached, motion_planning_aux_info = self.franka_rrt.execution(jkey, env_obj, grasp_q, None, place_q_select, self.plane_params, gripper_width=0.05, 
                                        refinement=True, early_stop=True, obj_in_hand=target_obj, pos_quat_eo=tutil.pq_inv(*pq_oe), non_selected_obj_pred=non_selected_obj_pred, verbose=1, compile=self.unique_id==1)
        if intermediate_traj is not None:
            traj_grasp_to_place, _, goal_reached, motion_planning_aux_info = self.franka_rrt.execution(jkey, env_obj, grasp_q, None, intermediate_traj[0], self.plane_params, gripper_width=0.05, 
                                    refinement=True, early_stop=True, obj_in_hand=target_obj, pos_quat_eo=tutil.pq_inv(*pq_oe), non_selected_obj_pred=non_selected_obj_pred, verbose=1, compile=self.unique_id==1)
            traj_grasp_to_place = jnp.concatenate([traj_grasp_to_place, intermediate_traj, place_q_select[None]], 0)
            traj_grasp_to_place = ifutil.way_points_to_trajectory(traj_grasp_to_place, 100, smoothing=False)
        traj_place_to_rest = ifutil.way_points_to_trajectory(jnp.stack([place_q_select, place_approach_q_select, rest_q], axis=0), 100, smoothing=False)
        traj_grasp_to_place = jax.block_until_ready(traj_grasp_to_place)
        self.time_tracker.set('generate motion')
        dt = self.time_tracker.dt['generate motion']
        print(f'generate motion / time {dt}')

        place_aux_info = (place_pos_candidates, place_quat_candidates, reachability_mask, col_mask, q_pick_idx, motion_planning_aux_info, goal_text_query_aux_info)

        return traj_grasp_to_place, traj_place_to_rest, place_aux_info



    def target_pnp(self, place_pos_candidates, place_quat_candidates, target_obj, env_obj, start_q, rest_q, jkey, intermediate_traj=None, rrt=False, output_grasp=False):
        self.time_tracker.set('pnp calculation')
        # calculate pick pose
        print('start grasp prediction')
        self.time_tracker.set('grasp prediction')
        # approach_len=0.050
        approach_len=0.070
        grasp_q, grasp_approaching_q, grasp_pq, grasp_approach_pq, grasp_aux_info = \
            self.get_grasp_prediction_jit(env_obj.outer_shape[0])(target_obj, *self.robot_base_pos_quat, start_q, jkey, env_obj, jnp.array([0,0,-approach_len]))
        mask_info_select, best_joint_pos, approaching_joint_pos, best_pq, approaching_pq, max_idx, mask_info = grasp_aux_info
        _, jkey = jax.random.split(jkey)
        self.time_tracker.set('grasp prediction')
        dt = self.time_tracker.dt['grasp prediction']
        print(f'finish grasp prediction / time: {dt} / mask: {mask_info_select} / num reachable {np.sum(mask_info[1])} / num non-col {np.sum(mask_info[2])}')
        pq_oe = tutil.pq_multi(*tutil.pq_inv(target_obj.pos, jnp.array([0,0,0,1.])), *grasp_pq)

        # ## grasp prediction visualization @@@test
        # camera_body = cxutil.create_box(np.array([0.025, 0.14, 0.03]), 32, 64)
        # camera_body = camera_body.set_z_with_models(jax.random.PRNGKey(0), self.models)
        # gripper_width = 0.05
        # cchecker = ifutil.SimpleCollisionChecker(self.models, self.franka_rrt.panda_link_obj.drop_gt_info(color=True), 
        #                                       env_obj.drop_gt_info(color=True), self.plane_params, *self.robot_base_pos_quat, 
        #                                       gripper_width, camera_body)
        # col_res_grasp, _ = cchecker.check_q(jkey, grasp_q.at[...,-1].add(np.pi/4.), visualize=True) # compensate last joint
        # print(col_res_grasp)
        # col_res_grasp, _ = cchecker.check_q(jkey, grasp_q, visualize=True) # compensate last joint
        # print(col_res_grasp)

        # evaluate rechability
        print('start eval reachability')
        self.time_tracker.set('eval reachability')
        place_pq_e, place_approach_pq_e, place_q, place_approach_q, reachability_mask = \
            self.calculate_place_q_and_reachability(place_pos_candidates, place_quat_candidates, pq_oe, rest_q, z_offset=self.place_z_offset)
        reachability_mask = jax.block_until_ready(reachability_mask)
        self.time_tracker.set('eval reachability')
        dt = self.time_tracker.dt['eval reachability']
        print(f'finish eval reachability / time {dt}')
        
        # evaluate collision
        print('start eval collision')
        self.time_tracker.set('eval collision')
        # col_mask = self.get_col_func_jit(env_obj.outer_shape[0])(place_q, place_approach_q, target_obj, pq_oe, env_obj, jkey)
        col_mask = jnp.ones_like(reachability_mask)
        _, jkey = jax.random.split(jkey)
        col_mask = jax.block_until_ready(col_mask)
        self.time_tracker.set('eval collision')
        dt = self.time_tracker.dt['eval collision']
        print(f'finish eval collision / time {dt}')

        total_mask = jnp.logical_and(reachability_mask, col_mask)

        # select place q
        q_pick_idx = jnp.argmin(jnp.linalg.norm(rest_q - place_approach_q, axis=-1) - 100*total_mask)
        place_q_select, place_approach_q_select, place_pq_select = jax.tree_util.tree_map(lambda x: x[q_pick_idx], (place_q, place_approach_q, place_pq_e))

        self.time_tracker.set('generate motion')
        traj_rest_to_grasp, traj_grasp_to_place, traj_place_to_rest = self.generate_pick_and_place_motion(grasp_q, grasp_approaching_q,
                                                                                                           place_pq_select, place_q_select, 
                                                                                                           place_approach_q_select, start_q, rest_q,
                                                                                                           env_obj, target_obj, pq_oe,
                                                                                                           jkey, intermediate_traj=intermediate_traj, rrt=rrt)
        traj_place_to_rest = jax.block_until_ready(traj_place_to_rest)
        self.time_tracker.set('generate motion')
        dt = self.time_tracker.dt['generate motion']
        print(f'generate motion / time {dt}')

        self.time_tracker.set('pnp calculation')
        dt = self.time_tracker.dt['pnp calculation']
        print(f'entire pnp calculation / time {dt}')
        if output_grasp:
            return (traj_rest_to_grasp, traj_grasp_to_place, traj_place_to_rest), grasp_pq
        else:
            return traj_rest_to_grasp, traj_grasp_to_place, traj_place_to_rest



    def rearrange_categorical_objects(self, geom_objrep:cxutil.LatentObjects, lan_aligned_objrep:jnp.ndarray, cat_per_obj:np.ndarray, target_cat:str, rest_q, jkey):
        '''
        geom_objrep: outer_shape - (NO, )
        lan_aligned_objrep: (NO, NF)
        cat_per_obj: List[str] size (NO, ) - category
        '''
        # get obb from objects
        self.time_tracker.set('obb')
        obbs = self.get_obb_func_jit(geom_objrep.outer_shape[0])(geom_objrep)
        obbs = jax.block_until_ready(obbs)
        self.time_tracker.set('obb')
        dt = self.time_tracker.dt['obb']
        print(f'obb cal dt: {dt}')
        
        # reorder by size
        obj_in_cat_indices = np.where(cat_per_obj==target_cat)[0]
        print(f'target object number: {len(obj_in_cat_indices)}')
        geom_objrep_cat, lan_aligned_objrep_cat, obbs = jax.tree_util.tree_map(lambda x: x[obj_in_cat_indices], (geom_objrep, lan_aligned_objrep, obbs))
        representative_lengths = jnp.max(obbs[-1], axis=-1)
        obj_idx_sorted = jnp.argsort(-representative_lengths)

        alignement_offset = 0.40
        if target_cat == 'bowl':
            # target_aligned_positions = 0.4*jnp.ones(len(obj_in_cat_indices), dtype=jnp.float32)
            target_aligned_positions = 0.36*jnp.ones(len(obj_in_cat_indices), dtype=jnp.float32)
            z_values = jnp.arange(len(obj_in_cat_indices), dtype=jnp.float32) * 0.050
        else:
            target_aligned_positions = np.linspace(0.55, 0.7, 2, endpoint=True)
            z_values = jnp.zeros_like(target_aligned_positions)
        target_aligned_positions = jnp.stack([target_aligned_positions, 
                                             alignement_offset*jnp.ones_like(target_aligned_positions),
                                             z_values], -1)
        target_aligned_positions += self.robot_base_pos_quat[0]
        traj_list = []
        for i, oid in enumerate(obj_idx_sorted):
            print(f'trial {i} obj len: {representative_lengths[oid]}')
            self.time_tracker.set('pnp traj')
            cur_obj, cur_obb, global_idx = jax.tree_util.tree_map(lambda x: x[oid], (geom_objrep_cat, obbs, obj_in_cat_indices))
            env_obj = jax.tree_util.tree_map(lambda x: x[:global_idx], geom_objrep)
            env_obj = env_obj.concat(jax.tree_util.tree_map(lambda x: x[global_idx+1:], geom_objrep), axis=0)
            # env_obj = env_obj.concat(self.camera_bodies, axis=0)

            # pnp motions
            target_pos = target_aligned_positions[i].at[...,-1].add(cur_obj.pos[...,-1])
            thetas = jnp.linspace(0, np.pi, 8, endpoint=False)
            target_quat_candidates = tutil.aa2q(thetas[...,None]*jnp.array([0,0,1.]))
            target_pos_candidates = target_pos + target_quat_candidates[...,:1]*0
            traj_list.append(self.target_pnp(target_pos_candidates, target_quat_candidates, cur_obj.extend_outer_shape(axis=0), env_obj, rest_q, rest_q, jkey))
            # move object
            geom_objrep = geom_objrep.replace(pos=geom_objrep.pos.at[global_idx].set(target_pos))
            _, jkey = jax.random.split(jkey)
            traj_list = jax.block_until_ready(traj_list)
            self.time_tracker.set('pnp traj')
            dt = self.time_tracker.dt['pnp traj']
            print(f'end traj calculations / dt: {dt}')

        return traj_list

    @partial(jax.jit, static_argnums=(0, 5))
    def calculate_place_q_and_reachability(self, target_pos_candidates, target_quat_candidates, pq_oe, rest_q, z_offset=0.080):
        '''
        input:
            target_pos_candidates: (NS, 3)
            target_quat_candidates: (NS, 4)
            pq_oe: [(3,), (4,)]
            rest_q: (7,)
        
        output:
            reachability_mask: (NS,)

        '''
        place_pq_e = tutil.pq_multi(target_pos_candidates, target_quat_candidates, *pq_oe)
        place_approach_pq = (place_pq_e[0] + jnp.array([0,0,z_offset+0.070]), place_pq_e[1])
        place_pq_e = (place_pq_e[0] + jnp.array([0,0,z_offset]), place_pq_e[1]) # add z directional offset for placing position
        place_q, (place_pos_cost, place_quat_cost) =\
            jax.vmap(partial(fkutil.Franka_IK_numetrical, robot_base_pos=self.robot_base_pos_quat[0], 
                                robot_base_quat=self.robot_base_pos_quat[1], itr_no=300, output_cost=True, grasp_basis=False, compensate_last_joint=True), (None,0))(
            rest_q, place_pq_e)
        place_approach_q, (place_approach_pos_cost, place_approach_quat_cost) =\
            jax.vmap(partial(fkutil.Franka_IK_numetrical, robot_base_pos=self.robot_base_pos_quat[0], 
                                robot_base_quat=self.robot_base_pos_quat[1], itr_no=300, output_cost=True, grasp_basis=False, compensate_last_joint=True), (0,0))(
            place_q, place_approach_pq)

        reachability_mask_place = jnp.logical_and(place_pos_cost < latent_grasp.REACHABILITY_POS_LIMIT, place_quat_cost < latent_grasp.REACHABILITY_QUAT_LIMIT)
        reachability_mask_approaching = jnp.logical_and(place_approach_pos_cost < latent_grasp.REACHABILITY_POS_LIMIT, place_approach_quat_cost < latent_grasp.REACHABILITY_QUAT_LIMIT)
        reachability_mask = jnp.logical_and(reachability_mask_place, reachability_mask_approaching)
        return place_pq_e, place_approach_pq, place_q, place_approach_q, reachability_mask

    def evaluate_place_collision(self, place_q, approach_q, target_obj, pq_oe, env_objs, jkey):
        '''
        input:
            place_q: (NS, 7)
            approach_q: (NS, 7)
        '''
        # object interactions
        camera_body = cxutil.create_box(np.array([0.025, 0.14, 0.03]), 32, 64)
        camera_body = camera_body.set_z_with_models(jax.random.PRNGKey(0), self.models)
        gripper_width = 0.05
        panda_link_obj = self.franka_rrt.panda_link_obj
        cchecker = ifutil.SimpleCollisionChecker(self.models, panda_link_obj.drop_gt_info(color=True), 
                                              env_objs.drop_gt_info(color=True), self.plane_params, *self.robot_base_pos_quat, 
                                              gripper_width, camera_body, target_obj, tutil.pq_inv(*pq_oe))
        col_res_place, _ = jax.vmap(cchecker.check_q, (None, 0))(jkey, place_q)
        col_res_approach, _ = jax.vmap(cchecker.check_q, (None, 0))(jkey, approach_q)
        # col_res_place, _ = jax.vmap(cchecker.check_q, (None, 0))(jkey, place_q.at[...,-1].add(np.pi/4.))
        # col_res_approach, _ = jax.vmap(cchecker.check_q, (None, 0))(jkey, approach_q.at[...,-1].add(np.pi/4.))
        col_mask = jnp.logical_not(jnp.logical_or(col_res_place, col_res_approach))
        return col_mask

    def generate_pick_and_place_motion(self, grasp_q, grasp_approach_q, place_pq_select, place_q_select, 
                                       place_approach_q_select, start_q, rest_q, env_objs, target_obj, pq_oe, jkey, intermediate_traj=None, rrt=False):

        place_back_pq = tutil.pq_multi(*place_pq_select, jnp.array([0,0,-0.06]), jnp.array([0,0,0,1.]))
        place_back_q = self.get_ik_func_jit(1)(place_q_select, place_back_pq)

        transition_q = GRASP_MID_Q

        if rrt==2:
            traj_rest_to_grasp1, _, goal_reached, _ = self.franka_rrt.execution(jkey, env_objs, start_q, None, grasp_approach_q, self.plane_params, gripper_width=0.05, 
                                                                                refinement=True, early_stop=True, verbose=1)
            traj_rest_to_grasp2 = ifutil.way_points_to_trajectory(jnp.stack([traj_rest_to_grasp1[-1], grasp_q], axis=0), 100, smoothing=False)
            traj_rest_to_grasp = jnp.concatenate([traj_rest_to_grasp1, traj_rest_to_grasp2], 0)
            traj_grasp_to_place, _, goal_reached, _ = self.franka_rrt.execution(jkey, env_objs, grasp_q, None, place_q_select, self.plane_params, gripper_width=0.05, 
                                      refinement=True, early_stop=True, obj_in_hand=target_obj, pos_quat_eo=tutil.pq_inv(*pq_oe), verbose=1)
        elif rrt==1:
            traj_rest_to_grasp = ifutil.way_points_to_trajectory(jnp.stack([start_q, grasp_approach_q, grasp_q], axis=0), 100, smoothing=False)
            if intermediate_traj is None:
                traj_grasp_to_place, _, goal_reached, _ = self.franka_rrt.execution(jkey, env_objs, grasp_q, None, place_q_select, self.plane_params, gripper_width=0.05, 
                                        refinement=True, early_stop=True, obj_in_hand=target_obj, pos_quat_eo=tutil.pq_inv(*pq_oe), verbose=1)
            if intermediate_traj is not None:
                traj_grasp_to_place, _, goal_reached, _ = self.franka_rrt.execution(jkey, env_objs, grasp_q, None, intermediate_traj[0], self.plane_params, gripper_width=0.05, 
                                        refinement=True, early_stop=True, obj_in_hand=target_obj, pos_quat_eo=tutil.pq_inv(*pq_oe), verbose=1)
                traj_grasp_to_place = jnp.concatenate([traj_grasp_to_place, intermediate_traj, place_q_select[None]], 0)
                traj_grasp_to_place = ifutil.way_points_to_trajectory(traj_grasp_to_place, 100, smoothing=False)
        else:
            traj_rest_to_grasp = ifutil.way_points_to_trajectory(jnp.stack([start_q, grasp_approach_q, grasp_q], axis=0), 100, smoothing=False)
            traj_mid = ifutil.Bezier_curve_3points(grasp_approach_q, transition_q, place_approach_q_select, jnp.linspace(0,1,10,True))
            traj_grasp_to_place = ifutil.way_points_to_trajectory(jnp.concatenate([grasp_q[None], traj_mid, place_q_select[None]], axis=0), 100, smoothing=False)
        traj_place_to_rest = ifutil.way_points_to_trajectory(jnp.stack([place_q_select, place_back_q, place_approach_q_select, rest_q], axis=0), 100, smoothing=False)

        return traj_rest_to_grasp, traj_grasp_to_place, traj_place_to_rest

    def pnp_to_with_relation(self, geom_objrep:cxutil.LatentObjects, lan_aligned_objrep:jnp.ndarray, 
                             target_obj_text, goal_obj_text, goal_relation_type, rest_q, jkey, clip_aux_info=None, goal_obj_text2=None, 
                             rrt=False):
        self.time_tracker.set('pnp planning')

        print('start text query')
        self.time_tracker.set('text query')
        target_idx, target_text_query_aux_info = self.text_query(lan_aligned_objrep, target_obj_text, visualize=True, clip_aux_info=clip_aux_info)
        target_obj = jax.tree_util.tree_map(lambda x: x[target_idx:target_idx+1], geom_objrep) # predefine selections
        if goal_obj_text is not None:
            goal_idx, goal_text_query_aux_info = self.text_query(lan_aligned_objrep, goal_obj_text, visualize=True, clip_aux_info=clip_aux_info)
            goal_obj = jax.tree_util.tree_map(lambda x: x[goal_idx:goal_idx+1], geom_objrep) # predefine selections
        if goal_obj_text2 is not None:
            goal_idx2, goal2_text_query_aux_info = self.text_query(lan_aligned_objrep, goal_obj_text2, visualize=True, clip_aux_info=clip_aux_info)
            goal_obj2 = jax.tree_util.tree_map(lambda x: x[goal_idx2:goal_idx2+1], geom_objrep) # predefine selections
        env_obj = jax.tree_util.tree_map(lambda x: x[:target_idx], geom_objrep) # predefine selections
        env_obj = env_obj.concat(jax.tree_util.tree_map(lambda x: x[target_idx+1:], geom_objrep), axis=0)
        # env_obj = env_obj.concat(self.camera_bodies, axis=0)
        env_obj = jax.block_until_ready(env_obj)
        self.time_tracker.set('text query')
        dt = self.time_tracker.dt['text query']
        print(f'finish text query / time {dt}')

        # calculate place pose
        self.time_tracker.set('calculate place pose')
        if goal_obj_text is None or goal_relation_type is None:
            # place_pos_candidates = jnp.array([0.4,0.4,0.2])
            # place_pos_candidates = jnp.array([0.35,-0.35,0.2])
            place_pos_candidates = self.robot_base_pos_quat[0] + jnp.array([0.0,0.5,0.2])
            thetas = jnp.linspace(0, np.pi, 8, endpoint=False)
            place_quat_candidates = tutil.aa2q(thetas[...,None]*jnp.array([0,0,1.]))
            place_pos_candidates += place_quat_candidates[...,:1]*0 # broadcasting
        
        # place pose candidates
        elif goal_relation_type in ['up', 'inside']:
            place_pos_candidates = goal_obj.pos + jnp.array([[0,0,0.06]], dtype=jnp.float32)
            thetas = jnp.linspace(0, np.pi, 8, endpoint=False)
            place_quat_candidates = tutil.aa2q(thetas[...,None]*jnp.array([0,0,1.]))
            place_pos_candidates += place_quat_candidates[...,:1]*0 # broadcasting
        elif goal_relation_type in ['left', 'right', 'front', 'behind']:
            place_pos_candidates = jnp.linspace(0.05, 0.20, 4, endpoint=True, dtype=jnp.float32)
            if goal_relation_type == 'left':
                base_axis = jnp.array([0,1.,0], dtype=jnp.float32)
            elif goal_relation_type == 'right':
                base_axis = jnp.array([0,-1.,0], dtype=jnp.float32)
            elif goal_relation_type == 'front':
                base_axis = jnp.array([1.0,0,0], dtype=jnp.float32)
            elif goal_relation_type == 'behind':
                base_axis = jnp.array([-1.0,0,0], dtype=jnp.float32)
            place_pos_candidates = goal_obj.pos + place_pos_candidates[...,None]*base_axis
            thetas = jnp.linspace(0, np.pi, 2, endpoint=False, dtype=jnp.float32)
            place_quat_candidates = tutil.aa2q(thetas[...,None]*jnp.array([0,0,1.]))
            place_quat_candidates = einops.repeat(place_quat_candidates, '... i j -> ... (r i) j', r=place_pos_candidates.shape[-2])
            place_pos_candidates = einops.repeat(place_pos_candidates, '... i j -> ... (i r) j', r=thetas.shape[-1])
            place_pos_candidates = place_pos_candidates.at[...,-1].set(target_obj.pos[...,-1])
        elif goal_relation_type in ['between']:
            place_pos_candidates = jnp.linspace(0.2, 0.8, 4, endpoint=True)
            place_pos_candidates = goal_obj.pos + place_pos_candidates[...,None]*(goal_obj2.pos - goal_obj.pos)
            thetas = jnp.linspace(0, np.pi, 2, endpoint=False, dtype=jnp.float32)
            place_quat_candidates = tutil.aa2q(thetas[...,None]*jnp.array([0,0,1.]))
            place_quat_candidates = einops.repeat(place_quat_candidates, '... i j -> ... (r i) j', r=place_pos_candidates.shape[-2])
            place_pos_candidates = einops.repeat(place_pos_candidates, '... i j -> ... (i r) j', r=thetas.shape[-1])
            place_pos_candidates = place_pos_candidates.at[...,-1].set(target_obj.pos[...,-1])
        place_pos_candidates = jax.block_until_ready(place_pos_candidates)
        self.time_tracker.set('calculate place pose')
        dt = self.time_tracker.dt['calculate place pose']
        print(f'finish calculate place pose / time {dt}')

        traj_list, grasp_pq = self.target_pnp(place_pos_candidates, place_quat_candidates, target_obj, env_obj, rest_q, rest_q, jkey, rrt=rrt, output_grasp=True)
        traj_list = jax.block_until_ready(traj_list)
        self.time_tracker.set('pnp planning')
        dt = self.time_tracker.dt['pnp planning']
        print(f'entire pnp planning / time {dt}')

        return [traj_list], grasp_pq

    def predict_reachable_grasp(self, target_obj, env_obj, rest_q, jkey, inf_aux_info=None):
        print('start grasp prediction')
        self.time_tracker.set('grasp prediction')

        if inf_aux_info is not None:
            idx_mask = jnp.arange(inf_aux_info.x_pred.outer_shape[0]) != inf_aux_info.obs_max_idx
            idx_where = jnp.where(idx_mask)
            non_selected_obj_pred = jax.tree_util.tree_map(lambda x: x[idx_where], inf_aux_info.x_pred)
        else:
            non_selected_obj_pred = None
        # approach_len = 0.25
        if len(target_obj.outer_shape) == 0:
            target_obj = target_obj.expand_outer_shape(0)
        cam_offset = jnp.array([-0.057,0,-0.23], dtype=jnp.float32)
        rest_q = jnp.array(rest_q).astype(jnp.float32)
        grasp_q, grasp_approaching_q, grasp_pq, grasp_approach_pq, grasp_aux_info = \
            self.get_grasp_prediction_jit(env_obj.outer_shape[0])(target_obj, *self.robot_base_pos_quat, rest_q, jkey, env_obj, cam_offset, non_selected_obj_pred=non_selected_obj_pred)
        mask_info_select, best_joint_pos, approaching_joint_pos, best_pq, approaching_pq, max_idx, mask_info = grasp_aux_info
        _, jkey = jax.random.split(jkey)
        self.time_tracker.set('grasp prediction')
        dt = self.time_tracker.dt['grasp prediction']
        print(f'finish grasp prediction / time: {dt} / mask: {mask_info_select} / num reachable {np.sum(mask_info[1])} / num non-col {np.sum(mask_info[2])}')

        taking_picture_traj = ifutil.way_points_to_trajectory(jnp.stack([rest_q, grasp_approaching_q], 0), 100)

        return taking_picture_traj, grasp_pq, grasp_aux_info


    def pregrasp_traj_with_target(self, geom_objrep:cxutil.LatentObjects, lan_aligned_objrep:jnp.ndarray, 
                             target_obj_text, rest_q, jkey, clip_aux_info=None, inf_aux_info=None, valid_target_mask=None, category_per_obj=None):

        print('start text query')
        self.time_tracker.set('text query')
        try:
            size_order = int(target_obj_text[:1])
            target_cat = target_obj_text[1:]

            # get obb from objects
            self.time_tracker.set('obb')
            obbs = self.get_obb_func_jit(geom_objrep.outer_shape[0])(geom_objrep)
            obbs = jax.block_until_ready(obbs)
            self.time_tracker.set('obb')
            dt = self.time_tracker.dt['obb']
            print(f'obb cal dt: {dt}')
            
            # reorder by size
            obj_in_cat_indices = np.where(category_per_obj==target_cat)[0]
            print(f'target object number: {len(obj_in_cat_indices)}')
            geom_objrep_cat, lan_aligned_objrep_cat, obbs = jax.tree_util.tree_map(lambda x: x[obj_in_cat_indices], (geom_objrep, lan_aligned_objrep, obbs))
            representative_lengths = jnp.max(obbs[-1], axis=-1)
            obj_idx_sorted = jnp.argsort(representative_lengths)
            target_idx = obj_in_cat_indices[obj_idx_sorted[size_order]]
            target_text_query_aux_info = None
        except:
            target_idx, target_text_query_aux_info = self.text_query(lan_aligned_objrep, target_obj_text, visualize=clip_aux_info is not None, clip_aux_info=clip_aux_info, valid_target_mask=valid_target_mask)
        target_obj = jax.tree_util.tree_map(lambda x: x[target_idx:target_idx+1], geom_objrep) # predefine selections
        env_obj = jax.tree_util.tree_map(lambda x: x[:target_idx], geom_objrep) # predefine selections
        env_obj = env_obj.concat(jax.tree_util.tree_map(lambda x: x[target_idx+1:], geom_objrep), axis=0)
        env_obj = jax.block_until_ready(env_obj)
        self.time_tracker.set('text query')
        dt = self.time_tracker.dt['text query']
        print(f'finish text query / time {dt}')

        return *self.predict_reachable_grasp(target_obj, env_obj, rest_q, jkey, inf_aux_info=inf_aux_info), target_obj.squeeze_outer_shape(0), target_idx, target_text_query_aux_info



    def postgrasp_pnp_with_relation(self, geom_objrep:cxutil.LatentObjects, lan_aligned_objrep:jnp.ndarray, 
                             target_idx, goal_obj_text, goal_relation_type, start_q, rest_q, jkey, intermediate_traj=None, clip_aux_info=None, goal_obj_text2=None, 
                             rrt=False):
        self.time_tracker.set('pnp planning')

        print('start text query')
        self.time_tracker.set('text query')
        target_obj = jax.tree_util.tree_map(lambda x: x[target_idx:target_idx+1], geom_objrep) # predefine selections
        if goal_obj_text is not None:
            goal_idx, goal_text_query_aux_info = self.text_query(lan_aligned_objrep, goal_obj_text, visualize=False, clip_aux_info=clip_aux_info)
            goal_obj = jax.tree_util.tree_map(lambda x: x[goal_idx:goal_idx+1], geom_objrep) # predefine selections
        if goal_obj_text2 is not None:
            goal_idx2, goal2_text_query_aux_info = self.text_query(lan_aligned_objrep, goal_obj_text2, visualize=False, clip_aux_info=clip_aux_info)
            goal_obj2 = jax.tree_util.tree_map(lambda x: x[goal_idx2:goal_idx2+1], geom_objrep) # predefine selections
        env_obj = jax.tree_util.tree_map(lambda x: x[:target_idx], geom_objrep) # predefine selections
        env_obj = env_obj.concat(jax.tree_util.tree_map(lambda x: x[target_idx+1:], geom_objrep), axis=0)
        # env_obj = env_obj.concat(self.camera_bodies, axis=0)
        env_obj = jax.block_until_ready(env_obj)
        self.time_tracker.set('text query')
        dt = self.time_tracker.dt['text query']
        print(f'finish text query / time {dt}')

        # calculate place pose
        self.time_tracker.set('calculate place pose')
        if goal_obj_text is None or goal_relation_type is None:
            if self.env_type == 'table':
                # place_pos_candidates = jnp.array([0.4,0.4,0.2])
                place_pos_candidates = jnp.array([0.35,-0.35,0.2])
            elif self.env_type == 'shelf':
                # place_pos_candidates = jnp.array([-0.320,0.2,0.1])
                # place_pos_candidates = self.robot_base_pos_quat[0] + jnp.array([0.0,0.4,0.1])
                # place_pos_candidates = self.robot_base_pos_quat[0] + jnp.array([0.5,0.0,0.1])
                place_pos_candidates = self.robot_base_pos_quat[0] + jnp.array([0.5,0.0,0.97])
            thetas = jnp.linspace(0, np.pi, 8, endpoint=False)
            place_quat_candidates = tutil.aa2q(thetas[...,None]*jnp.array([0,0,1.]))
            place_pos_candidates += place_quat_candidates[...,:1]*0 # broadcasting
        
        # place pose candidates
        elif goal_relation_type in ['up', 'inside']:
            place_pos_candidates = goal_obj.pos + jnp.array([[0,0,0.06]], dtype=jnp.float32)
            thetas = jnp.linspace(0, np.pi, 8, endpoint=False)
            place_quat_candidates = tutil.aa2q(thetas[...,None]*jnp.array([0,0,1.]))
            place_pos_candidates += place_quat_candidates[...,:1]*0 # broadcasting
        elif goal_relation_type in ['left', 'right', 'front', 'behind']:
            place_pos_candidates = jnp.linspace(0.05, 0.20, 4, endpoint=True, dtype=jnp.float32)
            if goal_relation_type == 'left':
                base_axis = jnp.array([0,1.,0], dtype=jnp.float32)
            elif goal_relation_type == 'right':
                base_axis = jnp.array([0,-1.,0], dtype=jnp.float32)
            elif goal_relation_type == 'front':
                base_axis = jnp.array([1.0,0,0], dtype=jnp.float32)
            elif goal_relation_type == 'behind':
                base_axis = jnp.array([-1.0,0,0], dtype=jnp.float32)
            place_pos_candidates = goal_obj.pos + place_pos_candidates[...,None]*base_axis
            thetas = jnp.linspace(0, np.pi, 2, endpoint=False, dtype=jnp.float32)
            place_quat_candidates = tutil.aa2q(thetas[...,None]*jnp.array([0,0,1.]))
            place_quat_candidates = einops.repeat(place_quat_candidates, '... i j -> ... (r i) j', r=place_pos_candidates.shape[-2])
            place_pos_candidates = einops.repeat(place_pos_candidates, '... i j -> ... (i r) j', r=thetas.shape[-1])
            place_pos_candidates = place_pos_candidates.at[...,-1].set(target_obj.pos[...,-1])
        elif goal_relation_type in ['between']:
            place_pos_candidates = jnp.linspace(0.2, 0.8, 4, endpoint=True)
            place_pos_candidates = goal_obj.pos + place_pos_candidates[...,None]*(goal_obj2.pos - goal_obj.pos)
            thetas = jnp.linspace(0, np.pi, 2, endpoint=False, dtype=jnp.float32)
            place_quat_candidates = tutil.aa2q(thetas[...,None]*jnp.array([0,0,1.]))
            place_quat_candidates = einops.repeat(place_quat_candidates, '... i j -> ... (r i) j', r=place_pos_candidates.shape[-2])
            place_pos_candidates = einops.repeat(place_pos_candidates, '... i j -> ... (i r) j', r=thetas.shape[-1])
            place_pos_candidates = place_pos_candidates.at[...,-1].set(target_obj.pos[...,-1])
        place_pos_candidates = jax.block_until_ready(place_pos_candidates)
        self.time_tracker.set('calculate place pose')
        dt = self.time_tracker.dt['calculate place pose']
        print(f'finish calculate place pose / time {dt}')

        traj_list, grasp_pq = self.target_pnp(place_pos_candidates, place_quat_candidates, target_obj, env_obj, start_q, rest_q, jkey,
                                              intermediate_traj=intermediate_traj, rrt=rrt, output_grasp=True)
        traj_list = jax.block_until_ready(traj_list)
        self.time_tracker.set('pnp planning')
        dt = self.time_tracker.dt['pnp planning']
        print(f'entire pnp planning / time {dt}')

        return [traj_list], grasp_pq

    def pour_water(self, geom_objrep, lan_aligned_objrep, target_obj_text, goal_obj_text, rest_q, jkey, clip_aux_info=None, rrt=False):

        # grasp prediction with side
        print('start text query')
        self.time_tracker.set('text query')
        target_idx, target_text_query_aux_info = self.text_query(lan_aligned_objrep, target_obj_text, visualize=True, clip_aux_info=clip_aux_info)
        target_obj = jax.tree_util.tree_map(lambda x: x[target_idx:target_idx+1], geom_objrep) # predefine selections
        goal_idx, goal_text_query_aux_info = self.text_query(lan_aligned_objrep, goal_obj_text, visualize=True, clip_aux_info=clip_aux_info)
        goal_obj = jax.tree_util.tree_map(lambda x: x[goal_idx:goal_idx+1], geom_objrep) # predefine selections
        env_obj = jax.tree_util.tree_map(lambda x: x[:target_idx], geom_objrep) # predefine selections
        env_obj = env_obj.concat(jax.tree_util.tree_map(lambda x: x[target_idx+1:], geom_objrep), axis=0)
        env_obj = env_obj.concat(self.camera_bodies, axis=0)
        env_obj = jax.block_until_ready(env_obj)
        self.time_tracker.set('text query')
        dt = self.time_tracker.dt['text query']
        print(f'finish text query / time {dt}')

        # calculate pick pose
        print('start grasp prediction')
        self.time_tracker.set('grasp prediction')
        grasp_q, grasp_approaching_q, best_pq, approaching_pq, grasp_aux_info = \
            self.grasp_prediction_side_func(target_obj, *self.robot_base_pos_quat, rest_q, jkey, env_obj)
        _, jkey = jax.random.split(jkey)
        self.time_tracker.set('grasp prediction')
        dt = self.time_tracker.dt['grasp prediction']
        print(f'finish grasp prediction / time {dt}')
        pq_oe = tutil.pq_multi(*tutil.pq_inv(target_obj.pos, jnp.array([0,0,0,1.])), *best_pq)

        # calculate place pose
        self.time_tracker.set('calculate place pose')
        place_pos_candidates = goal_obj.pos + jnp.array([[0,0,0.06]], dtype=jnp.float32)
        thetas = jnp.linspace(0, np.pi, 8, endpoint=False)
        place_quat_candidates = tutil.aa2q(thetas[...,None]*jnp.array([0,0,1.]))
        place_pos_candidates += place_quat_candidates[...,:1]*0 # broadcasting

        # evaluate rechability
        print('start eval reachability')
        self.time_tracker.set('eval reachability')
        place_pq_e, place_approach_pq_e, place_q, place_approach_q, reachability_mask = \
            self.calculate_place_q_and_reachability(place_pos_candidates, place_quat_candidates, pq_oe, rest_q, z_offset=0.08)
        self.time_tracker.set('eval reachability')
        dt = self.time_tracker.dt['eval reachability']
        print(f'finish eval reachability / time {dt}')
        
        # evaluate collision
        print('start eval collision')
        self.time_tracker.set('eval collision')
        col_mask = self.get_col_func_jit(env_obj.outer_shape[0])(place_q, place_approach_q, target_obj, pq_oe, env_obj, jkey)
        # col_mask = self.evaluate_place_collision(place_q, place_approach_q, target_obj, pq_oe, env_obj, jkey)
        _, jkey = jax.random.split(jkey)
        self.time_tracker.set('eval collision')
        dt = self.time_tracker.dt['eval collision']
        print(f'finish eval collision / time {dt}')

        total_mask = jnp.logical_and(reachability_mask, col_mask)
        self.time_tracker.set('calculate place pose')
        dt = self.time_tracker.dt['calculate place pose']
        print(f'finish calculate place pose / time {dt}')

        # select place q
        q_pick_idx = jnp.argmin(jnp.linalg.norm(rest_q - place_approach_q, axis=-1) - 100*total_mask)
        place_q_select, place_approach_q_select, place_pq_select = jax.tree_util.tree_map(lambda x: x[q_pick_idx], (place_q, place_approach_q, place_pq_e))

        traj_rest_to_grasp, traj_grasp_to_place, traj_place_to_rest = self.generate_pick_and_place_motion(grasp_q, grasp_approaching_q,
                                                                                                           place_pq_select, place_q_select, 
                                                                                                           place_approach_q_select, rest_q, rest_q,
                                                                                                           env_obj, target_obj, pq_oe,
                                                                                                           jkey, rrt=rrt)
        traj_rest_to_grasp = ifutil.interval_based_interpolations(traj_rest_to_grasp, self.interpolation_gap)
        traj_grasp_to_place = ifutil.interval_based_interpolations(traj_grasp_to_place, self.interpolation_gap)
        traj_place_to_rest = ifutil.interval_based_interpolations(traj_place_to_rest, self.interpolation_gap)
        return [[traj_rest_to_grasp, traj_grasp_to_place, traj_place_to_rest]]



    def robot_execution_in_sim(self, robot, planning_output):
        (traj1, grasp_traj, traj2), (target_obj_idx, pos_quat_eo), (grasp_aux_info, motion_planning_aux_info1, motion_planning_aux_info2) = planning_output

        robot.reset(traj1[0])

        robot.transparent()
        print('planning visualization')
        print('grasp visualization')
        best_joint_pos, approaching_joint_pos = grasp_aux_info[1], grasp_aux_info[2]
        for q in best_joint_pos:
            robot.reset(q)
            time.sleep(0.04)

        robot.reset(traj1[0])
        time.sleep(0.5)
        print('planning visualization')
        for q in motion_planning_aux_info1[0][:motion_planning_aux_info1[1]]:
            robot.reset(q)
            time.sleep(0.002)

        for q in motion_planning_aux_info2[0][:motion_planning_aux_info2[1]]:
            robot.reset(q)
            time.sleep(0.002)

        # Execute
        print('motion solution')
        sim_dt = pb.getPhysicsEngineParameters()['fixedTimeStep']
        robot.transparent(1.0)
        robot.reset(traj1[0])
        def follow_traj(traj_):
            for q in traj_:
                robot.update_arm_control(q)
                for _ in range(int(0.05/sim_dt)):
                    pb.stepSimulation()
                time.sleep(0.05)
        
        robot.reset_finger(robot.finger_travel/2.0)
        follow_traj(traj1)
        follow_traj(grasp_traj)
        robot.close_gripper(sleep=True)
        follow_traj(traj2)
        robot.release()


    def execution(self, images, camera_posquat, intrinsic, target_obj_text, jkey, visualize=False):
        obj_pred_select, language_aligned_obj_feat, clip_aux_info = self.estimation(images, camera_posquat, intrinsic, jkey)
        _, jkey = jax.random.split(jkey)
        traj1, grasp_traj, traj2 = self.fetch_target_obj(target_obj_text, obj_pred_select, language_aligned_obj_feat, jkey, clip_aux_info, visualize=visualize)
        return traj1, grasp_traj, traj2

from imm_pb_util.bullet_envs.env import BulletEnv
from examples.pb_examples.common.common import configure_bullet

class SimPnPEnv(BulletEnv):
    """
    ?
    """
    def __init__(self, bc: BulletClient, env_type):
        """Load ground plane.

        Args:
            bc (BulletClient): _
        """
        super().__init__(bc, False)

        # table_id = bc.createMultiBody(
        #         baseMass = 0.0, 
        #         basePosition = [-0.2,0,-0.1],
        #         baseCollisionShapeIndex = bc.createCollisionShape(bc.GEOM_BOX, halfExtents=[1.0, 0.8, 0.1]),
        #         baseVisualShapeIndex = bc.createVisualShape(bc.GEOM_BOX, halfExtents=[1.0, 0.8, 0.1]))
        #     #   Table height randomization
        # # bc.changeVisualShape(table_id, -1, rgbaColor=[0.9670298390136767, 0.5472322491757223, 0.9726843599648843, 1.0])
        # bc.changeVisualShape(table_id, -1, rgbaColor=list(np.random.uniform(0,1,size=(3,)))+[1,])
        # # Register
        # self.env_assets['table'] = table_id


        if env_type=='table':
            table_id = bc.createMultiBody(
                    baseMass = 0.0, 
                    basePosition = [-0.2,0,-0.1],
                    baseCollisionShapeIndex = bc.createCollisionShape(bc.GEOM_BOX, halfExtents=[1.0, 0.8, 0.1]),
                    baseVisualShapeIndex = bc.createVisualShape(bc.GEOM_BOX, halfExtents=[1.0, 0.8, 0.1]))
                #   Table height randomization
            # bc.changeVisualShape(table_id, -1, rgbaColor=[0.9670298390136767, 0.5472322491757223, 0.9726843599648843, 1.0])
            bc.changeVisualShape(table_id, -1, rgbaColor=list(np.random.uniform(0,1,size=(3,)))+[1,])
            # Register
            self.env_assets['table'] = table_id
        elif env_type=='shelf':
            table_id = bc.loadURDF(fileName='imm_pb_util/urdf/cabinet/cabinet_topless.urdf')
            self.env_assets['shelf'] = table_id


    @property
    def env_uids(self) -> List[int]:
        return list(self.env_assets.values())

class VirtualWorld(object):
    
    def __init__(self, robot_base_pos_quat, env_type='table'):
        # bc = BulletClient(connection_mode=pb.GUI)
        # if env_type=='table':
        config_file_path = Path(__file__).parent.parent / 'examples/pb_examples' / "pb_cfg" / "pb_sim_pnp.yaml"
        # elif env_type=='shelf':
        #     config_file_path = Path(__file__).parent.parent / 'examples/pb_examples' / "pb_cfg" / "pb_sim_pnp_shelf.yaml"
        with open(config_file_path, "r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        self.bc = configure_bullet(config)
        config["robot_params"]["franka_panda"]["base_pos"] = robot_base_pos_quat[0]
        config["robot_params"]["franka_panda"]["base_orn"] = self.bc.getEulerFromQuaternion(robot_base_pos_quat[1])
        env = SimPnPEnv(self.bc, env_type)
        self.robot = FrankaPanda(self.bc, config)
        self.bc.resetDebugVisualizerCamera(
            cameraDistance       = 1.2,
            cameraYaw            = -67.80,
            cameraPitch          = -24.80, 
            cameraTargetPosition = [0.38, 0.07, -0.13] )
        self.objs_uid = []

        self.panda_gripper = PandaGripperVisual(self.bc, Path("imm_pb_util")/"urdf"/"panda_gripper"/"panda_gripper_visual.urdf")
        self.panda_gripper.reset_finger(0.04)
        self.panda_gripper.freeze()
        self.panda_gripper.transparent(0.7)
        self.panda_gripper.set_gripper_pose(np.array([0,0,5.0]), np.array([0,0,0,1]))

    def register_latent_obj(self, env_obj, models, jkey):
        # load mesh
        for i in range(env_obj.outer_shape[0]):
            obj_ = jax.tree_util.tree_map(lambda x: x[i], env_obj)
            if np.all(np.abs(obj_.pos) < 1.0):
                mesh_pred = create_mesh_from_latent(jkey, models, obj_.translate(-obj_.pos), density=128, visualize=False)
                o3d.io.write_triangle_mesh(f'tmp/latent_mesh{i}.obj', mesh_pred)
                obj_id = self.bc.createMultiBody(baseMass=0.0, baseVisualShapeIndex=pb.createVisualShape(pb.GEOM_MESH, fileName=f'tmp/latent_mesh{i}.obj'))
                self.objs_uid.append(obj_id)
                self.bc.resetBasePositionAndOrientation(obj_id, np.array(obj_.pos), [0,0,0,1])
                print(f'{i} loaded')

    def visualize_similarity(self, env_obj: cxutil.LatentObjects, clip_sim: jnp.ndarray, normalize: bool = False):
        """ Visualize color to objects in pybullet.

        Args:
            env_obj: Used to filter out valid objects. Outer shape = [#NO]
            clip_sim: Same shape as env_obj. Shape = [#NO]
            normalize: Normalize to [0, 1] when true
        """
        if normalize:
            clip_sim = np.array(clip_sim)
            valid_mask1 = np.all(np.abs(env_obj.pos) < 1.0, axis=-1)
            valid_mask2 = clip_sim>0
            valid_mask = np.logical_and(valid_mask1, valid_mask2)
            if np.sum(valid_mask) > 1:
                min_val = clip_sim[valid_mask].min()
                max_val = clip_sim[valid_mask].max()
                clip_sim = (clip_sim - min_val) / (max_val - min_val)
            else:
                clip_sim[valid_mask] = 1.

        cmap = plt.get_cmap("viridis")
        # Iterate objs
        uid_counter = 0
        for i in range(env_obj.outer_shape[0]):
            obj_ = jax.tree_util.tree_map(lambda x: x[i], env_obj)
            # Only apply color to valid objects
            if np.all(np.abs(obj_.pos) < 1.0):
                sim = clip_sim[i]
                color = cmap(sim)
                obj_uid = self.objs_uid[uid_counter]
                self.bc.changeVisualShape(obj_uid, -1, rgbaColor=color)
                uid_counter += 1

    def send_latent_obj_faraway(self, env_obj:  cxutil.LatentObjects, idx: int):
        # Iterate objs
        uid_counter = 0
        for i in range(env_obj.outer_shape[0]):
            obj_ = jax.tree_util.tree_map(lambda x: x[i], env_obj)
            if np.all(np.abs(obj_.pos) < 1.0):
                if idx == i:
                    obj_uid = self.objs_uid[uid_counter]
                    self.bc.resetBasePositionAndOrientation(obj_uid, [0,0,5], [0,0,0,1])
                    return
                uid_counter += 1


    def simulate_traj(self, traj, target_idx=None, pos_quat_oe=None):
        
        if pos_quat_oe is not None and pos_quat_oe[0].ndim == 2:
            pos_quat_oe = (np.array(pos_quat_oe[0].squeeze(0)), np.array(pos_quat_oe[1]))

        joint_pos_start =  traj[0]
        # Execute
        self.robot.reset(joint_pos_start)
        print("traj simulation start")
        for q in traj:
            self.robot.reset(q)
            time.sleep(0.02)
            if target_idx is not None:
                obj_uid = self.objs_uid[target_idx]
                pos_quat_ee = self.robot.get_endeffector_pose()
                pos_quat_ee = (pos_quat_ee[0], (sciR.from_quat(pos_quat_ee[1]) * sciR.from_euler('z', np.pi/4)).as_quat())
                pos_quat_eo = pb.invertTransform(pos_quat_oe[0], pos_quat_oe[1])
                pos_quat_go = pb.multiplyTransforms(*pos_quat_ee, *pos_quat_eo)
                pb.resetBasePositionAndOrientation(obj_uid, *pos_quat_go)
        print("traj simulation end")


    def clean_objs(self):
        for uid in self.objs_uid:
            self.bc.removeBody(uid)
        self.objs_uid = []

    def visualize_gripper_pose(self, pos, quat):
        self.panda_gripper.set_gripper_pose(pos, quat)


if __name__ == '__main__':
    pass

    # # data_dir = 'experiment/exp_data/01242024-190819'
    # data_dir = 'experiment/exp_data/01242024-180446'
    # # data_dir = 'logs_realexp_rrt/01302024-155258'

    # with open(os.path.join(data_dir,'traj_data.pkl'), 'rb') as f:
    #     loaded_data = pickle.load(f)

    # # Manually create local variables for each key in the dictionary
    # color = loaded_data['color']
    # depth = loaded_data['depth']
    # pcds = loaded_data['pcds']
    # colors = loaded_data['colors']
    # cam_pq = loaded_data['cam_pq']
    # robot_base_pos = loaded_data['robot_base_pos']
    # robot_base_quat = loaded_data['robot_base_quat']
    # intrinsic = loaded_data['intrinsic']
    # original_img_size = (color.shape[1], color.shape[2])


    # jkey = jax.random.PRNGKey(0)
    
    # langauge_guided_pnp = LanguageGuidedPnP(robot_base_pos, robot_base_quat)
    # models = langauge_guided_pnp.models

    # build = ioutil.BuildMetadata.from_str("32_64_1_v4")

    # pixel_size = models.pixel_size

    # # simulation env setting
    # bc = BulletClient(connection_mode=pb.GUI)
    # config_file_path = Path(__file__).parent.resolve() / "pb_examples/pb_cfg" / "pb_real_eval_debug.yaml"
    # with open(config_file_path, "r") as f:
    #     config = yaml.load(f, Loader=yaml.FullLoader)
    # config["robot_params"]["franka_panda"]["base_pos"] = langauge_guided_pnp.robot_base_pos_quat[0]
    # config["robot_params"]["franka_panda"]["base_orn"] = bc.getEulerFromQuaternion(langauge_guided_pnp.robot_base_pos_quat[1])
    # robot = FrankaPanda(bc, config)
    # manip = FrankaManipulation(bc, robot, config)
    # env = SimpleEnv(bc, config, False)

    # # estimation
    # obj_pred_select, language_aligned_obj_feat, clip_aux_info = langauge_guided_pnp.estimation(color, cam_pq, intrinsic, jkey)

    # category_per_obj = langauge_guided_pnp.category_overlay(language_aligned_obj_feat)

    # # load mesh
    # print('create mesh and load obj in pybullet')
    # obj_idx_uid = {}
    # for i in range(obj_pred_select.outer_shape[0]):
    #     obj_ = jax.tree_util.tree_map(lambda x: x[i], obj_pred_select)
    #     if np.all(np.abs(obj_.pos) < 2.0):
    #         print(f'category: {category_per_obj[i]}')
    #         mesh_pred = create_mesh_from_latent(jkey, models, obj_.translate(-obj_.pos), density=128, visualize=False)
    #         o3d.io.write_triangle_mesh(f'tmp/latent_mesh{i}.obj', mesh_pred)
    #         obj_id = bc.createMultiBody(baseMass=0.0, baseVisualShapeIndex=pb.createVisualShape(pb.GEOM_MESH, fileName=f'tmp/latent_mesh{i}.obj'))
    #         bc.resetBasePositionAndOrientation(obj_id, np.array(obj_.pos), [0,0,0,1])
    #         obj_idx_uid[i] = obj_id
    
    # def reset_obj_pos():
    #     for i in range(obj_pred_select.outer_shape[0]):
    #         obj_ = jax.tree_util.tree_map(lambda x: x[i], obj_pred_select)
    #         if np.all(np.abs(obj_.pos) < 2.0):
    #             bc.resetBasePositionAndOrientation(obj_idx_uid[i], np.array(obj_.pos), [0,0,0,1])

    # _, jkey = jax.random.split(jkey)
    # for target_obj_text in ['a black bowl', 'a pink cup', 'an object that can hold soup', 'a black mug']:
    #     reset_obj_pos()

    #     print(f'start execution with target obj: {target_obj_text}')

    #     planning_output =\
    #           langauge_guided_pnp.fetch_target_obj(target_obj_text, obj_pred_select, 
    #                                                 language_aligned_obj_feat, RRT_INITIAL_Q, RRT_INITIAL_Q, jkey, visualize=False)
    #     _, jkey = jax.random.split(jkey)
    #     print('finish motion calculations')
    #     with open(os.path.join(langauge_guided_pnp.logs_dir,f'{target_obj_text}_save_data.pkl'), 'wb') as f:
    #             pickle.dump(planning_output, f)

    #     # load from saved data
    #     # load_plan_data_dir = 'logs_pnp/04062024-230007'
    #     # with open(os.path.join(load_plan_data_dir,f'{target_obj_text}_save_data.pkl'), 'rb') as f:
    #     #         planning_output = pickle.load(f)
    #     # print('load planning results from saved data')

    #     (traj1, grasp_traj, traj2), (target_obj_idx, pos_quat_eo), (grasp_aux_info, motion_planning_aux_info1, motion_planning_aux_info2) = planning_output

    #     robot.reset(traj1[0])

    #     robot.transparent()
    #     print('planning visualization')
    #     print('grasp visualization')
    #     best_joint_pos, approaching_joint_pos = grasp_aux_info[1], grasp_aux_info[2]
    #     for q in best_joint_pos:
    #         robot.reset(q)
    #         time.sleep(0.04)

    #     robot.reset(traj1[0])
    #     time.sleep(0.5)
    #     print('planning visualization')
    #     for q in motion_planning_aux_info1[0][:motion_planning_aux_info1[1]]:
    #         robot.reset(q)
    #         time.sleep(0.005)

    #     for q in motion_planning_aux_info2[0][:motion_planning_aux_info2[1]]:
    #         robot.reset(q)
    #         time.sleep(0.005)

    #     # Execute
    #     print('motion solution')
    #     robot.transparent(1.0)
    #     robot.reset(traj1[0])
    #     def follow_traj(traj_, obj_uid=None, pos_quat_eo=None):
    #         for q in traj_:
    #             robot.reset(q)
    #             if obj_uid is not None:
    #                 pos_quat_ee = robot.get_endeffector_pose()
    #                 pos_quat_ee = (pos_quat_ee[0], (sciR.from_quat(pos_quat_ee[1]) * sciR.from_euler('z', np.pi/4)).as_quat())
    #                 pos_quat_go = pb.multiplyTransforms(*pos_quat_ee, *pos_quat_eo)
    #                 pb.resetBasePositionAndOrientation(obj_uid, *pos_quat_go)
    #             time.sleep(0.05)
        
    #     robot.reset_finger(robot.finger_travel/2.0)
    #     follow_traj(traj1)
    #     follow_traj(grasp_traj)
    #     robot.reset_finger(0)
    #     follow_traj(traj2, obj_idx_uid[target_obj_idx], pos_quat_eo)
    #     robot.reset_finger(robot.finger_travel/2.0)
    

    # print(1)


        
