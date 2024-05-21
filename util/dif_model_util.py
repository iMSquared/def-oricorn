import flax.linen as nn
import typing
import jax.numpy as jnp
import numpy as np
import jax
import einops
import importlib
from flax.core.frozen_dict import FrozenDict
from flax import struct
import pickle
import time
import os, sys
from dataclasses import replace

BASEDIR = os.path.dirname(os.path.dirname(__file__))
if BASEDIR not in sys.path:
    sys.path.insert(0, BASEDIR)

import util.cvx_util as cxutil
import util.ev_util.ev_util as eutil
import util.ev_util.ev_layers as evl
import util.ev_util.rotm_util as rmutil
import util.transform_util as tutil
import util.camera_util as cutil
import util.diffusion_util as dfutil
import util.structs as structs
import data_generation.scene_generation as sg



class ImageFeature(nn.Module):
    base_dim:int=8
    depth:int=1

    @nn.compact
    def __call__(self, x, train=False):

        def cnn(base_dim, filter, depth, x_):
            for _ in range(depth):
                x_ = nn.Conv(base_dim, (filter,filter))(x_)
                x_ = nn.relu(x_)
            return x_

        if x.dtype in [jnp.uint8, jnp.int16, jnp.int32]:
            x = x.astype(jnp.float32)/255.

        # down
        c_list = []
        for _ in range(2):
            x = nn.Conv(self.base_dim, (5,5))(x)
            x = nn.relu(x)
        x = cnn(2*self.base_dim, 3, self.depth, x)
        c_list.append(x)
        x = nn.Conv(2*self.base_dim, (3,3), strides=(2,2))(x)
        x = nn.relu(x)
        x = cnn(4*self.base_dim, 3, self.depth, x)
        c_list.append(x)
        x = nn.Conv(4*self.base_dim, (3,3), strides=(2,2))(x)
        x = nn.relu(x)
        x = cnn(8*self.base_dim, 3, self.depth, x)
        c_list.append(x)
        x = nn.Conv(8*self.base_dim, (3,3), strides=(2,2))(x)
        x = nn.relu(x)

        x = nn.Conv(8*self.base_dim, (3,3), kernel_dilation=(2,2))(x)
        x = nn.relu(x)
        x = nn.Conv(8*self.base_dim, (5,5), kernel_dilation=(2,2))(x)
        x = nn.relu(x)
        x = cnn(8*self.base_dim, 3, self.depth, x)

        def repeat_ft(x, r, ft_dim):
            x = nn.Dense(ft_dim)(x)
            x = nn.relu(x)
            x = einops.repeat(x, '... i j k -> ... (i r1) (j r2) k', r1=r, r2=r)
            return x

        # up
        c_list = list(reversed(c_list))
        # p_list = [einops.repeat(x, '... i j k -> ... (i r1) (j r2) k', r1=8, r2=8)]
        p_list = [repeat_ft(x, 8, 4*self.base_dim)]
        x = nn.ConvTranspose(8*self.base_dim, (3,3), strides=(2,2))(x)
        x = nn.relu(x) + c_list[0]
        # p_list.append(einops.repeat(x, '... i j k -> ... (i r1) (j r2) k', r1=4, r2=4))
        p_list.append(repeat_ft(x, 4, 4*self.base_dim))
        x = nn.ConvTranspose(4*self.base_dim, (3,3), strides=(2,2))(x)
        x = nn.relu(x) + c_list[1]
        # p_list.append(einops.repeat(x, '... i j k -> ... (i r1) (j r2) k', r1=2, r2=2))
        p_list.append(repeat_ft(x, 2, 4*self.base_dim))
        x = nn.ConvTranspose(2*self.base_dim, (3,3), strides=(2,2))(x)
        x = nn.relu(x) + c_list[2]
        p_list.append(repeat_ft(x, 1, 4*self.base_dim))

        x = jnp.concatenate(p_list, axis=-1)

        return x

class ImageFeatureEntire(nn.Module):
    args:typing.NamedTuple
    base_dim:int=8
    depth:int=1
    @nn.compact
    def __call__(self, imgs, cam_posquat, cam_intrinsic, train=False):

        img_feat_structs = structs.ImgFeatures(intrinsic=cam_intrinsic, cam_posquat=cam_posquat, img_feat=ImageFeature(self.base_dim, self.depth)(imgs, train=train))
        # if self.args.dif_model_version in [1,4]:
        if self.args.spatial_pe:
            img_feat_structs = SpatialPE()(img_feat_structs)
        
            if self.args.dif_model_version == 4:
                # Add positional embedding
                img_base_dim = img_feat_structs.img_feat.shape[-1]
                img_feat_flat = img_feat_structs.img_feat + img_feat_structs.spatial_PE                
                if self.args.parq_resize:
                    # Downsize image feature!
                    img_size = img_feat_flat.shape[-3:-1]
                    img_resize_ratio = self.args.parq_resize_ratio
                    resize_output_shape = (*img_feat_flat.shape[:-3], int(img_size[0]*img_resize_ratio), int(img_size[1]*img_resize_ratio), img_feat_flat.shape[-1])
                    img_feat_flat = jax.image.resize(img_feat_flat, resize_output_shape , method='linear')
                    # img_feat_flat = nn.relu(nn.Conv(img_base_dim, (3,3), strides=(2,2))(img_feat_flat))
                    # img_feat_flat = nn.relu(nn.Conv(img_base_dim, (3,3), strides=(2,2))(img_feat_flat))
                    # img_feat_flat = nn.relu(nn.Conv(img_base_dim, (3,3), strides=(2,2))(img_feat_flat))

                # Flatten
                img_feat_flat = einops.rearrange(img_feat_flat, '... v i j f -> ... (v i j) f')
                return img_feat_structs.replace(img_feat_patch=img_feat_flat)
            else:
                return img_feat_structs
        else:
            return img_feat_structs

class SpatialPE(nn.Module):
    @nn.compact
    def __call__(self, img_feat: structs.ImgFeatures) -> structs.ImgFeatures:
        
        pixel_size = img_feat.img_feat.shape[-3:-1]
        img_feat_dim = img_feat.img_feat.shape[-1]
        cam_posquat = img_feat.cam_posquat # (... NV 7)
        cam_intrinsic = img_feat.intrinsic # (... NV 6)
        near = 0.050
        far = 1.5
        nsamples = 40

        ray_start_pnts, ray_end_pnts, ray_dir = cutil.pixel_ray(pixel_size, cam_posquat[...,:3], cam_posquat[...,3:], cam_intrinsic, near=near, far=far)
        ray_grid = ray_start_pnts[...,None,:] + ray_dir[...,None,:]*jnp.linspace(near, far, nsamples)[...,None]
        ray_grid = einops.rearrange(ray_grid, '... i j -> ... (i j)')

        ray_grid = nn.Dense(img_feat_dim)(ray_grid)
        ray_grid = nn.relu(ray_grid)
        ray_grid = nn.Dense(img_feat_dim)(ray_grid)

        # skip = ray_grid
        # ray_grid = nn.relu(ray_grid)
        # ray_grid = nn.Conv(img_feat_dim, (3,3))(ray_grid)
        # ray_grid = nn.relu(ray_grid)
        # ray_grid = nn.Conv(img_feat_dim, (3,3))(ray_grid)
        # ray_grid += skip
        
        return img_feat.replace(spatial_PE=ray_grid)


def extract_pixel_features(center, img_fts_structs:structs.ImgFeatures):
    '''
    center : (... NO 3)
    img_fts : (... NC NI NJ NF)
    cam_info :
        cam_pq : (... NC 7)
        intrinsic : (... NC 6)
    '''
    # # project points to img space
    # center = jax.lax.stop_gradient(center)

    # cam_posquat, intrinsic = cam_info
    cam_posquat = img_fts_structs.cam_posquat
    intrinsic = img_fts_structs.intrinsic
    img_fts = img_fts_structs.img_feat
    if img_fts_structs.spatial_PE is not None:
        img_fts = img_fts + img_fts_structs.spatial_PE
    if img_fts_structs.img_state is not None:
        img_fts = jnp.c_[jnp.broadcast_to(img_fts, jnp.broadcast_shapes(img_fts.shape, img_fts_structs.img_state[...,:1].shape)), img_fts_structs.img_state]
    intrinsic_ext = intrinsic[...,None,:,:] # (... 1 NC 7)
    input_img_size = intrinsic_ext[...,:2]
    cam_posquat_ext = cam_posquat[...,None,:,:]
    cam_pos_quat_ext = (cam_posquat_ext[...,:3], cam_posquat_ext[...,3:]) # (... NR, ...)
    cam_Rm_ext = einops.rearrange(tutil.q2R(cam_pos_quat_ext[1]), '... i j -> ... (i j)')

    img_ft_size = jnp.array((img_fts.shape[-3], img_fts.shape[-2]))
    img_fts_flat = einops.rearrange(img_fts, '... i j k -> ... (i j) k') # (... NC NIJ NF)
    
    # batch version
    intrinsic_ext_, cam_posquat_, input_img_size_, img_fts_flat_ = \
        jax.tree_map(lambda x: x[...,None,:], (intrinsic_ext, cam_pos_quat_ext, input_img_size, img_fts_flat))
    # img_fts_flat_ : (... NC NIJ 1 NF)
    # q_pnts = local_points_ + qp_dirs # (... NO NG 3)
    q_pnts = center[...,None,:,:]
    px_coord_ctn, out_pnts_indicator = cutil.global_pnts_to_pixel(intrinsic_ext_, cam_posquat_, q_pnts) # (... NO NC NG 2)
    px_coord_ctn = px_coord_ctn/input_img_size_ * img_ft_size
    px_coord = jnp.floor(px_coord_ctn).astype(jnp.int32)

    # interpolation
    # px_coord_residual = (px_coord_ctn - px_coord) # (... NO NC NG 2)
    def extract_img_fts(px_coord:jnp.ndarray):
        px_coord = px_coord.clip(0, jnp.array(img_ft_size)-1)
        px_flat_idx = px_coord[...,1] + px_coord[...,0] * img_ft_size[...,1] # (... NO NC)
        # selected_img_fts = jnp.take_along_axis(img_fts_flat_, einops.rearrange(px_flat_idx, '... i j k -> ... j i k')[...,None], axis=-3) # (... NC NO NG NF)
        selected_img_fts = jnp.take_along_axis(img_fts_flat_.squeeze(-2), einops.rearrange(px_flat_idx, '... i j k -> ... j (i k)')[...,None], axis=-2) # (... NC NO*NG NF)
        selected_img_fts = einops.rearrange(selected_img_fts, '... (r i) j -> ... r i j', r=px_coord.shape[-4]) # (... NC NO NG NF)
        img_fts = einops.rearrange(selected_img_fts, '... i j p k -> ... j i p k') # (... NO NC NG NF)
        return img_fts
    
    img_fts_list = []
    residuals = []
    px_coord_ctn_offset = (px_coord_ctn -0.5).clip(0, jnp.array(img_ft_size)-1)
    for sft in [np.array([0,0]), np.array([0,1]), np.array([1,0]), np.array([1,1])]:
        img_fts_list.append(extract_img_fts(px_coord+sft))
        resd_ = jnp.abs(px_coord_ctn_offset - (px_coord+(1-sft))) + 1e-2
        resd_ = resd_[...,0:1] * resd_[...,1:2]
        residuals.append(resd_)
    weights = jnp.stack(residuals, axis=-1)
    weights = weights/jnp.sum(weights, axis=-1, keepdims=True) # (NO NC NG 1 4)
    img_fts = jnp.sum(jnp.stack(img_fts_list, axis=-1) * weights, -1) # (NO NC NG NF)
    img_fts = einops.rearrange(img_fts, '... i j p k -> ... i p j k') # (... NO NG NC NF)
    
    intrinsic_cond = jnp.concatenate([intrinsic_ext[...,2:3], intrinsic_ext[...,2:3]], axis=-1)
    cam_fts = jnp.concatenate([cam_pos_quat_ext[0], cam_Rm_ext, intrinsic_cond/intrinsic_ext[...,1:2]], axis=-1) # (... 1 NC NF)

    return img_fts, cam_fts


class AdaLayerNorm(nn.Module):
    '''
    FiLM (FiLM: Visual Reasoning with a General Conditioning Layer)
    AdaGN in (https://proceedings.neurips.cc/paper_files/paper/2021/hash/49ad23d1ec9fa4bd8d77d02681df5cfa-Abstract.html)
    DiT
    '''

    @nn.compact
    def __call__(self, x, context):
        emb = nn.Dense(2*x.shape[-1])(context)
        scale, shift = jnp.split(emb, 2, -1)
        x = nn.LayerNorm()(x) * (1 + scale) + shift
        return x



# Resnet Blocks
class ResnetBlockFC(nn.Module):
    args:typing.NamedTuple
    size_h:int
    size_out:int

    @nn.compact
    def __call__(self, x):
        size_in = x.shape[-1]
        net = nn.relu(x)
        net = nn.Dense(self.size_h)(net)
        dx = nn.relu(net)
        dx = nn.Dense(self.size_out)(dx)
        if size_in == self.size_out:
            x_s = x
        else:
            x_s = nn.Dense(self.size_out)(x)
        return x_s + dx

class Aggregator(nn.Module):
    args:typing.NamedTuple
    depth:int=2
    axis:int=-2
    pooled_out:bool=True
    pooling_func:str=jnp.max
    final_activation:bool=True

    @nn.compact
    def __call__(self, x):
        if x.shape[-1] == self.args.base_dim:
            skip = x
        else:
            skip = nn.Dense(self.args.base_dim)(x)
        
        for k in range(self.depth):
            x = nn.relu(nn.Dense(self.args.base_dim)(x)) + skip
            if x.shape[self.axis] != 1:
                x_pooled = self.pooling_func(x, axis=self.axis, keepdims=True)
                x = jnp.c_[x, einops.repeat(x_pooled, '... r i -> ... (k r) i', k=x.shape[self.axis])]
            x = nn.Dense(self.args.base_dim)(x)
            if k<self.depth-1 or self.final_activation:
                x = nn.relu(x)
        if self.pooled_out:
            return self.pooling_func(x, axis=self.axis)
        else:
            return x

class DeformableCrossAttention(nn.Module):
    args:typing.Sequence
    dropout:float

    @nn.compact
    def __call__(self, z, img_features, train=False):
        '''
        z: (NB NO 1 ND)
        img_features (NB NO NP NC)
        '''
        assert z.shape[-2] == 1
        if img_features.shape[-2] == 1:
            features = nn.Dense(z.shape[-1])(img_features)
            return z+nn.Dropout(self.dropout)(features, deterministic=not train)
        attention_weights = nn.DenseGeneral(axis=-1, features=(self.args.mixing_head, 1))(img_features).squeeze(-1)
        attention_weights = nn.softmax(attention_weights, axis=-1)
        features = nn.DenseGeneral(axis=-1, features=(self.args.mixing_head, self.args.base_dim//2))(img_features)
        features = jnp.einsum('...ij,...ijk->...jk', attention_weights, features) # (NB NO NH NC)
        features = einops.rearrange(features, '... i j -> ... (i j)')
        features = nn.Dense(z.shape[-1])(features)
        return z + nn.Dropout(self.dropout)(features[...,None,:], deterministic=not train)


class TransformerDecoder(nn.Module):
    args:typing.Sequence
    dropout:float
    local_only:bool=False
    
    @nn.compact
    def __call__(self, z, context, img_feat_flat=None, train=False):
        '''
        if turn on local_only, only one attention layer is appied
            Deformable DETR - dif_model_version 6 - Deformable Cross Attention between z (obj queries) and img_feat_flat (extracted_img_fts)
            Voxle-CA,pixelCA - dif_model_version 4,5 - Cross Attention between z (obj queries) and img_feat_flat (pixel_flat_fts or voxel_flat_fts)
            SPS - dif_model_version 1 - self Attention between z with context (extracted img fts) conditioning
        '''
        ncd = z.shape[-2]
        if self.local_only and self.args.dif_model_version in [6]:
            # deformable Attention
            z = DeformableCrossAttention(self.args, self.dropout)(z, img_feat_flat, train=train)
            z = AdaLayerNorm()(z, context)

        elif self.local_only and self.args.dif_model_version in [4,5]:
            # local mixing only for pixel and voxel
            z_gb = einops.rearrange(z, '... s c f -> ... (s c) f')
            z2_gb = nn.MultiHeadDotProductAttention(num_heads=self.args.mixing_head, qkv_features=self.args.mixing_head//2*self.args.base_dim, 
                                    out_features=self.args.base_dim, dropout_rate=0.0)(z_gb, img_feat_flat, deterministic=True)
            z2 = einops.rearrange(z2_gb, '... (r i) k -> ... r i k', i=ncd)
            z = z + nn.Dropout(self.dropout)(z2, deterministic=not train)
            z = AdaLayerNorm()(z, context)
        else:
            # first mixing
            if ncd > 1:
                z2 = nn.SelfAttention(num_heads=self.args.mixing_head, qkv_features=self.args.mixing_head//2*self.args.base_dim, 
                                        out_features=self.args.base_dim, dropout_rate=0.0)(z, deterministic=True)
            else:
                z_gb = einops.rearrange(z, '... s c f -> ... (s c) f')
                z2_gb = nn.SelfAttention(num_heads=self.args.mixing_head, qkv_features=self.args.mixing_head//2*self.args.base_dim, 
                                        out_features=self.args.base_dim, dropout_rate=0.0)(z_gb, deterministic=True)
                z2 = einops.rearrange(z2_gb, '... (r i) k -> ... r i k', i=ncd)
            z = z + nn.Dropout(self.dropout)(z2, deterministic=not train)
            z = AdaLayerNorm()(z, context)

        # second mixing -> conditioning (cross attention)
        if not self.local_only:
            if self.args.dif_model_version in [6]:
                assert img_feat_flat is not None
                z = DeformableCrossAttention(self.args, self.dropout)(z, img_feat_flat, train=train)
                z = AdaLayerNorm()(z, context)
            else:
                z_gb = einops.rearrange(z, '... s c f -> ... (s c) f')
                if self.args.dif_model_version in [3,4,5]:
                    assert img_feat_flat is not None
                    z2_gb = nn.MultiHeadDotProductAttention(num_heads=self.args.mixing_head, qkv_features=self.args.mixing_head//2*self.args.base_dim, 
                                            out_features=self.args.base_dim, dropout_rate=0.0)(z_gb, img_feat_flat, deterministic=True)
                elif self.args.dif_model_version in [1]:
                    assert img_feat_flat is None
                    z2_gb = nn.SelfAttention(num_heads=self.args.mixing_head, qkv_features=self.args.mixing_head//2*self.args.base_dim, 
                                            out_features=self.args.base_dim, dropout_rate=0.0)(z_gb, deterministic=True)
                z2 = einops.rearrange(z2_gb, '... (r i) k -> ... r i k', i=ncd)
                z = z + nn.Dropout(self.dropout)(z2, deterministic=not train)
                z = AdaLayerNorm()(z, context)

        # final linear
        z2 = nn.Dense(self.args.base_dim)(nn.Dropout(self.dropout)(nn.relu(nn.Dense(self.args.base_dim)(z)), deterministic=not train))
        z = z + nn.Dropout(self.dropout)(z2, deterministic=not train)
        z = AdaLayerNorm()(z, context)

        return z


class DenoisingModelV4(nn.Module):
    args:typing.NamedTuple
    rot_args:typing.NamedTuple
    rot_configs:typing.Sequence

    @nn.compact
    def __call__(self, objects_ptcl:cxutil.LatentObjects, con_feat:structs.ImgFeatures, time, batch_condition_mask=None, confidence_out=False, train=False):
        '''
        objects_ptcl : (nb, ns, ...)
        con_feat : (nb, nc, nf)
        time : (nb, ) # if EDM it is sigma
        '''
        if time.ndim == 0:
            time = time[None]

        extended = False
        if len(objects_ptcl.outer_shape) == 1:
            extended = True
            objects_ptcl, con_feat = jax.tree_map(lambda x:x[None], (objects_ptcl, con_feat))

        dropout=0.1
        nb = objects_ptcl.outer_shape[0]
        no = objects_ptcl.outer_shape[-1]
        nd = objects_ptcl.nd
        ncd = objects_ptcl.dc_rel_centers.shape[-2]

        time, (c_skip, c_out, c_in) = dfutil.calculate_cs(time, dfutil.EDMP, self.args)

        embeddings = nn.Dense(self.args.base_dim)(time)
        embeddings = jnp.sin(embeddings)
        embeddings = nn.Dense(self.args.base_dim)(embeddings)
        embeddings = embeddings[...,None,None,:]
        if self.args.one_step_test == 3:
            embeddings = jnp.zeros_like(embeddings)

        # extract features
        c_in_ = c_in
        for _ in range(objects_ptcl.dc_centers_tf.ndim - c_in.ndim):
            c_in_ = c_in_[...,None]
        
        if not self.args.use_p or self.args.dif_model_version in [6]:
            ncd = 1

        def feature_extractor(c_, p_, con_feat_, stop_gradient=False, z_prior=None, img_feat_flat_prior=None):

            pos_emb = nn.Dense(self.args.base_dim//2)(c_)
            pos_emb = jnp.sin(pos_emb)
            pos_emb = nn.Dense(self.args.base_dim//2)(pos_emb)
            
            if self.args.use_p:
                cen_emb = nn.Dense(self.args.base_dim//2)(p_)
                cen_emb = jnp.sin(cen_emb)
                cen_emb = nn.Dense(self.args.base_dim//2)(cen_emb)
            else:
                cen_emb = None

            # feature extraction
            if self.args.use_p:
                if stop_gradient:
                    extracted_img_fts, cam_fts = extract_pixel_features(jax.lax.stop_gradient(p_), con_feat_)
                else:
                    extracted_img_fts, cam_fts = extract_pixel_features(p_, con_feat_)
            else:
                if stop_gradient:
                    extracted_img_fts, cam_fts = extract_pixel_features(jax.lax.stop_gradient(c_[...,None,:]), con_feat_)
                else:
                    extracted_img_fts, cam_fts = extract_pixel_features(c_[...,None,:], con_feat_)
            extracted_aggregated_img_fts = Aggregator(self.args)(extracted_img_fts) # (B NO ND d)

            if batch_condition_mask is not None:
                extracted_aggregated_img_fts = jnp.where(batch_condition_mask[...,None,None,None], extracted_aggregated_img_fts, 0)

                
            if self.args.dif_model_version in [6]:
                # deformable DETR
                assert self.args.use_p
                context = embeddings
                img_feat_flat = jnp.c_[extracted_aggregated_img_fts, cen_emb]

                # generate object queries
                if z_prior is None:
                    z = jnp.c_[einops.repeat(pos_emb, '... d -> ... r2 d', r2=ncd), 
                        einops.repeat(c_in[...,None,None]*objects_ptcl.z_flat, '... i d -> ... (r2 i) d', r2=ncd)] # (B NO ND d)
                else:
                    z = jnp.c_[einops.repeat(pos_emb, '... d -> ... r2 d', r2=ncd), z_prior] # (B NO ND d)
                z = AdaLayerNorm()(nn.Dense(self.args.base_dim)(z), context)
                
            elif self.args.dif_model_version in [1]:
                if self.args.use_p:
                    # ours
                    if self.args.one_step_test == 3:
                        context = jnp.c_[extracted_aggregated_img_fts, cen_emb]
                    else:
                        context = jnp.c_[extracted_aggregated_img_fts, cen_emb, 
                                        jnp.broadcast_to(embeddings, jnp.broadcast_shapes(extracted_aggregated_img_fts.shape, embeddings.shape))]
                else:
                    # DETR3D
                    if self.args.one_step_test == 3:
                        context = jnp.c_[extracted_aggregated_img_fts, pos_emb[...,None,:]]
                    else:
                        context = jnp.c_[extracted_aggregated_img_fts, pos_emb[...,None,:], 
                                        jnp.broadcast_to(embeddings, jnp.broadcast_shapes(extracted_aggregated_img_fts.shape, embeddings.shape))]
                img_feat_flat = None

                # generate object queries
                if z_prior is None:
                    z = jnp.c_[einops.repeat(pos_emb, '... d -> ... r2 d', r2=ncd), 
                            einops.repeat(c_in[...,None,None]*objects_ptcl.z_flat, '... i d -> ... (r2 i) d', r2=ncd)] # (B NO ND d)
                else:
                    z = jnp.c_[einops.repeat(pos_emb, '... d -> ... r2 d', r2=ncd), z_prior] # (B NO ND d)
                z = AdaLayerNorm()(nn.Dense(self.args.base_dim)(z), context)

            elif self.args.dif_model_version in [4]:
                # pixel-CA (PARQ)
                context = embeddings

                # bring img features from ImgFeatreEntire module
                img_feat_flat = con_feat.img_feat_patch

                # generate object queries
                if z_prior is None:
                    z = jnp.c_[einops.repeat(pos_emb, '... d -> ... r2 d', r2=ncd), 
                        einops.repeat(c_in[...,None,None]*objects_ptcl.z_flat, '... i d -> ... (r2 i) d', r2=ncd),
                        extracted_aggregated_img_fts] # (B NO ND d)
                else:
                    z = jnp.c_[einops.repeat(pos_emb, '... d -> ... r2 d', r2=ncd), z_prior, extracted_aggregated_img_fts] # (B NO ND d)
                if self.args.use_p:
                    z = jnp.c_[z, cen_emb]
                z = AdaLayerNorm()(nn.Dense(self.args.base_dim)(z), context)

            elif self.args.dif_model_version in [5]:
                # voxel-CA (RayTran)
                context = embeddings

                # generate object queries
                if z_prior is None:
                    z = jnp.c_[einops.repeat(pos_emb, '... d -> ... r2 d', r2=ncd), 
                        einops.repeat(c_in[...,None,None]*objects_ptcl.z_flat, '... i d -> ... (r2 i) d', r2=ncd),
                        extracted_aggregated_img_fts] # (B NO ND d)
                else:
                    z = jnp.c_[einops.repeat(pos_emb, '... d -> ... r2 d', r2=ncd), z_prior, extracted_aggregated_img_fts] # (B NO ND d)
                if self.args.use_p:
                    z = jnp.c_[z, cen_emb]
                z = AdaLayerNorm()(nn.Dense(self.args.base_dim)(z), context)

                if img_feat_flat_prior is None:
                    # generate grid features
                    grid_resolution = self.args.voxel_resolution
                    grid_size = [grid_resolution,grid_resolution,grid_resolution//3]
                    x_ = np.linspace(-0.45, 0.45, grid_size[0])
                    y_ = np.linspace(-0.45, 0.45, grid_size[1])
                    z_ = np.linspace(-0.1, 0.2, grid_size[2])
                    xv, yv, zv = np.meshgrid(x_, y_, z_)
                    grid = np.stack([xv, yv, zv]).astype(np.float32).reshape(3, -1).transpose() # (np, 3)
                    grid = jnp.array(grid)
                    grid = grid[None,None] # (1 1 NP 3)
                    voxel_fts_multiple_viewpoints, cam_fts = extract_pixel_features(grid, con_feat) # (NB 1 NP NC d)

                    if not self.args.spatial_pe:
                        for _ in range(2):
                            cam_fts = nn.Dense(voxel_fts_multiple_viewpoints.shape[-1]//2)(cam_fts)
                            cam_fts = jnp.sin(cam_fts)
                        cam_fts = cam_fts[...,None,:,:]
                        cam_fts = jnp.broadcast_to(cam_fts, jnp.broadcast_shapes(cam_fts.shape, voxel_fts_multiple_viewpoints[...,:1].shape))
                        voxel_fts_multiple_viewpoints = jnp.c_[voxel_fts_multiple_viewpoints, cam_fts] # (B NO ND NC d)

                    voxel_fts = Aggregator(self.args)(voxel_fts_multiple_viewpoints) # (B 1 NG d)
                    
                    voxel_fts = voxel_fts.reshape((voxel_fts.shape[0], *grid_size, voxel_fts.shape[-1])) # (NB NG1 NG2 NG3 d)
                    for _ in range(3):
                        voxel_fts = nn.Conv(self.args.base_dim, kernel_size=(3,3,3))(voxel_fts)
                        voxel_fts = nn.relu(voxel_fts)

                    img_feat_flat = voxel_fts.reshape((voxel_fts.shape[0], -1, voxel_fts.shape[-1])) # (NB NP d)
                else:
                    img_feat_flat = img_feat_flat_prior

            if batch_condition_mask is not None and img_feat_flat is not None:
                batch_condition_mask_ = batch_condition_mask
                for _ in range(img_feat_flat.ndim - batch_condition_mask.ndim):
                    batch_condition_mask_ = batch_condition_mask_[...,None]
                img_feat_flat = jnp.where(batch_condition_mask_, img_feat_flat, 0)
            
            return z, context, img_feat_flat, pos_emb, cen_emb
        
        def c_p_prediction_head(z, context, pos_emb, cen_emb, conf_out=False):
            # heads for the rest
            dz = z
            for i in range(2):
                dz = nn.Dense(self.args.base_dim)(dz)
                dz = AdaLayerNorm()(dz, context)
                dz = nn.relu(dz)
                dz = jnp.c_[dz, einops.repeat(jnp.mean(dz, axis=-2), '... c -> ... r c', r=ncd)]
            dz = nn.relu(nn.Dense(self.args.base_dim)(dz))
            dcenter = dz
            dpos=conf=jnp.mean(dz, axis=-2)

            # # pos branch
            dpos = jnp.c_[dpos, nn.Dense(dpos.shape[-1])(AdaLayerNorm()(pos_emb, embeddings.squeeze(-2)))]
            dpos = nn.Dense(3)(dpos)

            if self.args.add_c_skip:
                pos_scaled = c_skip[...,None]*objects_ptcl.pos + c_out[...,None]*dpos
            else:
                c_skip_ = nn.sigmoid(nn.Dense(1)(embeddings.squeeze(-2)))
                c_out_ = nn.sigmoid(nn.Dense(1)(embeddings.squeeze(-2)))
                pos_scaled = c_skip_*objects_ptcl.pos + c_out_*dpos

            # # conf branch
            if conf_out:
                conf = nn.Dense(1)(conf)
            else:
                conf = None

            if self.args.use_p:
                # center pred branch
                cen_emb_tmp = AdaLayerNorm()(cen_emb, embeddings)
                dcenter = jnp.c_[jnp.broadcast_to(dcenter, jnp.broadcast_shapes(cen_emb_tmp[...,:1].shape, dcenter.shape)), 
                                jnp.broadcast_to(cen_emb_tmp, jnp.broadcast_shapes(dcenter[...,:1].shape, cen_emb_tmp.shape))]
                dcenter = nn.Dense(self.args.base_dim)(dcenter)
                dcenter = nn.relu(dcenter)
                dcenter = nn.Dense(3)(dcenter)

                if self.args.add_c_skip:
                    center_scaled = c_skip[...,None,None]*objects_ptcl.dc_centers_tf + c_out[...,None,None]*dcenter
                else:
                    c_skip_ = nn.sigmoid(nn.Dense(1)(embeddings))
                    c_out_ = nn.sigmoid(nn.Dense(1)(embeddings))
                    center_scaled = c_skip_*objects_ptcl.dc_centers_tf + c_out_*dcenter
            else:
                center_scaled = None
            return pos_scaled, center_scaled, conf

        z, context, img_feat_flat, pos_emb_init, cen_emb_init = \
            feature_extractor(objects_ptcl.pos, objects_ptcl.dc_centers_tf, con_feat, stop_gradient=True, img_feat_flat_prior=None, z_prior=None)
        # apply transformer Decoders
        for first_mixing_itr in range(self.args.first_mixing_depth):
            z = TransformerDecoder(self.args, dropout)(z, context, img_feat_flat, train=train)

            if self.args.recurrent_in_first_mixing:
                # heads for the rest
                pos_scaled, center_scaled, conf = c_p_prediction_head(z, context,  pos_emb_init, cen_emb_init, conf_out=(first_mixing_itr == self.args.first_mixing_depth-1))
                z, context, img_feat_flat, _, _ = feature_extractor(pos_scaled, center_scaled, con_feat, stop_gradient=(first_mixing_itr == self.args.first_mixing_depth-1), 
                                                                    img_feat_flat_prior=img_feat_flat, z_prior=z)
        if not self.args.recurrent_in_first_mixing:
            pos_scaled, center_scaled, conf = c_p_prediction_head(z, context,  pos_emb_init, cen_emb_init, conf_out=True)
            if self.args.second_mixing_depth != 0:
                z, context, img_feat_flat, _, _ = feature_extractor(pos_scaled, center_scaled, con_feat, stop_gradient=True,
                                                                        img_feat_flat_prior=img_feat_flat, z_prior=z)

        z_sh = jnp.c_[einops.repeat(c_in[...,None,None]*objects_ptcl.z_flat, '... i j -> ... (r i) j', r=ncd), z]
        z_sh = nn.Dense(self.args.base_dim)(z_sh)
        z_sh = AdaLayerNorm()(z_sh, context)

        for _ in range(np.maximum(1, self.args.second_mixing_depth)):
            z_sh = TransformerDecoder(self.args, dropout, local_only=True)(z_sh, context, img_feat_flat, train=train)

        if self.args.dif_model_type in [1,3]:
            for _ in range(1):
                z_sh = ResnetBlockFC(self.args, self.args.base_dim, self.args.base_dim)(z_sh)
                z_sh = AdaLayerNorm()(z_sh, context)
                if ncd!=1:
                    pooled_global = einops.repeat(jnp.mean(z_sh, -2), '... d -> ... r d', r=ncd)
                    z_sh = jnp.concatenate([z_sh, pooled_global], -1)
            z_sh = ResnetBlockFC(self.args, self.args.base_dim, self.args.base_dim)(z_sh)
            z_sh = jnp.mean(z_sh, axis=-2, keepdims=True)
            if self.args.dif_model_type==1:
                z_sh = nn.Dense(objects_ptcl.z_flat.shape[-1])(z_sh)
                z_sh = einops.rearrange(z_sh, '... (r i) -> ... r i', i=objects_ptcl.nz)

        if self.args.dif_model_type in [2,3]:
            fer_base_dim = objects_ptcl.nz
            z_sh = nn.Dense(2*fer_base_dim*3)(z_sh)
            z_sh = nn.selu(z_sh)
            z_sh = einops.rearrange(z_sh, '... (r i) -> ... r i', r=3)
            z_sh = nn.Dense(2*fer_base_dim)(z_sh)
            z_sh = nn.selu(z_sh)

            # equivariant layers
            z_sh = evl.MakeHDFeature(self.rot_args, self.rot_configs)(z_sh) # (... ND NF NZ)
            z_sh = nn.Dense(fer_base_dim, use_bias=False)(z_sh)
            z_sh = evl.EVNNonLinearity(self.rot_args)(z_sh)
            if not self.args.one_step_test:
                z_sh = z_sh + nn.Dense(fer_base_dim, use_bias=False)(c_in[...,None,None,None]*objects_ptcl.z)
            if self.args.dif_model_type==2:
                context_dense = nn.Dense(fer_base_dim*fer_base_dim, kernel_init=nn.initializers.zeros, bias_init=nn.initializers.zeros)(context)
                context_dense = einops.rearrange(context_dense, '... (r i) -> ... r i', r=fer_base_dim)
            for i in range(1):
                z_sh = evl.EVNResnetBlockFC(self.rot_args, fer_base_dim, fer_base_dim)(z_sh)
                if self.args.dif_model_type==2:
                    z_sh = z_sh + jnp.einsum('...ij,...fj->...fi',context_dense,z_sh)
                    if ncd!=1:
                        pooled_global = einops.repeat(jnp.mean(z_sh, -3), '... f d -> ... r f d', r=ncd)
                        z_sh = jnp.concatenate([z_sh, pooled_global], -1)
            z_sh = evl.EVNResnetBlockFC(self.rot_args, fer_base_dim, fer_base_dim)(z_sh)
            z_sh = evl.EVNNonLinearity(self.rot_args)(z_sh)

            if nd==1 and ncd!=1:
                z_sh = jnp.mean(z_sh, axis=-3, keepdims=True)

        if self.args.add_c_skip:
            z_sh = c_skip[...,None,None,None]*objects_ptcl.z + c_out[...,None,None,None]*z_sh
        else:
            c_skipout = nn.sigmoid(nn.Dense(2)(embeddings))
            z_sh = c_skipout[...,None,0:1]*objects_ptcl.z + c_skipout[...,None,1:2]*z_sh
        if self.args.use_p:
            obj_pred0 = cxutil.LatentObjects().replace(dc_rel_centers=center_scaled-pos_scaled[...,None,:], pos=pos_scaled, z=z_sh)
        else:
            obj_pred0 = cxutil.LatentObjects().replace(dc_rel_centers=jax.lax.stop_gradient(einops.repeat(pos_scaled, '... i -> ... r i', r=32)),
                                                        pos=pos_scaled, z=z_sh)
        if extended:
            if confidence_out:
                return jax.tree_map(lambda x: x[0], (obj_pred0, conf))
            else:
                return jax.tree_map(lambda x: x[0], obj_pred0)
        else:
            if confidence_out:
                return obj_pred0, conf
            else:
                return obj_pred0


class ObservationModel(nn.Module):
    args:typing.NamedTuple

    @nn.compact
    def __call__(self, objects_ptcl:cxutil.LatentObjects, con_feat:structs.ImgFeatures, time, train=False):
        '''
        objects_ptcl : (nb, ns, ...)
        con_feat : (nb, nc, nf)
        time : (nb, ) # if EDM it is sigma
        '''
        if time.ndim == 0:
            time = time[None]

        extended = False
        if len(objects_ptcl.outer_shape) == 1:
            extended = True
            objects_ptcl, con_feat = jax.tree_map(lambda x:x[None], (objects_ptcl, con_feat))

        dropout=0.1
        nb = objects_ptcl.outer_shape[0]
        no = objects_ptcl.outer_shape[-1]
        nd = objects_ptcl.nd
        ncd = objects_ptcl.dc_rel_centers.shape[-2]

        time, _ = dfutil.calculate_cs(time, dfutil.EDMP, self.args)

        embeddings = nn.Dense(self.args.base_dim)(time)
        embeddings = jnp.sin(embeddings)
        embeddings = nn.Dense(self.args.base_dim)(embeddings)
        embeddings = jnp.sin(embeddings)
        embeddings = nn.Dense(self.args.base_dim)(embeddings)
        embeddings = embeddings[...,None,None,:]

        if con_feat.img_state is not None:
            img_state_ft = nn.relu(nn.Dense(self.args.base_dim//2)(con_feat.img_state)) # (NB NO NV NI NJ NF)
            img_state_ft = jnp.max(img_state_ft, axis=-5) # (NB NV NI NJ NF)
            img_state_ft = nn.ConvTranspose(features=self.args.base_dim//2, kernel_size=(3,3), strides=(2,2))(img_state_ft)
            img_state_ft = nn.relu(img_state_ft)
            img_state_ft = nn.ConvTranspose(features=self.args.base_dim//2, kernel_size=(3,3), strides=(2,2))(img_state_ft)
            img_state_ft = nn.relu(img_state_ft)
            con_feat = con_feat.replace(img_state = img_state_ft)

        # extract features
        selected_img_fts, cam_fts = extract_pixel_features(objects_ptcl.dc_centers_tf, con_feat)
        for _ in range(2):
            cam_fts = nn.Dense(selected_img_fts.shape[-1]//2)(cam_fts)
            cam_fts = jnp.sin(cam_fts)
        cam_fts = cam_fts[...,None,:,:]
        cam_fts = jnp.broadcast_to(cam_fts, jnp.broadcast_shapes(cam_fts.shape, selected_img_fts[...,:1].shape))
        z_con = jnp.c_[selected_img_fts, cam_fts] # (B NO ND NC d)

        z_con = Aggregator(self.args)(z_con)

        bc_shape = jnp.broadcast_shapes(z_con.shape, embeddings.shape)
        context = jnp.c_[jnp.broadcast_to(z_con, bc_shape), jnp.broadcast_to(embeddings, bc_shape)]

        cen_emb = nn.Dense(self.args.base_dim//2)(objects_ptcl.dc_centers_tf)
        cen_emb = jnp.sin(cen_emb)
        cen_emb = nn.Dense(self.args.base_dim//2)(cen_emb)

        pos_emb = nn.Dense(self.args.base_dim//2)(objects_ptcl.pos)
        pos_emb = jnp.sin(pos_emb)
        pos_emb = nn.Dense(self.args.base_dim//2)(pos_emb)

        z_emb = objects_ptcl.z.swapaxes(-1,-2)
        z_emb = nn.Dense(z_emb.shape[-1], use_bias=False)(z_emb).swapaxes(-1,-2)
        z_emb = einops.rearrange(z_emb, '... i j -> ... (i j)')

        if nd == 1:
            z = jnp.c_[einops.repeat(pos_emb, '... d -> ... r2 d', r2=ncd), cen_emb, 
                       einops.repeat(z_emb, '... i d -> ... (r2 i) d', r2=ncd)] # (B NO ND d)
        else:
            z = jnp.c_[einops.repeat(pos_emb, '... d -> ... r2 d', r2=nd), cen_emb, z_emb, z_con] # (B NO ND d)
        skip = AdaLayerNorm()(nn.Dense(self.args.base_dim)(z), context)
        for i in range(3):
            z = nn.Dense(self.args.base_dim)(z)
            z = AdaLayerNorm()(z, context)
            z = nn.relu(z)
            if i ==1:
                z += skip

        mixing_depth = 4
        for _ in range(mixing_depth):
            if ncd > 1:
                # # local mixing
                z2 = nn.SelfAttention(num_heads=8, qkv_features=4*self.args.base_dim, out_features=self.args.base_dim, dropout_rate=0.0)(z, deterministic=True)
            else:
                z2 = nn.selu(nn.Dense(self.args.base_dim)(z))
            z = z + nn.Dropout(dropout)(z2, deterministic=not train)
            z = AdaLayerNorm()(z, context)
            # # # global mixing
            # z_gb = einops.rearrange(z, '... s c f -> ... (s c) f')
            # z2_gb = nn.SelfAttention(num_heads=8, qkv_features=4*self.args.base_dim, out_features=self.args.base_dim, dropout_rate=0.0)(z_gb, deterministic=True)
            # z2 = einops.rearrange(z2_gb, '... (r i) k -> ... r i k', i=ncd)
            # z = z + nn.Dropout(dropout)(z2, deterministic=not train)
            # z = AdaLayerNorm()(z, context)

        # conf branch
        out = z
        for i in range(3):
            out = nn.Dense(self.args.base_dim)(out)
            out = AdaLayerNorm()(out, context)
            out = nn.relu(out)
            out = jnp.c_[out, einops.repeat(jnp.mean(out, axis=-2), '... c -> ... r c', r=context.shape[-2])]
        out = nn.relu(nn.Dense(self.args.base_dim)(out))
        out = jnp.mean(out, axis=-2)
        # if out.shape[-2] !=1:
        #     out = out + nn.SelfAttention(num_heads=4, qkv_features=2*self.args.base_dim, out_features=self.args.base_dim, 
        #                             dropout_rate=dropout, broadcast_dropout=False)(out, deterministic=not train)
        out = nn.Dense(1)(out)
        # out = z
        # for i in range(2):
        #     out = nn.Dense(self.args.base_dim)(out)
        #     out = nn.relu(out)
        #     if i==0:
        #         out = jnp.mean(out, axis=-2)
        # out = nn.Dense(1)(out)
        
        if extended:
            return jax.tree_map(lambda x: x[0], out)
        else:
            return out


class SegModel(nn.Module):
    args:typing.NamedTuple

    @nn.compact
    def __call__(self, img_feat_struct, train=False):
        img_feat = img_feat_struct.img_feat
        
        for _ in range(3):
            img_feat = nn.Dense(self.args.base_dim)(img_feat)
            img_feat = nn.relu(img_feat)
        img_feat = nn.Dense(1)(img_feat)

        return img_feat


class SegModelCNN(nn.Module):
    args:typing.NamedTuple

    @nn.compact
    def __call__(self, x, train=False):
        depth =1

        def cnn(base_dim, filter, depth, x_):
            for _ in range(depth):
                x_ = nn.Conv(base_dim, (filter,filter))(x_)
                x_ = nn.relu(x_)
            return x_

        if x.dtype in [jnp.uint8, jnp.int16, jnp.int32]:
            x = x.astype(jnp.float32)/255.

        # down
        c_list = []
        for _ in range(2):
            x = nn.Conv(self.args.img_base_dim, (5,5))(x)
            x = nn.relu(x)
        x = cnn(2*self.args.img_base_dim, 3, depth, x)
        c_list.append(x)
        x = nn.Conv(2*self.args.img_base_dim, (3,3), strides=(2,2))(x)
        x = nn.relu(x)
        x = cnn(4*self.args.img_base_dim, 3, depth, x)
        c_list.append(x)
        x = nn.Conv(4*self.args.img_base_dim, (3,3), strides=(2,2))(x)
        x = nn.relu(x)
        x = cnn(8*self.args.img_base_dim, 3, depth, x)
        c_list.append(x)
        x = nn.Conv(8*self.args.img_base_dim, (3,3), strides=(2,2))(x)
        x = nn.relu(x)

        x = nn.Conv(8*self.args.img_base_dim, (3,3), kernel_dilation=(2,2))(x)
        x = nn.relu(x)
        x = nn.Conv(8*self.args.img_base_dim, (5,5), kernel_dilation=(2,2))(x)
        x = nn.relu(x)
        x = cnn(8*self.args.img_base_dim, 3, depth, x)

        def repeat_ft(x, r, ft_dim):
            x = nn.Dense(ft_dim)(x)
            x = nn.relu(x)
            x = einops.repeat(x, '... i j k -> ... (i r1) (j r2) k', r1=r, r2=r)
            return x

        # up
        c_list = list(reversed(c_list))
        # p_list = [einops.repeat(x, '... i j k -> ... (i r1) (j r2) k', r1=8, r2=8)]
        p_list = [repeat_ft(x, 8, 4*self.args.img_base_dim)]
        x = nn.ConvTranspose(8*self.args.img_base_dim, (3,3), strides=(2,2))(x)
        x = nn.relu(x) + c_list[0]
        # p_list.append(einops.repeat(x, '... i j k -> ... (i r1) (j r2) k', r1=4, r2=4))
        p_list.append(repeat_ft(x, 4, 4*self.args.img_base_dim))
        x = nn.ConvTranspose(4*self.args.img_base_dim, (3,3), strides=(2,2))(x)
        x = nn.relu(x) + c_list[1]
        # p_list.append(einops.repeat(x, '... i j k -> ... (i r1) (j r2) k', r1=2, r2=2))
        p_list.append(repeat_ft(x, 2, 4*self.args.img_base_dim))
        x = nn.ConvTranspose(2*self.args.img_base_dim, (3,3), strides=(2,2))(x)
        x = nn.relu(x) + c_list[2]
        p_list.append(repeat_ft(x, 1, 4*self.args.img_base_dim))

        x = jnp.concatenate(p_list, axis=-1)
        
        for _ in range(3):
            x = nn.Dense(self.args.base_dim)(x)
            x = nn.relu(x)
        x = nn.Dense(1)(x)

        return x


if __name__ == '__main__':
    np.random.seed(0)
    cond_feat = cutil.default_cond_feat(pixel_size=[32,32])
    cond_feat = cond_feat.replace(img_feat=np.arange(32*32*2).reshape(1,32,32,2).astype(jnp.float32))
    dc_centers_tf = np.random.uniform(-1,1,size=(10000,4,3))
    res = extract_pixel_features(dc_centers_tf, cond_feat)

    # grad = jax.grad(lambda x: jnp.sum(extract_pixel_features(*x)[0]))((dc_centers_tf, cond_feat))

    print(1)