'''
2023-11-29
Jaehyung Kim
'''
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

@struct.dataclass
class Models:
    args: typing.NamedTuple=None
    rot_configs: typing.Sequence=None
    latent_shape: typing.Sequence=None

    img_encoder_params: FrozenDict = None
    img_encoder_batch_stats: FrozenDict = None
    img_encoder_model: nn.Module = None

    detr_params: FrozenDict = None
    detr_batch_stats: FrozenDict = None
    detr_model: nn.Module = None

    def load_model(self, save_dir):
        with open(os.path.join(save_dir, 'saved.pkl'), 'rb') as f:
            loaded = pickle.load(f)
        
        params = loaded['params']
        args = loaded['args']
        args.checkpoint_dir = save_dir

        rot_configs = loaded['rot_configs']

        self = self.replace(args=args, rot_configs=rot_configs)
        self = self.set_params(params)
        return self

    def init_detr_model(self, pretrained_model, args, ds)->"Models":

        jkey = jax.random.PRNGKey(args.seed+41)
        imgenc_model = SmallImageFeature(args.img_base_dim)
        detr_model = DETRModel()

        # init all variables
        cvx_obj, rgbs, cam_info = jax.tree_map(lambda x : jnp.array(x), ds)

        cvx_obj:cxutil.CvxObjects
        latent_obj = cvx_obj.set_z_with_models(jkey, pretrained_model)
        _, jkey = jax.random.split(jkey)

        img_feat, imgenc_params = imgenc_model.init_with_output(jkey, rgbs, cam_info)
        _, jkey = jax.random.split(jkey)

        # img_feat_structs = structs.ImgFeatures(*cam_info, img_feat)
        
        detr_params = detr_model.init({'params':jkey, 'dropout':jkey}, img_feat)
        _, jkey = jax.random.split(jkey)

        self = self.replace(latent_shape=latent_obj.latent_shape)

        # network info out

        # def cal_num_params(params_):
        #     return np.sum(jax.tree_map(lambda x: np.prod(x.shape), jax.tree_util.tree_flatten(params_)[0]))
        
        # imgenc_num_params = cal_num_params(imgenc_params)
        # detr_num_params = cal_num_params(detr_params)

        # enc_app = jax.jit(imgenc_model.apply)
        # detr_app = jax.jit(detr_model.apply)

        names = ['img_encoder', 'detr']
        models = [imgenc_model, detr_model]
        params = (imgenc_params, detr_params)
        self = self.replace(**{name+'_model': model for name, model in zip(names, models)})
        self = self.replace(**{name+'_params': model for name, model in zip(names, params)})
        return self
    
    def apply(self, name, *args, train=False, **kwargs):
        model = getattr(self, name+'_model')
        params = getattr(self, name+'_params')
        if not train:
            params = jax.lax.stop_gradient(params)
        batch_stats = getattr(self, name+'_batch_stats')

        if batch_stats is None:
            return model.apply(params, *args, **kwargs)
        
    def set_params(self, params)->"Models":
        self = self.replace(**{k+'_params': params[k] for k in params})
        return self

    @property
    def names(self):
        return [k[:-6] for k in self.__annotations__ if k[-5:]=='model']

    @property
    def params(self):
        return {name: getattr(self, name+'_params') for name in self.names}

class SmallImageFeature(nn.Module):
    base_dim:int=8
    depth:int=1

    @nn.compact
    def __call__(self, x, train=False):

        def cnn(base_dim, filter, depth, x_):
            for _ in range(depth):
                x_ = nn.Conv(base_dim, (filter,filter))(x_)
                x_ = nn.relu(x_)
            return x_

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

        return x

        # def repeat_ft(x, r, ft_dim):
        #     x = nn.Dense(ft_dim)(x)
        #     x = nn.relu(x)
        #     x = einops.repeat(x, '... i j k -> ... (i r1) (j r2) k', r1=r, r2=r)
        #     return x

        # # up
        # c_list = list(reversed(c_list))
        # # p_list = [einops.repeat(x, '... i j k -> ... (i r1) (j r2) k', r1=8, r2=8)]
        # p_list = [repeat_ft(x, 8, 4*self.base_dim)]
        # x = nn.ConvTranspose(8*self.base_dim, (3,3), strides=(2,2))(x)
        # x = nn.relu(x) + c_list[0]
        # # p_list.append(einops.repeat(x, '... i j k -> ... (i r1) (j r2) k', r1=4, r2=4))
        # p_list.append(repeat_ft(x, 4, 4*self.base_dim))
        # x = nn.ConvTranspose(4*self.base_dim, (3,3), strides=(2,2))(x)
        # x = nn.relu(x) + c_list[1]
        # # p_list.append(einops.repeat(x, '... i j k -> ... (i r1) (j r2) k', r1=2, r2=2))
        # p_list.append(repeat_ft(x, 2, 4*self.base_dim))
        # x = nn.ConvTranspose(2*self.base_dim, (3,3), strides=(2,2))(x)
        # x = nn.relu(x) + c_list[2]
        # p_list.append(repeat_ft(x, 1, 4*self.base_dim))

        # x = jnp.concatenate(p_list, axis=-1)

        # return x

class MultiHeadAttentionLayer(nn.Module):
    feature_dim: int
    value_dim: int
    num_heads: int
    key_query_dim: int = None

    def setup(self):
        self.mha_layer = nn.Dense(self.feature_dim)


    def _multihead_attention(self, k: jnp.ndarray, q: jnp.ndarray, v: jnp.ndarray) -> jnp.ndarray:
        attn = jnp.einsum("btij,bsij->btsi", q, k) / jnp.sqrt(self.key_query_dim)
        attn = jax.nn.softmax(attn, axis=2)
        z = jnp.einsum("btsi,bsij->btij", attn, v).reshape(
            q.shape[0], q.shape[1], self.num_heads * self.value_dim
        )
        return self.mha_layer(z)
    
    @nn.compact
    def __call__(self, key: jnp.ndarray, query: jnp.ndarray, value: jnp.ndarray) -> jnp.ndarray:
        k = nn.Dense(self.key_query_dim * self.num_heads)(key).reshape((*key.shape[:-1], self.num_heads, self.key_query_dim))
        q = nn.Dense(self.key_query_dim * self.num_heads)(query).reshape((*query.shape[:-1], self.num_heads, self.key_query_dim))
        v = nn.Dense(self.value_dim * self.num_heads)(value).reshape((*value.shape[:-1], self.num_heads, self.value_dim))
        return self._multihead_attention(k, q, v)


class DecoderLayer(nn.Module):
    feature_dim: int
    num_heads: int
    dropout_rate: float
    feedforward_dim: int

    def setup(self):
        self.detr_attn = MultiHeadAttentionLayer(self.feature_dim, self.feature_dim, self.num_heads, self.feature_dim)
        self.mha = MultiHeadAttentionLayer(self.feature_dim, self.feature_dim, self.num_heads, self.feature_dim)
        self.layernorm1 = nn.LayerNorm()
        self.layernorm2 = nn.LayerNorm()
        self.layernorm3 = nn.LayerNorm()
        self.ffn_dense1 = nn.Dense(self.feedforward_dim)
        self.ffn_dense2 = nn.Dense(self.feature_dim)
    
    @nn.compact
    def __call__(
        self,
        encoder_features: jnp.ndarray,
        decoder_features: jnp.ndarray,
        query_encoding: jnp.ndarray,
        is_training: bool,
    ) -> jnp.ndarray:
        
        deterministic = not is_training
        y = self.detr_attn(decoder_features+query_encoding, decoder_features+query_encoding, decoder_features)
        y = decoder_features # + nn.Dropout(self.dropout_rate)(y, deterministic=deterministic)
        x = self.layernorm1(y)

        y = self.mha(encoder_features, x + query_encoding, encoder_features)
        y = x + y # nn.Dropout(0.1, deterministic=deterministic)(y)
        x = self.layernorm2(y)

        y = self.ffn_dense1(x)
        y = jax.nn.relu6(y)
        # y = nn.Dropout(0.1, deterministic=deterministic)(y)
        y = self.ffn_dense2(y)
        y = x + y # nn.Dropout(0.1, deterministic=deterministic)(y)

        return self.layernorm3(y)


class DETRModel(nn.Module):
    feature_dim: int = 128
    num_heads: int = 8
    num_decoder_layers: int = 3 # 6
    num_queries: int = 32
    feedforward_dim: int = 256 # 2048
    dropout_rate: float = 0.1
    no: int = 1
    nf: int = 8
    nz: int = 32

    @nn.compact
    def __call__(self, img_feat, is_training=True):
        NB, NV, H, W, NC = img_feat.shape # NC is the hidden_dim from 1dconv
        img_feat_axismoved = jnp.moveaxis(img_feat, -1, 2) # NB, NV, NC, H, W
        img_feat_flatten = img_feat.reshape(NB, NV, NC, -1)

        img_feat_one_view = img_feat_flatten.squeeze(1) # NB, NC, H*W

        query_encoding = self.param(
            "query_encoding",
            nn.initializers.normal(stddev=0.02),
            (1, self.num_queries, self.feature_dim)
        )
        query_encoding = jnp.tile(query_encoding, (NB, 1, 1))

        decoder_features = jnp.zeros_like(query_encoding)

        for _ in range(self.num_decoder_layers):
            decoder_features = DecoderLayer(
                self.feature_dim,
                self.num_heads,
                self.dropout_rate,
                feedforward_dim=self.feedforward_dim,
            )(
                img_feat_one_view,
                decoder_features,
                query_encoding,
                is_training
            )

        z = nn.Dense(self.nf*self.nz)(decoder_features)
        dc_rel_centers = nn.Dense(3)(decoder_features)
        
        decoder_features_flatten = decoder_features.reshape(NB, -1)
        pos = nn.Dense(3)(decoder_features_flatten)
        
        
        return z.reshape(NB, -1), dc_rel_centers.reshape(NB, -1), pos.reshape(NB, -1)


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
        selected_img_fts = jnp.take_along_axis(img_fts_flat_, einops.rearrange(px_flat_idx, '... i j k -> ... j i k')[...,None], axis=-3) # (... NC NO NG NF)
        img_fts = einops.rearrange(selected_img_fts, '... i j p k -> ... j i p k') # (... NO NC NG NF)
        # img_fts = jnp.concatenate([out_pnts_indicator[...,None].astype(jnp.float32), img_fts], axis=-1)
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



# auxiliary decoders

def import_module_from_full_path(module_path):
    spec = importlib.util.spec_from_file_location(os.path.basename(module_path)[:-3], module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def get_models_from_cp_dir(cp_dir)->'Models':
    # mutil = import_module_from_full_path(os.path.join(cp_dir, 'util/model_util.py'))
    # return mutil.Models().load_model(cp_dir)
    return Models().load_model(cp_dir)


if __name__ == '__main__':
    np.random.seed(0)
    cond_feat = cutil.default_cond_feat(pixel_size=[32,32])
    cond_feat = cond_feat.replace(img_feat=np.arange(32*32*2).reshape(1,32,32,2).astype(jnp.float32))
    dc_centers_tf = np.random.uniform(-1,1,size=(10000,4,3))
    res = extract_pixel_features(dc_centers_tf, cond_feat)

    # Initialize model
    model = DETRModel(feature_dim=256, num_heads=8, num_decoder_layers=6, num_queries=32, feedforward_dim=2048, dropout_rate=0.1)

    # Dummy image feature
    img_feat = jnp.ones((10, 1, 32, 32, 64))  # Example dimensions NB, NV, H, W, NC

    # Initialize parameters
    params = model.init(jax.random.PRNGKey(0), img_feat)

    # Forward pass
    output = model.apply(params, img_feat)

    print(1)