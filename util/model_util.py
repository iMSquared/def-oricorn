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
import util.dif_model_util as dmutil
import data_generation.scene_generation as sg

@struct.dataclass
class Models:
    args: typing.NamedTuple=None
    dif_args: typing.NamedTuple=None
    rot_configs: typing.Sequence=None
    latent_shape: typing.Sequence=None
    pixel_size: typing.Sequence[int]=None

    shape_encoder_params: FrozenDict = None
    shape_encoder_batch_stats: FrozenDict = None
    shape_encoder_model: nn.Module = None
    
    col_predictor_params: FrozenDict = None
    col_predictor_batch_stats: FrozenDict = None
    col_predictor_model: nn.Module = None
    
    occ_predictor_params: FrozenDict = None
    occ_predictor_batch_stats: FrozenDict = None
    occ_predictor_model: nn.Module = None
    
    pln_predictor_params: FrozenDict = None
    pln_predictor_batch_stats: FrozenDict = None
    pln_predictor_model: nn.Module = None

    ray_predictor_params: FrozenDict = None
    ray_predictor_batch_stats: FrozenDict = None
    ray_predictor_model: nn.Module = None

    img_encoder_params: FrozenDict = None
    img_encoder_batch_stats: FrozenDict = None
    img_encoder_model: nn.Module = None

    denoiser_params: FrozenDict = None
    denoiser_batch_stats: FrozenDict = None
    denoiser_model: nn.Module = None

    obs_model_params: FrozenDict = None
    obs_model_batch_stats: FrozenDict = None
    obs_model_model: nn.Module = None

    seg_predictor_params: FrozenDict = None
    seg_predictor_batch_stats: FrozenDict = None
    seg_predictor_model: nn.Module = None

    learnable_queries: jnp.ndarray = None

    @property
    def nh(self)->jnp.ndarray:
        return self.latent_shape[0] * self.latent_shape[1] * self.latent_shape[2] + 3 + 3 * 32


    def load_model(self, save_dir):
        with open(os.path.join(save_dir, 'saved.pkl'), 'rb') as f:
            loaded = pickle.load(f)
        
        params = loaded['params']
        args = loaded['args']
        args.checkpoint_dir = save_dir

        if 'occ_dec_type' not in args:
            args.occ_dec_type = 0
            args.ray_dec_type = 0
        if 'reduce_elements' not in args:
            args.reduce_elements = 0

        rot_configs = loaded['rot_configs']
        shape_enc, occ_dec, col_dec, pln_dec, ray_dec = get_encdec_models(args, rot_configs, relative_checkpoint_dir=save_dir)

        names = self.pretraining_names
        models = [shape_enc, occ_dec, col_dec, pln_dec, ray_dec]
        self = self.replace(**{name+'_model': model for name, model in zip(names, models)})
        self = self.replace(args=args, rot_configs=rot_configs)
        self = self.set_params(params)
        return self

    def init_model(self, args, ds):
        """
        args: args
        ds: batch sample for model init
        """

        jkey = jax.random.PRNGKey(args.seed)
        rot_configs = rmutil.init_rot_config(args.seed, dim_list=args.rot_order, rot_type=args.rot_type)
        shape_enc, occ_dec, col_dec, pln_dec, ray_dec = get_encdec_models(args, rot_configs)
        # init all variables
        ds = jax.tree_map(lambda x : jnp.array(x), ds)
        qpnts = jnp.array(ds[4])
        obj = cxutil.CvxObjects().init_vtx(*ds[:2]).init_pcd(*ds[2:4])
        latent_obj, shape_enc_params = shape_enc.init_with_output(jkey, obj, jkey=jkey) # latent_obj: embedding : (B, 2, C or 1, F(8), D)
        _, jkey = jax.random.split(jkey)
        occ_dec_params = occ_dec.init(jkey, latent_obj, qpnts)
        _, jkey = jax.random.split(jkey)
        col_dec_params = col_dec.init({'params':jkey, 'dropout':jkey}, latent_obj, jkey)
        _, jkey = jax.random.split(jkey)
        pln_dec_params = pln_dec.init(jkey, latent_obj, ds[7], ds[8])
        _, jkey = jax.random.split(jkey)
        ray_dec_params = ray_dec.init(jkey, latent_obj, ds[10], ds[11])
        _, jkey = jax.random.split(jkey)

        self = self.replace(latent_shape=latent_obj.latent_shape)

        # network info out
        def cal_num_params(params_):
            return np.sum(jax.tree_map(lambda x: np.prod(x.shape), jax.tree_util.tree_flatten(params_)[0]))
        enc_num_params = cal_num_params(shape_enc_params)
        occ_dec_num_params = cal_num_params(occ_dec_params)
        col_dec_num_params = cal_num_params(col_dec_params)
        pln_dec_num_params = cal_num_params(pln_dec_params)
        ray_dec_num_params = cal_num_params(ray_dec_params)


        enc_app = jax.jit(shape_enc.apply)
        occ_dec_app = jax.jit(occ_dec.apply)
        col_dec_app = jax.jit(col_dec.apply)
        ray_dec_app = jax.jit(ray_dec.apply)
        pln_dec_app = jax.jit(pln_dec.apply)
        def test_time(func, inputs):
            for i in range(101):
                if i==1:
                    stim = time.time()
                y = func(*inputs)
            tim_ = (time.time() - stim)
            tim_ = tim_/100
            return tim_
        
        tim_enc = test_time(enc_app, (shape_enc_params, obj, jkey))
        tim_occ_dec = test_time(occ_dec_app, (occ_dec_params, latent_obj, qpnts))
        tim_col_dec = test_time(col_dec_app, (col_dec_params, latent_obj))
        tim_pln_dec = test_time(pln_dec_app, (pln_dec_params, latent_obj, ds[7], ds[8]))
        tim_ray_dec = test_time(ray_dec_app, (ray_dec_params, latent_obj, ds[10], ds[11]))

        print(f'enc num params: {enc_num_params//1000}k, occ_dec: {occ_dec_num_params//1000}k, col_dec: {col_dec_num_params//1000}k, ray_dec: {ray_dec_num_params//1000}k, pln_dec: {pln_dec_num_params//1000}k')
        print(f'duration enc: {tim_enc*1000}ms, occ_dec: {tim_occ_dec*1000}ms, col_dec: {tim_col_dec*1000}ms, ray_dec: {tim_ray_dec*1000}ms, pln_dec: {tim_pln_dec*1000}ms')

        names = self.pretraining_names
        models = [shape_enc, occ_dec, col_dec, pln_dec, ray_dec]
        params = (shape_enc_params, occ_dec_params, col_dec_params, pln_dec_params, ray_dec_params)
        self = self.replace(**{name+'_model': model for name, model in zip(names, models)})
        self = self.replace(**{name+'_params': model for name, model in zip(names, params)})
        self = self.replace(args=args)
        self = self.replace(rot_configs=rot_configs)
        return self

    def init_dif_model_scenedata(self, args, ds:sg.SceneCls.SceneData)->"Models":

        jkey = jax.random.PRNGKey(args.seed+41)
        imgenc_model = dmutil.ImageFeatureEntire(args, args.img_base_dim)
        if args.dif_model_version in [1,4,5,6]:
            denoise_model = dmutil.DenoisingModelV4(args, self.args, self.rot_configs)
        elif args.dif_model_version == 2:
            denoise_model = dmutil.DenoisingModelV2(args, self.args, self.rot_configs)
        elif args.dif_model_version == 3:
            denoise_model = dmutil.DenoisingModelVoxel(args, self.args, self.rot_configs)


        # init all variables
        rgbs = ds["rgbs"]
        if rgbs is None:
            rgbs = jnp.zeros(ds['cam_info']['cam_intrinsics'].shape[:-1] + (48,48,3), dtype=jnp.float32)
        cam_info = ds["cam_info"]
        obj_info = ds["obj_info"]
        cvx_obj = cxutil.CvxObjects().init_obj_info(obj_info)
        latent_obj = cvx_obj.set_z_with_models(jkey, self, dc_center_no=args.volume_points_no)
        _, jkey = jax.random.split(jkey)
        img_feat_structs, imgenc_params = imgenc_model.init_with_output(jkey, rgbs, cam_info["cam_posquats"].astype(jnp.float32), cam_info["cam_intrinsics"].astype(jnp.float32))
        _, jkey = jax.random.split(jkey)
        time_cond = jnp.ones((latent_obj.outer_shape[0], ))
        if ds["rgbs"] is None:
            img_feat_structs = None
            obs_model = None
            obs_params = None
            seg_model = None
            seg_params = None
        else:
            self = self.replace(pixel_size = rgbs.shape[-3:-1])
            _, jkey = jax.random.split(jkey)
            obs_model = None
            obs_params = None
            _, jkey = jax.random.split(jkey)
            if args.separate_seg_model:
                seg_model = dmutil.SegModelCNN(args)
                seg_params = seg_model.init(jkey, rgbs)
            else:
                seg_model = dmutil.SegModel(args)
                seg_params = seg_model.init(jkey, img_feat_structs)
            
        denoiser_params = denoise_model.init({'params':jkey, 'dropout':jkey}, latent_obj, img_feat_structs, time_cond)

        self = self.replace(latent_shape=latent_obj.latent_shape)

        if args.one_step_test == 3:
            _, jkey = jax.random.split(jkey)
            learnable_queries = jax.random.normal(jkey, (args.nparticles, self.nh))
            learnable_queries = dfutil.noise_FER_projection(learnable_queries, self.latent_shape, self.rot_configs)
            self = self.replace(learnable_queries=learnable_queries)

        names = self.difmodel_names
        models = [imgenc_model, denoise_model, obs_model, seg_model]
        params = (imgenc_params, denoiser_params, obs_params, seg_params)
        self = self.replace(**{name+'_model': model for name, model in zip(names, models)})
        self = self.replace(**{name+'_params': model for name, model in zip(names, params)})
        self = self.replace(dif_args=args)

        return self

    def cal_statics(self):

        jkey = jax.random.PRNGKey(71)
        test_nb = 1
        test_nv = 3
        test_no = 7
        rgb_dummy = jax.random.randint(jkey, shape=(test_nb, test_nv, *self.pixel_size, 3), minval=0, maxval=255, dtype=jnp.uint8)
        cam_posquats_dummy = jax.random.normal(jkey, shape=(test_nb, test_nv, 7), dtype=jnp.float32)
        cam_intrinsic_dummy = jax.random.normal(jkey, shape=(test_nb, test_nv, 6), dtype=jnp.float32)
        latent_obj_dummy = cxutil.LatentObjects().init_h(jax.random.normal(jkey, shape=(test_nb, test_no, self.nh), dtype=jnp.float32), self.latent_shape)
        time_dummy = jax.random.uniform(jkey, shape=(test_nb,))
        
        # network info out
        def cal_num_params(params_):
            return np.sum(jax.tree_map(lambda x: np.prod(x.shape), jax.tree_util.tree_flatten(params_)[0]))
        imgenc_num_params = cal_num_params(self.img_encoder_params)
        denoiser_num_params = cal_num_params(self.denoiser_params)

        enc_app = jax.jit(self.img_encoder_model.apply)
        den_app = jax.jit(self.denoiser_model.apply)
        def test_time(func, inputs):
            for i in range(101):
                if i==1:
                    stim = time.time()
                y = func(*inputs, rngs={'dropout':jkey})
                y = jax.block_until_ready(y)
            tim_ = (time.time() - stim)
            tim_ = tim_/100
            return tim_, y
        
        tim_enc, img_feat_struct_dummy = test_time(enc_app, (self.img_encoder_params, rgb_dummy, cam_posquats_dummy, cam_intrinsic_dummy))
        tim_dec, _ = test_time(den_app, (self.denoiser_params, latent_obj_dummy, img_feat_struct_dummy, time_dummy))

        print(f'img enc num params: {imgenc_num_params//1000}k, denoiser: {denoiser_num_params//1000}k')
        print(f'duration img_enc: {tim_enc*1000}ms, denoiser: {tim_dec*1000}ms')

        return {'img_enc_num_params':imgenc_num_params, 'img_enc_time':tim_enc, 'den_num_params':denoiser_num_params, 'den_time':tim_dec}


    def apply(self, name, *args, train=False, **kwargs):
        model = getattr(self, name+'_model')
        params = getattr(self, name+'_params')
        if not train:
            params = jax.lax.stop_gradient(params)
        batch_stats = getattr(self, name+'_batch_stats')
        if batch_stats is None:
            return model.apply(params, train=train, *args, **kwargs)
        
    def set_params(self, params:typing.Dict)->"Models":
        if 'learnable_queries' in params:
            self = self.replace(learnable_queries = params['learnable_queries'])
        self = self.replace(**{k+'_params': params[k] for k in params if k!='learnable_queries'})
        return self

    def pairwise_collision_prediction(self, objsA:cxutil.LatentObjects, objsB:cxutil.LatentObjects, jkey, train=False):
        '''
        objsA: outer shape (... NA)
        objsB: outer shape (... NB)
        return collsion results (... NA NB)
        '''
        na = objsA.outer_shape[-1]
        nb = objsB.outer_shape[-1]
        valid_objA_mask = jnp.all(objsA.pos<8.0, axis=-1) # objects which are outside of 8-m box are invalid
        valid_objB_mask = jnp.all(objsB.pos<8.0, axis=-1)
        objsA = objsA.extend_and_repeat_outer_shape(nb, -1)
        objsB = objsB.extend_and_repeat_outer_shape(na, -2)
        if 'implicit_baseline' in self.dif_args and self.dif_args.implicit_baseline:
            assert objsA.rel_pcd is not None
            assert objsB.rel_pcd is not None
            occ_res_A = self.apply('occ_predictor', objsA, objsB.pcd_tf, train=train)
            occ_res_B = self.apply('occ_predictor', objsB, objsA.pcd_tf, train=train)
            col_res = jnp.maximum(jnp.max(occ_res_A, axis=-1), jnp.max(occ_res_B, axis=-1))
        else:
            objsAB = objsA.stack(objsB, -1)
            col_res = self.apply('col_predictor', objsAB, rngs={'dropout':jkey}, train=train)
        col_res = jnp.where(valid_objA_mask[...,None], col_res, -100)
        col_res = jnp.where(valid_objB_mask[...,None,:], col_res, -100)
        return col_res

    @property
    def names(self):
        return [k[:-6] for k in self.__annotations__ if k[-5:]=='model']

    @property
    def pretraining_names(self):
        return ['shape_encoder', 'occ_predictor', 'col_predictor', 'pln_predictor', 'ray_predictor']

    @property
    def difmodel_names(self):
        return['img_encoder', 'denoiser', 'obs_model', 'seg_predictor']

    @property
    def params(self):
        if self.dif_args is not None and self.dif_args.one_step_test == 3:
            params_ = {name: getattr(self, name+'_params') for name in self.names}
            params_['learnable_queries'] = self.learnable_queries
            return params_
        else:
            return {name: getattr(self, name+'_params') for name in self.names}

    @property
    def pretraining_params(self):
        return {name: getattr(self, name+'_params') for name in self.pretraining_names}




class VSONCvxEncoder(nn.Module):
    args:typing.NamedTuple
    rot_configs:typing.Sequence

    @nn.compact
    def __call__(self, obj:cxutil.CvxObjects, jkey=None, det=True, train=False):
        base_dim = self.args.enc_base_dim

        dc_valid_mask = obj.dc_valid_mask.astype(jnp.float32)
        cen = obj.dc_rel_centers * dc_valid_mask[...,None]
        cen_origin = cen # (B 2 C 3)
        rel_pnts = obj.rel_pcd # (B 2 P 3)
        # rel_pnts_origin = rel_pnts
        dc_idx = obj.pcd_dc_idx # (B 2 P)
        ndc = dc_valid_mask.shape[-1]

        # reduce to batch size
        origin_outer_shape = obj.outer_shape
        cen = cen.reshape((-1,) + cen.shape[-2:]) # (B C 3)
        rel_pnts = rel_pnts.reshape((-1,) + rel_pnts.shape[-2:]) # (B P 3)
        dc_idx = dc_idx.reshape((-1,)+dc_idx.shape[-1:]) # (B P)
        dc_valid_mask_reduced = dc_valid_mask.reshape((-1,) + dc_valid_mask.shape[-1:])

        x = jnp.expand_dims(rel_pnts, -1)
        x = eutil.get_graph_feature(x, k=10, cross=True) # (B, P, K, F, D)
        x = evl.MakeHDFeature(self.args, self.rot_configs)(x)
        x = evl.EVLinearLeakyReLU(self.args, base_dim)(x)
        x = jnp.mean(x, -3) # (B, P, F, D)

        def cvx_meanpool(x):
            x = jax.vmap(lambda x, di: jax.ops.segment_sum(x, di, ndc))(x, dc_idx) # (B P F D) -> (B C F D)
            x = x/(jnp.sum((dc_idx[...,None]==jnp.arange(ndc)).astype(jnp.float32), axis=-2)[...,None,None] + 1e-5)
            return x

        def recover_dctopcd(x):
            return jnp.take_along_axis(x, dc_idx[...,None,None], -3)

        np = x.shape[-3]
        x = nn.Dense(self.args.enc_base_dim, use_bias=False)(x)
        for i in range(4):
            x = evl.EVNResnetBlockFC(self.args, self.args.enc_base_dim, self.args.enc_base_dim)(x)
            # Local + global feature
            if (('use_local_feature_in_encoder' not in self.args or 'use_global_feature_in_encoder' not in self.args) or
                (self.args.use_local_feature_in_encoder and 
                self.args.use_global_feature_in_encoder)):
                pooled_local = recover_dctopcd(cvx_meanpool(x))
                pooled_global = einops.repeat(jnp.mean(x, -3), '... f d -> ... r f d', r=np)
                x = jnp.concatenate([x, pooled_local, pooled_global], -1)
            # Local only
            elif self.args.use_local_feature_in_encoder:
                pooled_local = recover_dctopcd(cvx_meanpool(x))
                x = jnp.concatenate([x, pooled_local], -1)
            # Global only
            elif self.args.use_global_feature_in_encoder:
                pooled_global = einops.repeat(jnp.mean(x, -3), '... f d -> ... r f d', r=np)
                x = jnp.concatenate([x, pooled_global], -1)
            # Select at least one...
            else:
                raise ValueError("Select at least one between global feature and local feature")
        x = evl.EVNResnetBlockFC(self.args, self.args.enc_base_dim, self.args.enc_base_dim)(x)

        # Aggregation
        if 'use_global_representation_only' in self.args and self.args.use_global_representation_only:
            x = jnp.mean(x, axis=-3, keepdims=True) # (B, 1, F, D)
        else:
            x_global = jnp.mean(x, axis=-3, keepdims=True) # (B, 1, F, D)
            x = cvx_meanpool(x) # (B, C, F, D)
            x = jnp.where(dc_valid_mask_reduced[...,None,None] > 0.5, x, x_global) # Fill masked region to global embedding...
        x = evl.EVNNonLinearity(self.args)(x)
        # Experimental convex reducing...
        if self.args.reduce_set_no:
            x = evl.InvCrossAttention(self.args, 2*self.args.enc_base_dim, 2*self.args.enc_base_dim, query_size=ndc//2, multi_head=4)(x)
        
        # Final layer
        x = nn.Dense(self.args.enc_base_dim//2, use_bias=False)(x)
        x = x.reshape(origin_outer_shape + x.shape[-3:])

        # create random centers
        cen_concat = cen_origin
        # _, jkey = jax.random.split(jkey)
        # idx = jax.random.randint(jkey, shape=rel_pnts_origin[...,:ndc,:1].shape, minval=0, maxval=ndc)
        # sample_spnts = jnp.take_along_axis(rel_pnts_origin, idx, axis=-2)
        # cen_concat = jnp.where(dc_valid_mask[...,None] > 0.5, cen_origin, sample_spnts)

        # x: embedding : (B, 2, C or 1, F(8), D)

        latnet_obj = cxutil.LatentObjects()
        latnet_obj = replace(latnet_obj, z=x, dc_rel_centers=cen_concat, pos=obj.pos)
        return latnet_obj


class VSONDecoder(nn.Module):
    args:typing.NamedTuple
    rot_configs:typing.Sequence

    @nn.compact
    def __call__(self, latent_obj:cxutil.LatentObjects, p:jnp.ndarray, jkey=None, det=True, features_out=False, train=False):
        '''
        z: (... C F D)
        p: (... P 3)
        '''

        if self.args.occ_dec_type==1:
            p = p-latent_obj.pos[...,None,:] # centering
            Rp = rmutil.make_Rp(p, base_axis='z', normalize_input=True)
            objects_ = latent_obj.translate(-latent_obj.pos).drop_gt_info()
            objects_:cxutil.LatentObjects = jax.tree_util.tree_map(lambda x: jnp.expand_dims(x, axis=len(objects_.outer_shape)), objects_)
            objects_:cxutil.LatentObjects = objects_.rotate_z(tutil.Rm2q(tutil.Rm_inv(Rp)), self.rot_configs) # (... NQ ...)
            qp_norm = jnp.linalg.norm(p, axis=-1, keepdims=True)
            for _ in range(2):
                qp_norm = nn.Dense(16)(qp_norm)
                qp_norm = jnp.sin(qp_norm)
            
            # z = nn.MultiHeadDotProductAttention(4, qkv_features=self.args.dec_base_dim, out_features=self.args.dec_base_dim, deterministic=True)(qp_norm[...,None,:], objects_.z_flat)
            # z = jnp.squeeze(z, axis=-2)
            # z = nn.Dense(1)(z).squeeze(-1)

            z = jnp.c_[einops.repeat(qp_norm, '... i -> ... r i', r=objects_.z.shape[-3]), objects_.z_flat]
            for i in range(3):
                z = nn.Dense(self.args.dec_base_dim)(z)
                z = nn.relu(z)
                if i == 1:
                    z = jnp.max(z, axis=-2)
            z = nn.Dense(1)(z)
            z = jnp.squeeze(z, -1)

            return z

        elif self.args.occ_dec_type==0:

            z = latent_obj.z
            pos = latent_obj.pos
            np = p.shape[-2]
            p = p-pos[...,None,:] # centering
            
            # p projection
            # if not train:
            #     p_norm = jnp.linalg.norm(p, axis=-1, keepdims=True)
            #     p = jnp.where(p_norm>0.5, p/p_norm*0.5, p)

            p_ext = evl.MakeHDFeature(self.args, self.rot_configs)(p[...,None]).squeeze(-1)

            if 'use_global_representation_only' not in self.args or ((not self.args.use_global_representation_only) and self.args.reduce_elements):
                z = evl.QueryElements(self.args)(p_ext, z) # (... P 1 F D)
            nc = z.shape[-3]

            net = (p_ext * p_ext).sum(-1, keepdims=True) # (... P 1)
            z_dir = nn.Dense(z.shape[-1], use_bias=False)(z)
            z_inv = (z * z_dir).sum(-2) # (... C D)
            if z.ndim - p_ext.ndim == 1:
                net_z = jnp.einsum('...pf,...cfd->...pcd', p_ext, z) # (... P C D)
                z_inv = einops.repeat(z_inv, '... c b -> ... r c b', r=np)
            elif z.ndim - p_ext.ndim == 2:
                net_z = jnp.einsum('...pf,...pcfd->...pcd', p_ext, z) # (... P C D)
            net = einops.repeat(net, '... b -> ... r b', r=nc)
            net = jnp.concatenate([net, net_z, z_inv], axis=-1) # (... P C D)

            net = nn.Dense(self.args.dec_base_dim)(net)
            activation = nn.relu
            for i in range(5):
                x_s = net
                net = nn.Dense(self.args.dec_base_dim)(activation(net))
                net = nn.Dense(self.args.dec_base_dim)(activation(net)) + x_s
                if i==0:
                    net = jnp.max(net, axis=-2) # remove C dimensions
                if i==1:
                    pnt_feature = activation(net)
            out = nn.Dense(1)(activation(net))
            out = out.squeeze(-1)
            if features_out:
                return out, z, pnt_feature
            else:
                return out



class ColDecoder(nn.Module):
    args:typing.NamedTuple
    rot_configs:typing.Sequence

    @nn.compact
    def __call__(self, obj_pair:cxutil.LatentObjects, jkey=None, train=False):
        '''
        obj_pair - outer_shape: (B, 2)
        '''
        
        # z_pair, cen_pair, pos_pair = emb_pair

        # Center
        obj_pair_tf = obj_pair.translate(-jnp.mean(obj_pair.pos, axis=-2, keepdims=True))
        # Align pair to z-axis
        dc_tf_norm = jnp.linalg.norm(obj_pair_tf.dc_centers_tf, axis=-1) # (B 2 D)
        dc_tf_max = jnp.take_along_axis(obj_pair_tf.dc_centers_tf, jnp.argmax(dc_tf_norm, axis=-1)[...,None,None], axis=-2) # (B 2 1 3)
        dc_tf_max_norm = jnp.linalg.norm(dc_tf_max, axis=-1)
        z_axis = jnp.where(dc_tf_max_norm[...,0:1,:] > dc_tf_max_norm[...,1:2,:], obj_pair_tf.pos[...,0:1,:], obj_pair_tf.pos[...,1:2,:]) # (B 1 3)
        y_axis = jnp.where(dc_tf_max_norm[...,0:1,:] > dc_tf_max_norm[...,1:2,:], dc_tf_max[...,0,:,:], dc_tf_max[...,1,:,:])  # (B 1 3)
        y_axis = jnp.where(jnp.abs(jnp.sum(y_axis*z_axis, -1, keepdims=True))<1e-4, 
            jnp.sum(obj_pair_tf.z[...,0,:3,0], axis=-2, keepdims=True), y_axis)
        # Align pair to z-axis
        qoff = tutil.line2q(z_axis, yaxis=y_axis)
        # qoff = tutil.line2q(obj_pair_tf.pos[...,0:1,:], yaxis=jnp.array([1,0,0]))
        qoffinv = tutil.qinv(qoff) ## (... 1 ND 3 3)
        obj_pair_tf:cxutil.LatentObjects = obj_pair_tf.apply_pq_z(jnp.zeros((3,),jnp.float32), qoffinv, self.rot_configs)
        # z-only
        mu_dc_AB = obj_pair_tf.pos[...,-1:] 
        for i in range(2):
            mu_dc_AB = nn.Dense(self.args.dec_base_dim//2)(mu_dc_AB) # (B, 2, 16)
            mu_dc_AB = jnp.sin(mu_dc_AB)

        z_dc_AB = obj_pair_tf.z # (B, 2, C, F, D)

        base_dim_factor = 4
        # base_dim_factor = 2
        if not self.args.use_global_representation_only:
            # aligned z is rot-invariant -> now flatten ok. flatten is better than invariant layer.
            z_dc_AB = einops.rearrange(z_dc_AB, '... f d -> ... (f d)') # (B, 2, C, F*D)
            cen_tf_AB = obj_pair_tf.dc_centers_tf
            for i in range(2):
                cen_tf_AB = nn.Dense(self.args.dec_base_dim//2)(cen_tf_AB)
                cen_tf_AB = jnp.sin(cen_tf_AB)

            z_dc_AB = jnp.concatenate([z_dc_AB, cen_tf_AB], axis=-1)
            # Flip pair for the faster cross attention
            z_dc_BA = z_dc_AB[..., ::-1,:,:]

            # z_dc_AB = nn.MultiHeadDotProductAttention(4, qkv_features=self.args.dec_base_dim*2, out_features=self.args.dec_base_dim*2, dropout_rate=0.1)(z_dc_AB, z_dc_BA, deterministic=det)
            z_dc_AB = nn.MultiHeadDotProductAttention(4, qkv_features=self.args.dec_base_dim*base_dim_factor, out_features=self.args.dec_base_dim*base_dim_factor, 
                                                    broadcast_dropout=False, dropout_rate=0.1)(z_dc_AB, z_dc_BA, deterministic=(not train)) # (B, 2, C, 4*D)
            # Cvx mean pool
            # z_dc_AB = jnp.max(z_dc_AB, axis=-2)
            z_dc_AB = jnp.mean(z_dc_AB, axis=-2) # (B, 2, 4*D)
        else:
            # aligned z is rot-invariant -> now flatten ok. flatten is better than invariant layer.
            z_dc_AB = einops.rearrange(z_dc_AB, '... f d -> ... (f d)') # (B, 2, C, F*D)
            z_dc_AB = jnp.squeeze(z_dc_AB, axis=-2) # (B, 2, F*D)

        x = jnp.concatenate([z_dc_AB, mu_dc_AB], axis=-1)
        for i in range(3):
            x = nn.Dense(self.args.dec_base_dim*base_dim_factor)(x)
            x = nn.relu(x)
            if i==0:
                skip = x
        x += skip
        x = nn.Dense(self.args.dec_base_dim*base_dim_factor)(x)
        x = nn.relu(x)

        # Inter object max pool
        x = jnp.max(x, axis=-2)
        x = nn.Dense(x.shape[-1])(x)
        x = nn.relu(x)
        x = nn.Dense(1)(x)
        x = jnp.squeeze(x, axis=-1)
        # x = nn.sigmoid(x)
        return x



class PlnPredictor(nn.Module):
    args:typing.NamedTuple
    rot_configs:typing.Sequence

    @nn.compact
    def __call__(self, objects:cxutil.LatentObjects, pln_p=jnp.zeros((3)), pln_n=jnp.array([0,0,1.]), train=False):
        '''
        objects CvxObjects
        '''
        bc_shape = jnp.broadcast_shapes(pln_p.shape, pln_n.shape)
        pln_p = jnp.broadcast_to(pln_p, bc_shape)
        pln_n = jnp.broadcast_to(pln_n, bc_shape)

        if hasattr(self.args, 'train_pln_dec') and not self.args.train_pln_dec:
            pnts_dif = objects.pcd_tf[...,None,:,:] - pln_p[...,None,:] # (... NO NP NPCD 3)
            pln_components = jnp.einsum('...ijkl,...jl->...ijk', pnts_dif, pln_n)
            return 100*jnp.min(pln_components, axis=-1)

        objects = objects.drop_gt_info()


        objects_rt:cxutil.LatentObjects = jax.tree_util.tree_map(lambda x: jnp.expand_dims(x, axis=len(objects.outer_shape)), objects)
        if (not self.args.use_global_representation_only) and self.args.reduce_elements:
            new_z = evl.QueryElements(self.args, self.rot_configs)(pln_n, objects.z)
            objects_rt = objects_rt.replace(z = new_z)
        objects_rt = objects_rt.apply_pq_z(*tutil.pq_inv(pln_p, tutil.line2q(pln_n)), self.rot_configs) # (... NQ ...)
        z, c = objects_rt.z_flat, objects_rt.pos
        if not train:
            c_norm = c[...,2:]
            c = jnp.where(c_norm>0.7, c/c_norm*0.7, c)
            objects_rt = objects_rt.replace(pos=c)
        c = c[...,None,2:]
        for _ in range(2):
            c = nn.Dense(16)(c)
            c = jnp.sin(c)
        z = jnp.c_[jnp.broadcast_to(c, jnp.broadcast_shapes(c.shape, z[...,:1].shape)), z]

        for i in range(3):
            z = nn.Dense(self.args.dec_base_dim)(z)
            z = nn.relu(z)
            if i == 1:
                z = jnp.max(z, axis=-2)
        
        z = nn.Dense(1)(z) # (... NQ ND 1)
        z = jnp.squeeze(z, axis=-1)
        if pln_p.ndim==1:
            return jnp.squeeze(z, -2)
        else:
            return z


class RayPredictor(nn.Module):
    args:typing.NamedTuple
    rot_configs:typing.Sequence
    @nn.compact
    def __call__(self, objects:cxutil.LatentObjects, rpnts, rdir, feature_out=False, train=False):
        '''
        objects CvxObjects
        '''

        # object reorientation
        rdir = tutil.normalize(rdir)
        rpnts = rpnts - objects.pos[...,None,:]
        rpnts_proj_offset = jnp.einsum('...i,...i',rpnts,rdir)
        rpnts = rpnts-rpnts_proj_offset[...,None]*rdir
        if not train:
            rpnts_norm = jnp.linalg.norm(rpnts, axis=-1, keepdims=True)
            out_mask = rpnts_norm>0.7
            rpnts = jnp.where(out_mask, rpnts/rpnts_norm*0.7, rpnts)
            # out_mask = rpnts_norm>0.22
            # rpnts = jnp.where(out_mask, rpnts/rpnts_norm*0.22, rpnts)
        objects_ = objects.translate(-objects.pos).drop_gt_info()
        rdir = jnp.broadcast_to(rdir, jnp.broadcast_shapes(rdir.shape, rpnts.shape))


        if self.args.ray_dec_type == 0:
            rpnts_dir = jnp.where(jnp.sum(jnp.abs(rpnts), -1, keepdims=True) > 1e-6, tutil.normalize(rpnts), tutil.normalize(jnp.cross(jnp.array([0.123,-0.156, 0.865]), rdir)))
            Rm = jnp.stack([rpnts_dir, jnp.cross(rdir, rpnts_dir), rdir], axis=-1)
            Rm = tutil.normalize(Rm)
            rot_q = tutil.Rm2q(Rm)
            align_q = tutil.qinv(rot_q)
            
            objects_rt:cxutil.LatentObjects = jax.tree_util.tree_map(lambda x: jnp.expand_dims(x, axis=len(objects_.outer_shape)), objects_)
            if 'use_global_representation_only' not in self.args or ((not self.args.use_global_representation_only) and self.args.reduce_elements):
                new_z = evl.QueryElements(self.args, self.rot_configs)(rpnts, objects_.z)
                objects_rt = objects_rt.replace(z = new_z)
            objects_rt:cxutil.LatentObjects = objects_rt.rotate_z(align_q, self.rot_configs) # (... NQ ...)
            rpnts = tutil.qaction(align_q, rpnts)
            r = rpnts[..., :1]
            for _ in range(2):
                r = nn.Dense(16)(r)
                r = jnp.sin(r)
            r = r[...,None,:]
            bc_sp = jnp.broadcast_shapes(r[...,:1].shape, objects_rt.z_flat[...,:1].shape)
            z = jnp.c_[jnp.broadcast_to(r, jnp.broadcast_shapes(r.shape, bc_sp)), jnp.broadcast_to(objects_rt.z_flat, jnp.broadcast_shapes(objects_rt.z_flat.shape, bc_sp))]
            for i in range(3):
                z = nn.Dense(self.args.dec_base_dim)(z)
                z = nn.relu(z)
                if i==1:
                    z = jnp.max(z, axis=-2)
            
        elif self.args.ray_dec_type == 1:
            z = objects.z
            nc = objects.z.shape[-3]
            np_ = rdir.shape[-2]
            rdir_ext = evl.MakeHDFeature(self.args, self.rot_configs)(rdir[...,None]).squeeze(-1)
            rpnts_ext = evl.MakeHDFeature(self.args, self.rot_configs)(rpnts[...,None]).squeeze(-1)

            net_d = (rdir_ext * rdir_ext).sum(-1, keepdims=True) # (... P 1)
            net_p = (rpnts_ext * rpnts_ext).sum(-1, keepdims=True) # (... P 1)

            net_zd = jnp.einsum('...pf,...cfd->...pcd', rdir_ext, z) # (... P C D)
            net_zp = jnp.einsum('...pf,...cfd->...pcd', rpnts_ext, z) # (... P C D)

            z_dir = nn.Dense(z.shape[-1], use_bias=False)(z)
            z_inv = (z * z_dir).sum(-2) # (... C D)
            z_inv = einops.repeat(z_inv, '... c b -> ... r c b', r=np_)
            net_d = einops.repeat(net_d, '... b -> ... r b', r=nc)
            net_p = einops.repeat(net_p, '... b -> ... r b', r=nc)
            net = jnp.concatenate([net_d, net_p, net_zd, net_zp, z_inv], axis=-1) # (... P C D)

            net = nn.Dense(self.args.dec_base_dim)(net)
            activation = nn.relu
            for i in range(5):
                x_s = net
                net = nn.Dense(self.args.dec_base_dim)(activation(net))
                net = nn.Dense(self.args.dec_base_dim)(activation(net)) + x_s
                if i==0:
                    net = jnp.max(net, axis=-2) # remove C dimensions
            z = activation(net)
        if feature_out:
            return z
        z = nn.Dense(5)(z) # (... NQ ND 4)
        segment = z[...,:1]
        depth = z[...,1:2]-rpnts_proj_offset[...,None]
        normals = z[...,2:]
        if not train:
            segment = jnp.where(out_mask, -1e3, segment)
        if self.args.ray_dec_type == 0:
            normals = tutil.qaction(rot_q, normals)
        return segment.squeeze(-1), depth, normals

# auxiliary decoders

def import_module_from_full_path(module_path):
    spec = importlib.util.spec_from_file_location(os.path.basename(module_path)[:-3], module_path)
    module = importlib.util.module_from_spec(spec)
    module_name = os.path.basename(module_path)[:-3]
    sys.modules[module_name] = module   # Pickle will crash without this.
    spec.loader.exec_module(module)
    return module

def get_encdec_models(args, rot_configs, relative_checkpoint_dir=None):

    # if relative_checkpoint_dir is not None:
        # global cxutil
        # global eutil
        # global evl

        # cxutil = import_module_from_full_path(os.path.join(relative_checkpoint_dir, 'util/cvx_util.py'))
        # eutil = import_module_from_full_path(os.path.join(relative_checkpoint_dir, 'util/ev_util/ev_util.py'))
        # evl = import_module_from_full_path(os.path.join(relative_checkpoint_dir, 'util/ev_util/ev_layers.py'))

    enc = VSONCvxEncoder(args, rot_configs)
    dec = VSONDecoder(args, rot_configs)

    return enc, dec, ColDecoder(args, rot_configs), PlnPredictor(args, rot_configs), RayPredictor(args, rot_configs)

# def get_models_from_cp_dir(cp_dir)->'Models':
#     mutil = import_module_from_full_path(os.path.join(cp_dir, 'util/model_util.py'))
#     return mutil.Models().load_model(cp_dir)
#     # return Models().load_model(cp_dir)


def get_models_from_cp_dir(cp_dir, itr_no=1450)->'Models':
    print(f'loading models from cp_dir: {cp_dir}')

    global dmutil
    dmutil = import_module_from_full_path(os.path.join(cp_dir, 'dif_model_util.py'))
    with open(os.path.join(cp_dir, 'saved.pkl'), 'rb') as f:
        loaded = pickle.load(f)
    
    if cp_dir.split('/')[-1] in ['04042024-135303']:
        # with open(os.path.join('checkpoints/pretraining/01152024-074410', 'saved.pkl'), 'rb') as f:
        #     models = pickle.load(f)['models']
        with open(os.path.join('logs_dif/04022024-110148_2', 'saved.pkl'), 'rb') as f:
            models = pickle.load(f)['models']
    else:
        models = loaded['models']

    models_dict = {1:dmutil.DenoisingModelV4, 4:dmutil.DenoisingModelV4, 5:dmutil.DenoisingModelV4, 6:dmutil.DenoisingModelV4}
    # if cp_dir.split('/')[-1] in ['04022024-110148_2', '04022024-110148']:
    #     loaded['dif_args'].use_p = 0
    models = models.replace(denoiser_model=models_dict[loaded['dif_args'].dif_model_version](loaded['dif_args'], models.args, models.rot_configs))
    if itr_no is None or itr_no == 0:
        itr_no = ''
    with open(os.path.join(cp_dir, f'saved{itr_no}.pkl'), 'rb') as f:
        loaded1450 = pickle.load(f)
    models = models.set_params(loaded1450['ema_params'])
    return models

if __name__ == '__main__':
    np.random.seed(0)
    cond_feat = cutil.default_cond_feat(pixel_size=[32,32])
    cond_feat = cond_feat.replace(img_feat=np.arange(32*32*2).reshape(1,32,32,2).astype(jnp.float32))
    dc_centers_tf = np.random.uniform(-1,1,size=(10000,4,3))
    res = extract_pixel_features(dc_centers_tf, cond_feat)

    # grad = jax.grad(lambda x: jnp.sum(extract_pixel_features(*x)[0]))((dc_centers_tf, cond_feat))

    print(1)