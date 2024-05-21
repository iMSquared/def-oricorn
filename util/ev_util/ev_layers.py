import flax.linen as nn
import typing
import os, sys

if __name__=='__main__':
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from util.ev_util.ev_util import *
from util.ev_util.rotm_util import Y_func_V2

EPS = 1e-6

class MakeHDFeature(nn.Module):
    args:typing.NamedTuple
    rot_configs:typing.Sequence
    siren:bool = False

    @nn.compact
    def __call__(self, x:jnp.ndarray, feat_axis=-2):
        x_norm = jnp.linalg.norm(x, axis=feat_axis, keepdims=True)
        zero_norm = x_norm<1e-5
        x_norm += EPS
        x_hat = x/x_norm
        ft_list = []
        if self.args.psi_scale_type == 0 or self.args.psi_scale_type == 1:
            scale_params = self.param('scale_params_psi', nn.initializers.ones_init(),
                                    (len(self.rot_configs['dim_list']),))
        for i, dl in enumerate(self.rot_configs['dim_list']):
            x_ = Y_func_V2(dl, x_hat.swapaxes(-1,feat_axis), self.rot_configs, normalized_input=True).swapaxes(-1,feat_axis)
            x_ = jnp.where(zero_norm, 0., x_)
            if self.args.psi_scale_type == 0:
                x_norm_ = x_norm * scale_params[i]
            elif self.args.psi_scale_type == 1:
                x_norm_ = x_norm**dl * scale_params[i]
            elif self.args.psi_scale_type == 2:
                origin_size = x_norm.shape[-1]
                x_norm_ = nn.Dense(16)(x_norm)
                x = jnp.sin(x) if self.siren else nn.relu(x)
                x_norm_ = nn.Dense(16)(x_norm_)
                x = jnp.sin(x) if self.siren else nn.relu(x)
                x_norm_ = nn.Dense(origin_size)(x_norm_)
            # x_ = x_*x_norm_
            x_ = jnp.where(zero_norm, x_, x_*x_norm_)
            ft_list.append(x_)
        feat = jnp.concatenate(ft_list, -2)
        return feat


class EVNNonLinearity(nn.Module):
    args:typing.NamedTuple
    
    @nn.compact
    def __call__(self, x):
        '''
        x: point features of shape [B, D, N, ...]
        '''
        norm = jnp.linalg.norm(x, axis=-2, keepdims=True)
        zero_norm = norm<=EPS
        norm += EPS
        nf = norm.shape[-1]
        norm_bn = nn.Dense(nf//2)(norm)
        # TODO: deprecate siren forever
        norm_bn = jnp.sin(norm_bn) if ("siren" in self.args and self.args.siren) else nn.leaky_relu(norm_bn, self.args.negative_slope)
        if self.args.skip_connection==0:
            norm_bn = nn.Dense(nf)(norm_bn)
            x = jnp.where(zero_norm, x, x / norm * norm_bn)
        elif self.args.skip_connection==1:
            norm_bn = nn.Dense(nf)(norm_bn) + norm
            x = jnp.where(zero_norm, x, x / norm * norm_bn)
        elif self.args.skip_connection==2:
            norm_bn = nn.Dense(nf)(norm_bn)
            norm_bn = 2*nn.sigmoid(norm_bn)
            x = x * norm_bn
        return x

        
class EVLinearLeakyReLU(nn.Module):
    args:typing.NamedTuple
    out_channels:int
    
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.out_channels, use_bias=False)(x)
        x = EVNNonLinearity(self.args)(x)
        return x
    

class EVNStdFeature(nn.Module):
    args:typing.NamedTuple
    out_feat_dim:int
    
    @nn.compact
    def __call__(self, x):
        '''
        x: point features of shape [B, N_feat, 3, N_samples, ...]
        '''
        z0 = x
        nf = z0.shape[-1]
        z0 = EVLinearLeakyReLU(nf//2, self.args)(z0)
        z0 = EVLinearLeakyReLU(nf//4, self.args)(z0)
        z0 = nn.Dense(self.out_feat_dim, use_bias=False)(z0)
        return jnp.einsum('...ji,...jk->...ik', z0, x)


class EVSTNkd(nn.Module):
    args:typing.NamedTuple
    base_dim:int

    @nn.compact
    def __call__(self, x):
        nd = x.shape[-1]
        x = EVLinearLeakyReLU(self.args, self.base_dim)(x)
        x = EVLinearLeakyReLU(self.args, self.base_dim*2)(x)
        x = EVLinearLeakyReLU(self.args, self.base_dim*4)(x)
        x = jnp.mean(x, -3)

        x = EVLinearLeakyReLU(self.args, self.base_dim*2)(x)
        x = EVLinearLeakyReLU(self.args, self.base_dim)(x)
        x = EVLinearLeakyReLU(self.args, nd)(x)
        
        return x


def safe_norm(x, axis, keepdims=False, eps=0.0):
    is_zero = jnp.all(jnp.isclose(x,0.), axis=axis, keepdims=True)
    # temporarily swap x with ones if is_zero, then swap back
    x = jnp.where(is_zero, jnp.ones_like(x), x)
    n = jnp.linalg.norm(x, axis=axis, keepdims=keepdims)
    n = jnp.where(is_zero if keepdims else jnp.squeeze(is_zero, -1), 0., n)
    return n.clip(eps)

    
class AttnDropout(nn.Module):
    args:typing.NamedTuple
    
    def __call__(self, attn, jkey, det=True):
        if det or jkey is None:
            return attn
        keep_prob = 1.0 - self.args.dropout
        keep = jax.random.bernoulli(jkey, keep_prob, attn.shape)  # type: ignore
        multiplier = keep.astype(jnp.float32) / jnp.asarray(keep_prob, dtype=jnp.float32)
        attn = attn * multiplier
        return attn

class EVLayerNorm(nn.Module):
    args:typing.NamedTuple

    def __call__(self, x):
        x_norm = jnp.linalg.norm(x, axis=-2, keepdims=True)
        sigma = jnp.std(jnp.concatenate([x_norm, -x_norm], -1), axis=-1, keepdims=True)
        return x/(sigma+EPS)

class CrossAttention(nn.Module):
    args:typing.NamedTuple
    qk_dim:int
    v_dim:int
    multi_head:int
    attn_type:str='dot'
    
    @nn.compact
    def __call__(self, qfts, kvfts, jkey=None, det=True):
        '''
        fts: (b, c, 3, npoint)
        xyz: (b, npoint, 3)
        '''
        qfts = EVLinearLeakyReLU(self.args, self.qk_dim)(qfts)
        kvfts = EVLinearLeakyReLU(self.args, self.qk_dim)(kvfts)

        q = nn.DenseGeneral(axis=-1, features=(self.multi_head, self.qk_dim//self.multi_head), use_bias=False)(qfts)
        k = nn.DenseGeneral(axis=-1, features=(self.multi_head, self.qk_dim//self.multi_head), use_bias=False)(kvfts)
        v = nn.DenseGeneral(axis=-1, features=(self.multi_head, self.v_dim//self.multi_head), use_bias=False)(kvfts)
        q,k,v = map(lambda x: jnp.moveaxis(x, -2, 0), (q,k,v))
        
        if self.args.normalize_qk:
            q = EVLayerNorm(self.args)(q)
            k = EVLayerNorm(self.args)(k)
        
        if self.attn_type == 'sub':
            scale = np.sqrt(k.shape[-1])
            attn = q[...,None,:,:]-k[...,None,:,:,:]
            attn = jnp.einsum('...fd,...f', attn, jnp.mean(attn, axis=-1))
            attn = nn.Dense(1)(attn).squeeze(-1)
            attn = nn.softmax(attn/scale, axis=-1)
        elif self.attn_type == 'dot':
            scale = np.sqrt(k.shape[-1])
            attn = jnp.einsum('...qij,...kij->...qk', q, k) # (b n n)
            attn = nn.softmax(attn/scale, axis=-1)
        elif self.attn_type == 'slot':
            attn = jnp.einsum('...qij,...kij->...qk', q, k) # (b n n)
            attn = nn.softmax(attn, axis=-2)
            attn = safe_norm(attn, -2, keepdims=True)
        
        if self.args.dropout > 0.0 and not det:
            attn = AttnDropout(self.args)(attn, jkey, det=det)

        resi = jnp.einsum('...qk,...kfd->...qfd', attn, v)
        resi = einops.rearrange(resi, 'h ... f d -> ... f (h d)')
        resi = EVLinearLeakyReLU(self.args, self.v_dim)(resi)

        return resi
        
# Resnet Blocks
class EVNResnetBlockFC(nn.Module):
    args:typing.NamedTuple
    size_h:int
    size_out:int

    @nn.compact
    def __call__(self, x):
        size_in = x.shape[-1]
        net = EVNNonLinearity(self.args)(x)
        net = nn.Dense(self.size_h, use_bias=False)(net)
        dx = EVNNonLinearity(self.args)(net)
        dx = nn.Dense(self.size_out, use_bias=False)(dx)
        if size_in == self.size_out:
            x_s = x
        else:
            x_s = nn.Dense(self.size_out, use_bias=False)(x)
        return x_s + dx



class InvCrossAttention(nn.Module):
    args:typing.NamedTuple
    qk_dim:int
    v_dim:int
    query_size:int
    multi_head:int
    
    @nn.compact
    def __call__(self, x, jkey=None, det=True):
        '''
        x: (... F D)
        '''
        learnable_query = self.param('learnable_query', nn.initializers.lecun_normal(),
                                          (self.query_size, self.qk_dim))
        kvfts = EVLinearLeakyReLU(self.args, self.qk_dim)(x)
        kvfts_inv = jnp.einsum('...fd,...f', kvfts, jnp.mean(kvfts, axis=-1))

        q = nn.DenseGeneral(axis=-1, features=(self.multi_head, self.qk_dim//self.multi_head), use_bias=False)(learnable_query)
        k = nn.DenseGeneral(axis=-1, features=(self.multi_head, self.qk_dim//self.multi_head), use_bias=False)(kvfts_inv)
        v = nn.DenseGeneral(axis=-1, features=(self.multi_head, self.v_dim//self.multi_head), use_bias=False)(kvfts)
        q,k,v = map(lambda x: jnp.moveaxis(x, -2, 0), (q,k,v))
        
        if self.args.normalize_qk:
            q = nn.LayerNorm()(q)
            k = nn.LayerNorm()(k)

        scale = np.sqrt(k.shape[-1])
        attn = jnp.einsum('...qi,...bki->...bqk', q, k) # (b n n)
        attn = nn.softmax(attn/scale, axis=-1)
        
        if self.args.dropout > 0.0 and not det:
            attn = AttnDropout(self.args)(attn, jkey, det=det)

        resi = jnp.einsum('...qk,...kfd->...qfd', attn, v)
        resi = einops.rearrange(resi, 'h ... f d -> ... f (h d)')
        resi = EVLinearLeakyReLU(self.args, self.v_dim)(resi)

        return resi



class QueryElements(nn.Module):
    args:typing.NamedTuple
    rot_configs:typing.Sequence=None
    
    @nn.compact
    def __call__(self, qpnts, kvfts):
        '''
        qpnts: (... q f)
        kvfts: (... c f d)
        '''
        if qpnts.shape[-1] == 3:
            assert self.rot_configs is not None
            qpnts = MakeHDFeature(self.args, self.rot_configs)(qpnts[...,None]).squeeze(-1)
        scale = np.sqrt(kvfts.shape[-1])
        attn = jnp.einsum('...qf,...cfd->...qc', qpnts, kvfts) # (b n n)
        attn = nn.softmax(attn/scale, axis=-1)
        resi = jnp.einsum('...qc,...cfd->...qfd', attn, kvfts)
        resi = resi[...,None,:,:]
        return resi



if __name__ == '__main__':
    class Args:
        pass
    
    args = Args()
    args.feat_dim=8
    args.psi_scale_type=0
    args.negative_slope=0.0
    args.skip_connection=1
    args.normalize_qk = 1
    args.dropout = 0.0

    import ev_utils.rotm_util as rmutil
    import util.transform_util as trutil

    rot_configs = rmutil.init_rot_config(0, [1, 2], rot_type='wigner')

    jkey = jax.random.PRNGKey(0)
    x = jax.random.normal(jkey, shape=(10, 3, 8, 2))
    randR = rmutil.rand_matrix(10)

    model = InvCrossAttention(args, 16, 16, 4, 2)
    outputs, params = model.init_with_output(jkey, x)


    # model = EVLinearLeakyReLU(args, 4)
    # outputs, params = model.init_with_output(jkey, x)

    res = model.apply(params, x)
    resrot = rmutil.apply_rot(res, randR[...,None,:,:], rot_configs, -2)

    rotx = rmutil.apply_rot(x, randR[...,None,:,:], rot_configs, -2)
    res2 = model.apply(params, rotx)

    make_feature = MakeHDFeature(args, rot_configs)

    jkey = jax.random.PRNGKey(0)

    outputs, params = make_feature.init_with_output(jkey, x)

    randR = rmutil.rand_matrix(10)
    res = make_feature.apply(params, x)
    resrot = rmutil.apply_rot(res, randR[...,None,:,:], rot_configs, -2)
    rotx = jnp.einsum('...ij,...jk->...ik',randR,x)
    resrot2 = make_feature.apply(params, rotx)

    resid = jnp.abs(resrot - resrot2)

    print(1)