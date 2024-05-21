import jax.numpy as jnp
import jax
import numpy as np
import os, sys
import einops
import optax
from functools import partial
BASEDIR = os.path.dirname(os.path.dirname(__file__))
if BASEDIR not in sys.path:
    sys.path.insert(0, BASEDIR)

import util.cvx_util as cxutil
import util.ev_util.rotm_util as rmutil

class EdmParams:
    sigma_max=80
    sigma_min=0.002
    sigma_data=1.0
    rho=7
    P_mean = -1.2
    P_std = 1.2
    # P_mean = 1.0
    # P_std = 2.0

EDMP = EdmParams()

def get_sigma(t, dm_type, s=0.008):
    '''
    ddpm: t:[0,1] -> 0-small noise -> 1-large noise
    edm: t:[0,inf) -> 0-small noise -> inf-large noise
    '''
    if dm_type == 'ddpm':
        return jnp.sqrt(1-jnp.cos((t + s) / (1 + s) * np.pi * 0.5) ** 2)
    elif dm_type == 'ddpm_noise':
        return jnp.sqrt(1-jnp.cos((t + s) / (1 + s) * np.pi * 0.5) ** 2).clip(1e-5, 1-1e-5)
    elif dm_type == 'edm':
        return t

def forward_process(x, t, jkey, dm_type, noise=None, tp=None):
    if noise is None:
        noise = jax.random.normal(jkey, shape=x.shape)
        _, jkey = jax.random.split(jkey)
    if dm_type == 'ddpm' or dm_type == 'ddpm_noise':
        sigma = get_sigma(t, dm_type=dm_type)
        for _ in range(x.ndim - sigma.ndim):
            sigma = sigma[..., None]
        if tp is not None:
            noise2 = jax.random.normal(jkey, shape=x.shape)
            sigma2 = get_sigma(tp, dm_type=dm_type)
            for _ in range(x.ndim - sigma2.ndim):
                sigma2 = sigma2[..., None]
            sigma2 = sigma/sigma2*jnp.sqrt(1-(1-sigma2**2)/(1-sigma**2))
            return x*jnp.sqrt(1-sigma**2) + jnp.sqrt(sigma**2-sigma2**2) * noise + sigma2*noise2
        else:
            return x * jnp.sqrt(1-sigma**2) + sigma * noise
    elif dm_type == 'edm':
        for _ in range(x.ndim - t.ndim):
            t = t[..., None]
        return x  + noise * t


def noise_FER_projection(noise, latent_shape, rot_configs):
    if 'noise_projection' in rot_configs and rot_configs['noise_projection'] == 0:
        return noise
    tmp_obj = cxutil.LatentObjects().init_h(noise, latent_shape)
    zswap = tmp_obj.z.swapaxes(-1,-2)
    z_hat = zswap[...,:3]
    z_mag = jnp.linalg.norm(z_hat, axis=-1, keepdims=True)
    z_hat = z_hat/z_mag

    feat_list = []
    for i, dl in enumerate(rot_configs['dim_list']):
        x_ = rmutil.Y_func_V2(dl, z_hat, rot_configs, normalized_input=True)
        feat_list.append(x_ * zswap[...,:,3+i:4+i])
    feat = jnp.concatenate(feat_list, axis=-1).swapaxes(-1,-2)

    tmp_obj = tmp_obj.replace(z=feat)
    return tmp_obj.h


def forward_process_obj(x:cxutil.LatentObjects, t:jnp.ndarray, jkey, dm_type, noise=None, deterministic_mask=0, rot_configs=None):
    h = x.h
    if noise is None:
        noise = jax.random.normal(jkey, shape=h.shape)
        noise = noise_FER_projection(noise, x.latent_shape, rot_configs)
        _, jkey = jax.random.split(jkey)
    if dm_type == 'ddpm':
        sigma = get_sigma(t, dm_type=dm_type)
        for _ in range(h.ndim - sigma.ndim):
            sigma = sigma[..., None]
        
        noise2 = jax.random.normal(jkey, shape=h.shape)
        _, jkey = jax.random.split(jkey)
        noise2 = noise_FER_projection(noise2, x.latent_shape, rot_configs)
        sigma2 = jnp.sqrt(sigma**2 * 0.02)
        deterministic_mask = jnp.array(deterministic_mask)
        for _ in range(h.ndim - deterministic_mask.ndim):
            deterministic_mask = deterministic_mask[..., None]
        sigma2 = jnp.where(deterministic_mask, 0, sigma2)
        return x.set_h(h*jnp.sqrt(1-sigma**2) + jnp.sqrt(sigma**2-sigma2**2) * noise + sigma2*noise2)
    elif dm_type == 'edm':
        for _ in range(h.ndim - t.ndim):
            t = t[..., None]
        return x.set_h(h  + noise * t)

def calculate_cs(t, edm_params:EdmParams=None, args=None):
    t = jnp.array(t)[...,None]
    if args.dm_type=='edm':
        if args.add_c_skip:
            cskip = edm_params.sigma_data**2/(t**2+edm_params.sigma_data**2)
            cout =t * edm_params.sigma_data / jnp.sqrt(edm_params.sigma_data**2 + t**2)
        else:
            cskip = jnp.zeros_like(t)
            cout = jnp.ones_like(t)
        cin = 1/jnp.sqrt(t**2 + edm_params.sigma_data**2)
        t = 1/4.*jnp.log(t)
    elif args.dm_type=='ddpm':
        if args.add_c_skip:
            sigma = get_sigma(t, dm_type=args.dm_type)
            # cskip = jnp.sqrt(1-sigma**2)
            # cout =  sigma
            cskip = jnp.ones_like(sigma)
            cout =  jnp.ones_like(sigma)
        else:
            cskip = jnp.zeros_like(t)
            cout =  jnp.ones_like(t)
        cin = jnp.ones_like(t)
    elif args.dm_type == 'ddpm_noise':
        sigma = get_sigma(t, dm_type=args.dm_type)
        cskip = -sigma/jnp.sqrt(1-sigma**2)
        cout =  1./jnp.sqrt(1-sigma**2)
        cin = jnp.ones_like(t)
    return t, (cskip, cout, cin)


def get_t_schedule_for_sampling(max_time_steps, dm_type, edm_params:EdmParams=None):
    if dm_type=='ddpm':
        return np.arange(max_time_steps,0,-1)/max_time_steps
    elif dm_type=='edm':
        nrange = np.arange(max_time_steps)/(max_time_steps-1)
        ts = (edm_params.sigma_max**(1/edm_params.rho) + nrange*(edm_params.sigma_min**(1/edm_params.rho) - edm_params.sigma_max**(1/edm_params.rho)))**(edm_params.rho)
        ts = np.concatenate([ts, [0.]], -1)
        return ts

def get_t_schedule_for_sampling_stochastic(jkey, ns, max_time_steps, dm_type, edm_params:EdmParams=None):
    return jax.random.uniform(jkey, shape=(ns, max_time_steps,)).sort(-1)[...,::-1]


def get_t_schedule_for_sampling_multisteps(jkey, ns, max_time_steps):
    timesteps = jax.random.uniform(jkey, shape=(ns, max_time_steps,)).sort(-1)[...,::-1]
    ms_timesteps = jnp.concatenate([timesteps[...,::2].at[...,0].set(1), timesteps[...,1::2].at[...,0].set(1)], axis=-1)
    return jnp.where(jax.random.uniform(jkey, shape=(ns,1))>0.5, timesteps, ms_timesteps)

def euler_sampler(x_shape, model_apply_func, params, cond, jkey, dm_type, max_time_steps=10, edm_params:EdmParams=None):
    t_schedule =get_t_schedule_for_sampling(max_time_steps, edm_params=edm_params, dm_type=dm_type)
    x = jax.random.normal(jkey, shape=x_shape) * t_schedule[...,0]
    for i in range(max_time_steps):
        if cond is not None:
            w = 1.5
            x0_pred_cond = model_apply_func(params, x, t_schedule[i], cond, jnp.array([True]))
            x0_pred_uncond = model_apply_func(params, x, t_schedule[i], cond, jnp.array([False]))
            x0_pred = w*x0_pred_cond + (1-w)*x0_pred_uncond
        else:
            x0_pred = model_apply_func(params, x, t_schedule[i])
        if i+1==max_time_steps:
            x = x0_pred
            break
        sigma = get_sigma(t_schedule[i], dm_type)
        if dm_type == 'ddpm':
            noise_pred = (x - jnp.sqrt(1-sigma**2) * x0_pred)/sigma
        elif dm_type == 'edm':
            noise_pred = (x - x0_pred)/sigma
        # noise_pred = None
        x = forward_process(x0_pred, t_schedule[i+1], jkey, noise=noise_pred, dm_type=dm_type, tp=t_schedule[i])
        _, jkey = jax.random.split(jkey)
    return x

def get_noise_pred(t, x_ptb, x0_pred, dm_type)->jnp.ndarray:
    '''
    return: noise pred in h
    '''
    sigma = get_sigma(t, dm_type)
    for _ in range(len(x_ptb.outer_shape) + 1 - sigma.ndim):
        sigma = sigma[...,None]
    if dm_type == 'ddpm':
        noise_pred = (x_ptb.h - jnp.sqrt(1-sigma**2) * x0_pred.h)/sigma
    elif dm_type == 'edm':
        noise_pred = (x_ptb.h - x0_pred.h)/sigma
    return noise_pred

def euler_sampler_obj(x_shape, base_shape, model_apply_func, cond, jkey, dm_type, in_range_func=None, 
                      deterministic=jnp.array(False), w=jnp.array(1.0), max_time_steps=10, edm_params:EdmParams=None, 
                      rot_configs=None, sequence_out=False, conf_threshold=-0.5, conf_filter_out=True, learnable_queries=None,
                      output_obj_no=None, x_previous:cxutil.LatentObjects=None, start_time_steps=0):
    t_schedule =get_t_schedule_for_sampling(max_time_steps, edm_params=edm_params, dm_type=dm_type)
    t_schedule = t_schedule[start_time_steps:]
    if learnable_queries is not None:
        x = einops.repeat(learnable_queries, 'i ... -> r i ...', r=x_shape[0])
        x = cxutil.LatentObjects().init_h(x, base_shape)
    elif x_previous is not None:
        # warm start
        if len(x_previous.outer_shape) == 1:
            x_previous = jax.tree_map(lambda x: einops.repeat(x, '... -> r ...', r=x_shape[0]), x_previous)
        if in_range_func is not None:
            in_range_mask = in_range_func(x_previous.pos)
        x = forward_process_obj(x_previous, t_schedule[...,0], jkey, noise=None, dm_type=dm_type, deterministic_mask=deterministic, rot_configs=rot_configs)
        _, jkey = jax.random.split(jkey)
        if in_range_func is not None:
            x = x.set_h(jnp.where(in_range_mask, x.h, noise_FER_projection(jax.random.normal(jkey, shape=x_shape), base_shape, rot_configs)))
            _, jkey = jax.random.split(jkey)
    else:
        x = jax.random.normal(jkey, shape=x_shape) * t_schedule[...,0,None,None]
        _, jkey = jax.random.split(jkey)
        x = noise_FER_projection(x, base_shape, rot_configs)
        x = cxutil.LatentObjects().init_h(x, base_shape)

    if x.outer_shape[0]!=cond.img_feat.shape[0] and cond.img_feat.shape[0]==1:
        cond = jax.tree_map(lambda x_: einops.repeat(x_, 'i ... -> (r i) ...', r=x.outer_shape[0]), cond)

    if sequence_out:    
        x_diffusion_list = [x]
        x_pred_list = [x]
        conf_list = []
    for i in range(t_schedule.shape[0]):
        w = jnp.array(w)
        for _ in range(x.h.ndim - w.ndim):
            w = w[...,None]
        x0_pred, conf = model_apply_func(x, cond, t_schedule[i], None, jkey)
        _, jkey = jax.random.split(jkey)
        if learnable_queries is not None:
            x = x0_pred
            if sequence_out:
                x_pred_list.append(x0_pred)
                conf_list.append(conf)
                x_diffusion_list.append(x)
            if i+1==t_schedule.shape[0]:
                x = x0_pred
                break
        else:
            if x0_pred.outer_shape[-1]!=1:
                keep_mask1 = jnp.logical_and(conf>conf_threshold-1.2, i/max_time_steps < 0.7)
                keep_mask2 = jnp.logical_and(conf>conf_threshold, i/max_time_steps >= 0.7)
                keep_mask = jnp.logical_or(keep_mask1, keep_mask2)
                keep_mask = jnp.logical_or(keep_mask, jnp.abs(w-1)>0.5)
                x0_pred = x0_pred.set_h(jnp.where(keep_mask, x0_pred.h, noise_FER_projection(jax.random.normal(jkey, shape=x_shape), base_shape, rot_configs)))
                _, jkey = jax.random.split(jkey)
            # if in_range_func is not None:
            #     x0_pred = x0_pred.set_h(jnp.where(in_range_func(x0_pred.pos), x0_pred.h, noise_FER_projection(jax.random.normal(jkey, shape=x_shape), base_shape, rot_configs)))
            _, jkey = jax.random.split(jkey)
            if sequence_out:
                x_pred_list.append(x0_pred)
                conf_list.append(conf)
            if i+1==t_schedule.shape[0]:
                x = x0_pred
                if sequence_out:
                    x_diffusion_list.append(x)
                break
            noise_pred = get_noise_pred(t_schedule[...,i], x, x0_pred, dm_type)
            # noise_pred = jnp.where(t_schedule[...,i+1][...,None,None]>0.99999, 
            #                        noise_FER_projection(jax.random.normal(jkey, shape=noise_pred.shape), base_shape, rot_configs), 
            #                        noise_pred)
            # _, jkey = jax.random.split(jkey)
            x = forward_process_obj(x0_pred, t_schedule[...,i+1], jkey, noise=noise_pred, dm_type=dm_type, deterministic_mask=deterministic, rot_configs=rot_configs)
            if sequence_out:
                x_diffusion_list.append(x)
            _, jkey = jax.random.split(jkey)
    
    # ### confidence out ###
    if x_shape[-2] != 1 and output_obj_no is not None:
        pick_obj_indices = jnp.argsort(-conf, axis=1)[:,:output_obj_no]
        def align_rank(arr, rank):
            for _ in range(rank - arr.ndim):
                arr = arr[...,None]
            return arr
        x, conf = jax.tree_map(lambda x: jnp.take_along_axis(x, align_rank(pick_obj_indices, x.ndim), axis=1), (x, conf))
    if x_shape[-2] != 1 and conf_filter_out:
        keep_mask = conf>conf_threshold
        x = x.replace(pos=jnp.where(keep_mask, x.pos, jnp.array([0,0,10.])))

    if sequence_out:
        return x, (x_diffusion_list, x_pred_list, conf_list)
    else:
        return x


def euler_sampler_obj_fori(x_shape, base_shape, model_apply_func, cond, jkey, dm_type, previous_x_pred=None, guidance_func_args=None, in_range_func=None, 
                      deterministic=jnp.array(False), w=jnp.array(1.0), max_time_steps=10, guidance_grad_step=70, edm_params:EdmParams=None, 
                      rot_configs=None, sequence_out=False, conf_threshold=-0.5, conf_filter_out=True, learnable_queries=None,
                      output_obj_no=None, gradient_guidance_func=None, start_time_step=None):
    t_schedule =get_t_schedule_for_sampling(max_time_steps, edm_params=edm_params, dm_type=dm_type)
    t_schedule = jnp.array(t_schedule).astype(jnp.float32)
    if learnable_queries is not None:
        x = einops.repeat(learnable_queries, 'i ... -> r i ...', r=x_shape[0])
    else:
        if previous_x_pred is not None:
            assert start_time_step is not None
            x = jax.random.normal(jkey, shape=x_shape) * t_schedule[...,start_time_step,None,None]
            _, jkey = jax.random.split(jkey)
            x = noise_FER_projection(x, base_shape, rot_configs)
            valid_obj_mask = jnp.all(jnp.abs(previous_x_pred.pos)< 8.0, axis=-1, keepdims=True) 
            x = jnp.where(valid_obj_mask, previous_x_pred.h, 0) + x
        else:
            x = jax.random.normal(jkey, shape=x_shape) * t_schedule[...,0,None,None]
            _, jkey = jax.random.split(jkey)
            x = noise_FER_projection(x, base_shape, rot_configs)
    x = cxutil.LatentObjects().init_h(x, base_shape)

    if x.outer_shape[0]!=cond.img_feat.shape[0] and cond.img_feat.shape[0]==1:
        cond = jax.tree_map(lambda x_: einops.repeat(x_, 'i ... -> (r i) ...', r=x.outer_shape[0]), cond)

    w = jnp.array(w)
    for _ in range(x.h.ndim - w.ndim):
        w = w[...,None]
    opt_state = None
    if gradient_guidance_func is not None:
        # optimizer = optax.adamw(2e-2)
        optimizer = optax.adam(4e-3)
        opt_state = optimizer.init(x.h)
    def f_(carry, i, guidance=False):
        x,opt_state,jkey = carry
        x0_pred, conf = model_apply_func(x, cond, t_schedule[i], None, jkey)
        _, jkey = jax.random.split(jkey)
        if learnable_queries is not None:
            x = x0_pred
        else:
            if x0_pred.outer_shape[-1]!=1:
                keep_mask1 = jnp.logical_and(conf>conf_threshold-1.2, i/max_time_steps < 0.7)
                keep_mask2 = jnp.logical_and(conf>conf_threshold, i/max_time_steps >= 0.7)
                keep_mask = jnp.logical_or(keep_mask1, keep_mask2)
                keep_mask = jnp.logical_or(keep_mask, jnp.abs(w-1)>0.5)
                if guidance and gradient_guidance_func is not None:
                    # optimizer = optax.adam(4e-3)
                    # opt_state = optimizer.init(x.h)
                    valid_x0_pred = x0_pred.replace(pos=jnp.where(keep_mask, x0_pred.pos, jnp.array([0,0,10.0])))
                    updated_h = valid_x0_pred.h
                    for _ in range(guidance_grad_step):
                        grad, _ = gradient_guidance_func(updated_h, *guidance_func_args)
                        updates, opt_state = optimizer.update(grad, opt_state, updated_h)
                        updated_h = optax.apply_updates(updated_h, updates)
                else:
                    updated_h =x0_pred.h
                x0_pred = x0_pred.set_h(jnp.where(keep_mask, updated_h, noise_FER_projection(jax.random.normal(jkey, shape=x_shape), base_shape, rot_configs)))
                _, jkey = jax.random.split(jkey)
            _, jkey = jax.random.split(jkey)
            noise_pred = get_noise_pred(t_schedule[...,i], x, x0_pred, dm_type)
            x = forward_process_obj(x0_pred, t_schedule[...,i+1], jkey, noise=noise_pred, dm_type=dm_type, deterministic_mask=deterministic, rot_configs=rot_configs)
            _, jkey = jax.random.split(jkey)
        return (x,opt_state,jkey), (x0_pred, x, conf)
    
    if start_time_step is not None:
        if gradient_guidance_func is not None:
            (x, _, jkey), (x_pred_list, x_diffusion_list, conf_list) = jax.lax.scan(f_, (x,opt_state,jkey), jnp.arange(start_time_step, max_time_steps-2))
            (x, _, jkey), (x_pred_list, x_diffusion_list, conf_list) = jax.lax.scan(partial(f_, guidance=True), (x,opt_state,jkey), jnp.arange(max_time_steps-2, max_time_steps))
        else:
            (x, _, jkey), (x_pred_list, x_diffusion_list, conf_list) = jax.lax.scan(f_, (x,opt_state,jkey), jnp.arange(max_time_steps))
    else:
        if gradient_guidance_func is not None:
            (x, _, jkey), (x_pred_list, x_diffusion_list, conf_list) = jax.lax.scan(f_, (x,opt_state,jkey), jnp.arange(max_time_steps-4))
            (x, _, jkey), (x_pred_list, x_diffusion_list, conf_list) = jax.lax.scan(partial(f_, guidance=True), (x,opt_state,jkey), jnp.arange(max_time_steps-4, max_time_steps))
        else:
            (x, _, jkey), (x_pred_list, x_diffusion_list, conf_list) = jax.lax.scan(f_, (x,opt_state,jkey), jnp.arange(max_time_steps))
    x = x_pred_list[-1] # the last prediction is from zero prediction
    
    # ### confidence out ###
    if x_shape[-2] != 1 and output_obj_no is not None:
        pick_obj_indices = jnp.argsort(-conf, axis=1)[:,:output_obj_no]
        def align_rank(arr, rank):
            for _ in range(rank - arr.ndim):
                arr = arr[...,None]
            return arr
        x, conf = jax.tree_map(lambda x: jnp.take_along_axis(x, align_rank(pick_obj_indices, x.ndim), axis=1), (x, conf))
    if x_shape[-2] != 1 and conf_filter_out:
        keep_mask = conf>conf_threshold
        x = x.replace(pos=jnp.where(keep_mask, x.pos, jnp.array([0,0,10.])))

    if sequence_out:
        return x, (x_diffusion_list, x_pred_list, conf_list)
    else:
        return x




def perturb_recover_obj(obj:cxutil.LatentObjects, model_apply_jit, cond, t, jkey, dm_type, rot_configs):
    extended = False
    if len(obj.outer_shape) == 0:
        extended = True
        obj = obj.extend_outer_shape(0)
    obj_ptb = forward_process_obj(obj, t, jkey, dm_type=dm_type, rot_configs=rot_configs)
    _, jkey = jax.random.split(jkey)
    obj_rec, conf = model_apply_jit(obj_ptb, cond, t, jnp.array([True]), jkey)
    if extended:
        obj_rec = jax.tree_map(lambda x: x.squeeze(0), obj_rec)
    return obj_rec

# %%

def sample_t_train(jkey, shape, edm_params:EdmParams=None, dm_type='edm', add_t_sample_bias=0):
    if dm_type == 'ddpm' or dm_type == 'ddpm_noise':
        t_samples = jax.random.uniform(jkey, shape=shape)
        if add_t_sample_bias!=0:
            t_samples = jnp.sqrt(1-jnp.cos(t_samples * np.pi * 0.5) ** add_t_sample_bias)
    elif dm_type == 'edm':
        log_t = jax.random.normal(jkey, shape=shape) * edm_params.P_std + edm_params.P_mean
        t_samples = jnp.exp(log_t)
        t_samples = t_samples.clip(0, edm_params.sigma_max)
        # t_samples = edm_params.sigma_min + jax.random.uniform(jkey, shape=shape) * (edm_params.sigma_max - edm_params.sigma_min)
    return t_samples


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    jkey = jax.random.PRNGKey(0)


    sigma2 = get_sigma(jnp.ones(1000)*0.5, dm_type='ddpm').clip(1e-5, 1-1e-5)
    sigma = get_sigma(jnp.linspace(0,0.5,1000), dm_type='ddpm')
    sigma2 = sigma/sigma2*jnp.sqrt(1-(1-sigma2**2)/(1-sigma**2))
    
    plt.figure()
    plt.plot(sigma)
    plt.plot(sigma2)
    plt.show()

    # t_sp = sample_t_train(jkey, (50000,), EDMP, dm_type='edm')
    t_sp = sample_t_train(jkey, (50000,), EDMP, dm_type='ddpm', add_t_sample_bias=10)
    # sigma = get_sigma(t_sp, dm_type='ddpm')
    # plt.figure()
    # plt.hist(t_sp, bins=1000)
    # plt.show()

    # # sigma = get_sigma(np.arange(100)/100, dm_type='ddpm')
    # sigma = jnp.sqrt(1-jnp.cos(np.arange(100)/100 * np.pi * 0.5) ** 2.5)
    # plt.figure()
    # plt.plot(sigma)
    # plt.show()

    class Args:
        pass
    args = Args()
    args.dm_type = 'ddpm_noise'
    args.add_c_skip = 0

    t_schedule =get_t_schedule_for_sampling(1000, edm_params=EDMP, dm_type='ddpm_noise')
    time, (c_skip, c_out, c_in) = calculate_cs(t_schedule, EDMP, args)

    plt.figure()
    # plt.plot(time)
    plt.plot(c_skip)
    plt.plot(c_out)
    plt.plot(c_in)
    plt.legend(['c_skip', 'c_out', 'c_in'])
    plt.show()