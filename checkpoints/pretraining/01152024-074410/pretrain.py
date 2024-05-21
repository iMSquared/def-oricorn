# %%
# import libraries
import jax.numpy as jnp
import jax
import einops
from flax import linen as nn
import glob
import numpy as np
import typing
import optax
import datetime
import os, sys
import pickle
import shutil
import time
import random

try:
    import vessl
    vessl.init()
    vessl_on = True
    print('vessl on')
except:
    vessl_on = False
    print('vessl off')

BASEDIR = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, BASEDIR)

import util.cvx_util as cxutil
import util.transform_util as tutil
import util.model_util as mutil
from data.col_datagen import PretrainingDataGen

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, Dataset

# %%
# data generation

import argparse


if __name__ == '__main__':
    if jax.device_count('gpu') == 0:
        print('no gpu found. End process')
        raise AssertionError
    else:
        print('device found: ', jax.devices())

    parser = argparse.ArgumentParser()
    parser.add_argument("--rot_train", type=str, default='aligned')
    parser.add_argument("--rot_test", type=str, default='aligned')
    parser.add_argument("--npoint", type=int, default=512)
    parser.add_argument("--batch_query_size", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--enc_base_dim", type=int, default=64)
    parser.add_argument("--dec_base_dim", type=int, default=32)
    parser.add_argument("--rot_order", type=str, default='1-2')
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--psi_scale_type", type=int, default=2)
    parser.add_argument("--negative_slope", type=float, default=0.0)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--model", type=str, default='vson')
    parser.add_argument("--rot_type", type=str, default='custom')
    parser.add_argument("--skip_connection", type=int, default=1)
    parser.add_argument("--normalize_qk", type=int, default=1)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--reduce_set_no", type=int, default=0)
    parser.add_argument("--use_global_feature_in_encoder", type=int, default=1)
    parser.add_argument("--use_local_feature_in_encoder", type=int, default=1)
    parser.add_argument("--use_global_representation_only", type=int, default=1)
    parser.add_argument("--occ_loss_coef", type=float, default=10)
    parser.add_argument("--reg_loss_coef", type=float, default=8e-3)
    parser.add_argument("--occ_dec_type", type=int, default=0)
    parser.add_argument("--ray_dec_type", type=int, default=1)
    parser.add_argument("--reduce_elements", type=int, default=1)
    parser.add_argument("--checkpoints", type=str, default=None)
    parser.add_argument("--ds_limit", type=int, default=0)
    parser.add_argument("--data_gen_devider", type=int, default=2)
    parser.add_argument("--category", type=str, default="none")
    parser.add_argument("--augment_cabinet", type=int, default="0")
    parser.add_argument("--dataset", type=str, default="NOCS")

    args = parser.parse_args()

    class OccDataDirLoader(Dataset):
        def __init__(self, dataset='NOCS', eval_type='test'):
            dataset_dir=os.path.join(BASEDIR, f'data/{dataset}Occ')
            ds_dir_list = glob.glob(os.path.join(dataset_dir, '*/*.pkl'))
            sorted(ds_dir_list)
            if eval_type=='test':
                ds_dir_list = ds_dir_list[:len(ds_dir_list)//10]
            else:
                ds_dir_list = ds_dir_list[len(ds_dir_list)//10:]
            self.ds_dir_list = ds_dir_list
            self.eval_type = eval_type
            self.ds_init()

        def ds_init(self):

            try:
                del self.entire_ds
            except:
                pass
            
            def replace_np_ele(x, y, ns, i):
                x[ns*i:ns*(i+1)] = y
                return x
            self.ds_dir_list = np.random.permutation(self.ds_dir_list)
            print('start ds loading ' + self.eval_type)
            for i, dsdir in enumerate(self.ds_dir_list):
                with open(dsdir, "rb") as f:
                    loaded = pickle.load(f)
                if i ==0:
                    ns = jax.tree_util.tree_flatten(loaded)[0][0].shape[0]
                    if self.eval_type=='test' or args.ds_limit == 0:
                        self.entire_ds = jax.tree_map(lambda x: np.concatenate([x, np.zeros_like(einops.repeat(x, 'i ... -> (r i) ...', r=len(self.ds_dir_list)-1))], 0), loaded)
                    else:
                        self.entire_ds = jax.tree_map(lambda x: np.concatenate([x, np.zeros_like(einops.repeat(x, 'i ... -> (r i) ...', r=args.ds_limit//ns-1))], 0), loaded)
                else:
                    if self.eval_type=='train' and args.ds_limit!=0 and i >= args.ds_limit//ns:
                        break
                    self.entire_ds = jax.tree_map(lambda x,y: replace_np_ele(x, y, ns, i), self.entire_ds,loaded)
            print('end ds loading ' + self.eval_type)

        def __len__(self):
            return jax.tree_util.tree_flatten(self.entire_ds)[0][0].shape[0]

        def __getitem__(self, index):

            dpnts = jax.tree_map(lambda x: x[index], self.entire_ds)
            vtx, fcs, pcd, dc_idx = dpnts[0]
            qps, occ = dpnts[2]

            idx_pcd = np.random.randint(0, pcd.shape[1], size=(args.npoint,))
            idx_qps = np.random.randint(0, qps.shape[1], size=(args.batch_query_size,))
            idx_ray_pln = np.random.randint(0, dpnts[3][0].shape[1], size=(args.batch_query_size,))

            pcd_res, dc_idx_red, qps_res, occ_red = pcd[:,idx_pcd], dc_idx[:,idx_pcd], qps[:,idx_qps], occ[:,idx_qps]
            pln_ds, ray_ds = jax.tree_map(lambda x: np.array(x[:,idx_ray_pln]).astype(np.float32), (dpnts[3], dpnts[4]))

            return (vtx.astype(np.float32), fcs.astype(np.int32), pcd_res.astype(np.float32), dc_idx_red.astype(np.int32), 
                    qps_res.astype(np.float32), occ_red.astype(np.float32), dpnts[1].astype(np.float32), *pln_ds, *ray_ds)

    # Configure online data generation
    if args.data_gen_devider != 0:
        gen_batch_size = int(args.batch_size/args.data_gen_devider)
        pre_data_gen = PretrainingDataGen(args.npoint, gen_batch_size, args.batch_query_size, 1, dataset=args.dataset, 
                                          category=args.category, augment_cabinet=args.augment_cabinet)
    else:
        gen_batch_size = 0
        pre_data_gen = None

    # Dataset loading
    # res, jkey = pre_data_gen.data_gen_one_batch(jax.random.PRNGKey(0), save=False)
    train_loader = DataLoader(OccDataDirLoader(dataset=args.dataset, eval_type='train'), batch_size=args.batch_size-gen_batch_size, shuffle=True, num_workers=8, drop_last=True)
    eval_loader = DataLoader(OccDataDirLoader(dataset=args.dataset, eval_type='test'), batch_size=args.batch_size, shuffle=False, num_workers=8, drop_last=True)

    # Init model for training
    optimizer = optax.adam(args.lr)
    if args.checkpoints is None:
        for ds in eval_loader:
            ds_sample = ds
            break
        models:mutil.Models = mutil.Models().init_model(args=args, ds=ds_sample)
        opt_state = optimizer.init(models.params)
        params = models.params
        print('init model')
    else:
        # models = mutil.Models().load_model(args.checkpoints)
        with open(os.path.join(args.checkpoints, 'saved.pkl'), 'rb') as f:
            loaded = pickle.load(f)
        models = loaded['models']
        params = loaded['params']
        opt_state = loaded['opt_state']
        print('load checkpoints from ' + args.checkpoints)
    jkey = jax.random.PRNGKey(args.seed)


    # %%
    # start training

    def BCE_loss(yp_logit, yt):
        assert yp_logit.shape == yt.shape
        yp = nn.sigmoid(yp_logit).clip(1e-5, 1-1e-5)
        loss = - yt*jnp.log(yp) - (1-yt)*jnp.log(1-yp)
        return loss
    
    def Focal_loss(yp_logit, yt):
        assert yp_logit.shape == yt.shape
        yp = nn.sigmoid(yp_logit).clip(1e-5, 1-1e-5)
        yp = yt*yp + (1-yt)*(1-yp)
        gamma = 2
        loss = -(1-yp)**gamma*jnp.log(yp)
        return loss


    def cal_loss(params, ds, jkey):
        models_ = models
        obj = cxutil.CvxObjects().init_vtx(*ds[:2]).init_pcd(*ds[2:4])
        models_:mutil.Models = models_.set_params(params)
        # 2023-12-27
        # Before: latent_obj.z.shape = [64, 2, 32, 8, 32] where [#batch, $pair, #cvx, #rot, #channel]
        # TODO: latent_obj.z.shape = [64, 2, 1, 8, #channel] 
        latent_obj = models_.apply('shape_encoder', obj, jkey, False, train=True)
        _, jkey = jax.random.split(jkey)
        occ_pred = models_.apply('occ_predictor', latent_obj, ds[4], jkey, False, train=True)
        _, jkey = jax.random.split(jkey)
        col_pred = models_.apply('col_predictor', latent_obj, jkey, rngs={'dropout':jkey}, train=True)
        _, jkey = jax.random.split(jkey)
        pln_pred = models_.apply('pln_predictor', latent_obj, ds[7], ds[8], train=True)
        _, jkey = jax.random.split(jkey)
        ray_pred = models_.apply('ray_predictor', latent_obj, ds[10], ds[11], train=True)
        _, jkey = jax.random.split(jkey)

        occ_loss = jnp.mean(BCE_loss(occ_pred, ds[5]))
        col_loss = jnp.mean(BCE_loss(col_pred, ds[6]))
        pln_loss = jnp.mean(BCE_loss(pln_pred, ds[9]))

        depth_gt, normal_gt = ds[12], ds[13]
        seg_gt = (jnp.abs(depth_gt)<=5.0).astype(jnp.float32)
        loss_ray_depth = jnp.sum(seg_gt*jnp.square(ray_pred[1] - depth_gt), axis=-1)
        loss_ray_normal = -jnp.sum(seg_gt*tutil.normalize(normal_gt) * tutil.normalize(ray_pred[2]), -1)
        ray_seg_loss = BCE_loss(ray_pred[0], seg_gt.squeeze(-1))
        ray_loss = jnp.mean(ray_seg_loss+loss_ray_normal+loss_ray_depth)

        # reg loss
        z = latent_obj.z
        z = einops.rearrange(z, '... i j -> ... (i j)')
        z = z.reshape(-1, z.shape[-1])
        z_mu = jnp.mean(z, 0, keepdims=True)
        zdif = (z - z_mu)
        z_cov = jnp.einsum('ij,ir->jr', zdif, zdif) / zdif.shape[0]
        reg_loss = 10*jnp.sum(z_mu**2) + jnp.sum((z_cov - jnp.eye(z_cov.shape[-1]))**2)

        loss = args.occ_loss_coef * occ_loss + col_loss + args.reg_loss_coef * reg_loss + pln_loss + ray_loss
        return loss, {'train_occ_loss': occ_loss, 'train_col_loss': col_loss, 'train_total_loss': loss, 'train_reg_loss':reg_loss,
                      'train_pln_loss':pln_loss, 'train_ray_loss':ray_loss}

    cal_loss_grad = jax.grad(cal_loss, has_aux=True)

    # cal_loss_jit = cal_loss
    cal_loss_jit = jax.jit(cal_loss)

    # %%
    # define train func
    def l2_norm(tree):
        """Compute the l2 norm of a pytree of arrays. Useful for weight decay."""
        leaves, _ = jax.tree_util.tree_flatten(tree)
        return jnp.sqrt(sum(jnp.vdot(x, x) for x in leaves))
    
    def clip_grads(grad_tree, max_norm):
        """Clip gradients stored as a pytree of arrays to maximum norm `max_norm`."""
        norm = l2_norm(grad_tree)
        normalize = lambda g: jnp.where(norm < max_norm, g, g * (max_norm / norm))
        return jax.tree_map(normalize, grad_tree)

    def apply_rot_aug(ds, jkey, rot_aug):
        if rot_aug!='x' and rot_aug!='so3':
            return ds
        obj = cxutil.CvxObjects().init_vtx(*ds[:2]).init_pcd(*ds[2:4])
        qpnts = ds[4]
        if rot_aug=='x':
            xrot = jax.random.uniform(jkey, shape=obj.outer_shape, minval=0, maxval=np.pi)
            qrand = tutil.aa2q(jnp.stack([xrot, jnp.zeros_like(xrot), jnp.zeros_like(xrot)], -1))
        elif rot_aug=='so3':
            qrand = tutil.qrand(obj.outer_shape, jkey)
        obj = obj.rotate_rel_vtxpcd(qrand)
        qpnts = tutil.qaction(qrand[...,None,:], qpnts-obj.pos[...,None,:]) + obj.pos[...,None,:]
        _, jkey = jax.random.split(jkey)
        return obj.vtx_tf, obj.fc, obj.pcd_tf, obj.pcd_dc_idx, qpnts, ds[5]


    def train_func(params, opt_state, ds, jkey, cvxfc):
        if pre_data_gen is not None:
            ds2, jkey = pre_data_gen.data_gen_one_batch(jkey, cvxfc, save=False)
            ds = jax.tree_map(lambda *x: jnp.concatenate(x, 0), ds, ds2)
        ds = apply_rot_aug(ds, jkey, args.rot_train)
        _, jkey = jax.random.split(jkey)
        grad, train_loss_dict = cal_loss_grad(params, ds, jkey)
        grad = jax.tree_map(lambda x: jnp.where(jnp.isnan(x), 0, x), grad)
        if args.grad_clip != 0:
            grad = clip_grads(grad, args.grad_clip)
        _, jkey = jax.random.split(jkey)
        updates, opt_state = optimizer.update(grad, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, jkey, train_loss_dict, grad

    # train_func_jit = train_func
    train_func_jit = jax.jit(train_func)

    @jax.jit
    def occ_inf_test(params, ds, jkey, rot_aug=True):
        """inference function for evaluation"""
        models_ = models
        models_ = models_.set_params(params)
        if rot_aug:
            ds = apply_rot_aug(ds, jkey, args.rot_test)
        obj = cxutil.CvxObjects().init_vtx(*ds[:2]).init_pcd(*ds[2:4])
        qpnts = ds[4]
        latent_obj = models_.apply('shape_encoder', obj, jkey)
        _, jkey = jax.random.split(jkey)
        occ_pred = models_.apply('occ_predictor', latent_obj, qpnts, jkey)
        _, jkey = jax.random.split(jkey)
        col_pred = models_.apply('col_predictor', latent_obj, jkey, rngs={'dropout':jkey})
        _, jkey = jax.random.split(jkey)
        pln_pred = models_.apply('pln_predictor', latent_obj, ds[7], ds[8])
        _, jkey = jax.random.split(jkey)
        ray_pred = models_.apply('ray_predictor', latent_obj, ds[10], ds[11])
        occ_loss = BCE_loss(occ_pred, ds[5])
        col_loss = BCE_loss(col_pred, ds[6])
        pln_loss = BCE_loss(pln_pred, ds[9])
        ray_loss = BCE_loss(ray_pred[0], (jnp.abs(ds[12])<5.0).astype(jnp.float32).squeeze(-1))
        return occ_pred, col_pred, pln_pred, ray_pred[0], occ_loss, col_loss, pln_loss, ray_loss

    
    # %%
    # start training
    now = datetime.datetime.now() # current date and time
    date_time = now.strftime("%m%d%Y-%H%M%S")
    if vessl_on:
        logs_dir = os.path.join('/output', date_time)
    else:
        logs_dir = os.path.join('logs', date_time)
    writer = SummaryWriter(logs_dir)
    # Copy model utils
    shutil.copy(__file__, logs_dir)
    if not os.path.exists(os.path.join(logs_dir, "util")):
        os.mkdir(os.path.join(logs_dir, "util"))
    shutil.copy(os.path.join(BASEDIR, "util", "model_util.py"), os.path.join(logs_dir, "util", "model_util.py"))
    writer.add_text('args', args.__str__(), 0)

    best_occ_acc = 0
    best_col_acc = 0
    # params = models.params
    for itr in range(10000):
        train_loss = 0
        tr_cnt = 0
        for i, ds in enumerate(train_loader):
            ds = jax.tree_map(lambda x: jnp.asarray(x), ds) 
            '''
            # (vtx, fcs, pcd, dc_idx, qpnts, occ_label)
            convex mesh info
                vtx: (B, 2, C, V, 3) (float)
                fcs: (B, 2, C, F, 3) (int)
            pcd: (B, 2, P, C) (float)
            dc_idx: (B, 2, P) (int)
            qpnts: (B, 2, S, 3) (float)
            occ_label: (B, 2, S) (bool)
            '''
            if pre_data_gen is None:
                vtxfc = None
            else:
                if i%500==0:
                    vtxfc = pre_data_gen.init_objs(100)

            params, opt_state, jkey, train_loss_dict_, grad_ = train_func_jit(params, opt_state, ds, jkey, vtxfc)
            _, jkey = jax.random.split(jkey)

            # Temporary fault dataset handling
            for k, v in train_loss_dict_.items():
                if jnp.isnan(v).item():
                    print("Preventing NaN screwing up vessl")
                    train_loss_dict_[k] = jnp.array([0.], dtype=jnp.float32)
            # Log...
            if i == 0:
                train_loss_dict = train_loss_dict_
            else:
                train_loss_dict = jax.tree_map(lambda x,y: x+y, train_loss_dict, train_loss_dict_)
            tr_cnt += 1
            if i%100==0:
                print(i, train_loss_dict_)
                # # Debug here
                # print(f"debug: {i}, {train_loss_dict}")
                # if jnp.any(jnp.isnan(train_loss_dict["train_pln_loss"])).item():
                #     print('hold')
        train_loss_dict = jax.tree_map(lambda x: x/tr_cnt, train_loss_dict)
        # train_loss = train_loss/tr_cnt

        train_loader.dataset.ds_init()

        # evaluations
        total_occ = 0
        total_col = 0
        occ_acc = 0
        col_acc = 0
        pln_acc = 0
        seg_acc = 0
        eval_occ_loss = 0
        eval_col_loss = 0
        for i, ds in enumerate(eval_loader):
            _, jkey = jax.random.split(jkey)
            ds = jax.tree_map(lambda x: jnp.asarray(x), ds)
            occ_pred, col_pred, pln_pred, seg_pred, occ_loss_, col_loss_, pln_loss_, seg_loss_ = occ_inf_test(params, ds, jkey)
            eval_occ_loss += occ_loss_
            eval_col_loss += col_loss_
            occ_acc += jnp.sum(jnp.logical_and(occ_pred>0, ds[5]>0.5)) + jnp.sum(jnp.logical_and(occ_pred<0, ds[5]<0.5))
            col_acc += jnp.sum(jnp.logical_and(col_pred>0, ds[6]>0.5)) + jnp.sum(jnp.logical_and(col_pred<0, ds[6]<0.5))
            pln_acc += jnp.sum(jnp.logical_and(pln_pred>0, ds[9]>0.5)) + jnp.sum(jnp.logical_and(pln_pred<0, ds[9]<0.5))
            seg_gt = (jnp.abs(ds[12])<5.0).astype(jnp.float32).squeeze(-1)
            seg_acc += jnp.sum(jnp.logical_and(seg_pred>0, seg_gt>0.5)) + jnp.sum(jnp.logical_and(seg_pred<0, seg_gt<0.5))
            total_occ += ds[5].shape[0] * ds[5].shape[1] * ds[5].shape[2]
            total_col += ds[6].shape[0]
        cur_occ_acc = occ_acc/total_occ
        cur_col_acc = col_acc/total_col
        cur_pln_acc = pln_acc/total_occ
        cur_seg_acc = seg_acc/total_occ
        eval_occ_loss = jnp.sum(eval_occ_loss)/total_occ
        eval_col_loss = jnp.sum(eval_col_loss)/total_col
        if best_col_acc < cur_col_acc:
            best_col_acc = cur_col_acc
        if best_occ_acc < cur_occ_acc:
            best_occ_acc = cur_occ_acc
            with open(os.path.join(logs_dir, 'saved.pkl'), 'wb') as f:
                pickle.dump({'params':params, 'opt_state':opt_state, 'args':args, 'models':models.set_params(params), 'rot_configs':models.rot_configs}, f)
            if not vessl_on:
                import open3d as o3d
                from examples.visualize_occ import create_mesh
                mesh_path = os.path.join(BASEDIR, 'data/DexGraspNet/32_64_1_v4/sem-Bowl-8eab5598b81afd7bab5b523beb03efcd.obj')
                mesh_0, mesh_basename = create_mesh(args, jkey, None, mesh_path, input_type='cvx', models=models.set_params(params))
                o3d.io.write_triangle_mesh(os.path.join(logs_dir, mesh_basename), mesh_0)

        log_dict = {'occ_acc':cur_occ_acc, 'col_acc':cur_col_acc, 
                    "best_occ_acc":best_occ_acc, "best_col_acc":best_col_acc, "pln_acc":cur_pln_acc, "cur_seg_acc":cur_seg_acc,
                      "eval_occ_loss":eval_occ_loss, "eval_col_loss":eval_col_loss,
                        **train_loss_dict}
        log_dict = {k:np.array(log_dict[k]) for k in log_dict}
        print(f'itr: {itr}, {log_dict}')
        
        for k in log_dict:
            writer.add_scalar(k, log_dict[k], itr)

        if vessl_on:
            base_name = ""
            log_dict = {base_name+k: log_dict[k] for k in log_dict}
            vessl.log(step=itr, payload=log_dict)

