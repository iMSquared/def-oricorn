import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import jax
import jax.numpy as jnp
import numpy as np
import pickle
import glob
import os, sys
import einops
import datetime
from functools import partial
from typing import List, Tuple
import argparse
from tqdm import tqdm
import csv

np.seterr(over='raise')

base_dir = os.path.dirname(os.path.dirname(__file__))
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)

import util.cvx_util as cxutil
import util.io_util as ioutil


@partial(jax.jit, static_argnames=['increase_ratio'])
def occ_ds_given_spnts(jkey, vtx, fcs, spnts, increase_ratio=2):
    # spnts = cxutil.sampling_from_surface(jkey, vtx, fcs, ns=spnts.shape[-2])
    np = spnts.shape[-2]
    aabb = jnp.max(spnts, axis=-2, keepdims=True) - jnp.min(spnts, axis=-2, keepdims=True)
    obj_len = jnp.mean(aabb, axis=-1, keepdims=True)
    spnts1 = spnts[...,:np//20,:]
    spnts2 = spnts[...,np//20:,:]
    spnts2, spnts3 = spnts2[...,:spnts2.shape[-2]//2,:], spnts2[...,spnts2.shape[-2]//2:,:]
    # qpnts1 = spnts1[None] + jax.random.normal(jkey, shape=(2,)+spnts1.shape)*0.1
    qpnts1 = spnts1[None] + jax.random.uniform(jkey, shape=(increase_ratio,)+spnts1.shape, minval=-0.8, maxval=0.8)
    _, jkey = jax.random.split(jkey)
    qpnts2 = spnts2[None] + jax.random.normal(jkey, shape=(increase_ratio,)+spnts2.shape)*obj_len*0.05
    _, jkey = jax.random.split(jkey)
    qpnts3 = spnts3[None] + jax.random.normal(jkey, shape=(increase_ratio,)+spnts3.shape)*obj_len*0.005
    # qpnts3 = spnts3[None] + jax.random.normal(jkey, shape=(increase_ratio,)+spnts3.shape)*obj_len*0.001
    _, jkey = jax.random.split(jkey)
    qpnts = jnp.concatenate([qpnts1, qpnts2, qpnts3], axis=-2)
    qpnts = einops.rearrange(qpnts, 'i ... k j -> ... (i k) j')
    occ_res = cxutil.occ_query(vtx, fcs, qpnts)
    return qpnts, occ_res

class PretrainingDataGen(object):
    def __init__(self, nspnts=1000, nb=50, nq=600, occ_increase_ratio=2, dataset='DexGraspNet', category="all", augment_cabinet=False):
        '''
        nspnts: surface sample points number
        nb: number of collision pairs in a batch
        nq: number of queries for plane/ray dataset
        occ_increase_ratio: number of occupancy points=occ_increase_ratio*nspnts
        category: "cherrypick" or "all"
        '''
        self.nspnts = nspnts
        self.nb = nb
        self.nq = nq
        self.dataset = dataset
        self.occ_increase_ratio = occ_increase_ratio
        self.save_base_dir = os.path.join(base_dir, f'data/{dataset}Occ_debug/32_64_v4_col_{category}')
        self.augment_cabinet = augment_cabinet
        obj_base_dir = os.path.join(base_dir, f'data/{dataset}/32_64_1_v4_textured')
        if category == "cherrypick":
            print("Datagen uses CHERRYPICKED category")
            objfilename_list = ioutil.get_filepath_from_annotation(
                "data/annotations.csv", obj_base_dir, ["32_64_1_v4", "32_64_5_v4"])
        elif category == "cherrypick2":
            print("Datagen uses CHERRYPICKED ver.2 category")
            objfilename_list = ioutil.get_filepath_from_annotation(
                "data/annotations2.csv", obj_base_dir, ["32_64_1_v4", "32_64_5_v4"])
        else:
            print("Datagen uses ALL category (not cherrypicked)")
            objfilename_list = glob.glob(os.path.join(obj_base_dir, '*.obj'))
        
        # filtering
        if dataset=='NOCS':
            objfilename_list = [ol for ol in objfilename_list if os.path.basename(ol).split('v4')[1].split('-')[0] != 'laptop']
             # add categorical scale
            base_len = 0.65
            self.categorical_scale = {'can':1/base_len*0.15, 'bottle':1/base_len*0.15, 'bowl':1/base_len*0.2, 
                                    'camera':1/base_len*0.15, 'laptop':1/base_len*0.3, 'mug':1/base_len*0.12}

            # with open('class_id.csv', 'w', newline='') as csvfile:
            #     spamwriter = csv.writer(csvfile)
            #     for md in objfilename_list:
            #         meshid = md.split('/')[-1]
            #         clsid = os.path.basename(md).split('v4')[1].split('-')[0]
            #         spamwriter.writerow([clsid, meshid, self.categorical_scale[clsid]])

        objfilename_list += glob.glob('data/PANDA/*/*.obj')
        objfilename_list += glob.glob('data/PANDA/*.obj')
        objfilename_list += glob.glob('data/shelf/32_64_1_v4/*.obj')
        if len(glob.glob('data/shelf/32_64_1_v4/*.obj')) == 0:
            raise ValueError("cabinet not loaded")
        self.objfilename_list = objfilename_list

        self.sample_jit = jax.jit(lambda x,y,z: cxutil.sampling_from_surface_convex_dec(x,y,z, ns=self.nspnts))
        self.col_gen_jit = jax.jit(lambda jk, vt, fc: cxutil.gen_col_dataset(jk, self.nb, vt, fc))
        # self.col_gen_jit = lambda jk, vt, fc: cxutil.gen_col_dataset(jk, self.nb, vt, fc, visualize=True)

        self.pln_jit = jax.jit(lambda jk, obj: cxutil.gen_pln_dataset(jk, self.nq, obj))
        self.ray_jit = jax.jit(lambda jk, obj: cxutil.gen_ray_dataset(jk, self.nq, obj))

        # Data fault logger... we will fix this someday
        self.total_fault = 0


    def init_objs(self, load_obj_no):
        if load_obj_no==0 or load_obj_no >= len(self.objfilename_list):
            obj_fn_pool = self.objfilename_list
        else:
            obj_fn_pool = np.random.choice(self.objfilename_list, size=(load_obj_no,))

        # add finger and hand
        obj_fn_pool = list(obj_fn_pool)
        obj_fn_pool += glob.glob('data/PANDA/*/finger.obj')
        obj_fn_pool += glob.glob('data/PANDA/*/hand.obj')
        if self.augment_cabinet:
            print("augmenting cabinet")
            obj_fn_pool += glob.glob('data/shelf/32_64_1_v4/*.obj')
            if len(glob.glob('data/shelf/32_64_1_v4/*.obj')) == 0:
                raise ValueError("cabinet not loaded")

        mesh_name_list = []
        cvx_obj_list = []
        cvx_fcs_list = []
        fc_no_list = []
        vtx_no_list = []
        for ofn in obj_fn_pool:
            mesh_name_list.append(os.path.basename(ofn))
            dataset_name = os.path.basename(os.path.dirname(os.path.dirname(ofn)))
            if dataset_name == 'NOCS':
                cat = os.path.basename(ofn)[10:-4].split('-')[0]
                scale = self.categorical_scale[cat]
            else:
                scale = 1.0
            vtx_, fc_, vtx_no, face_no = cxutil.vex_obj_parsing(ofn, max_dec_size=32, max_vertices=64, scale=scale)
            cvx_obj_list.append(vtx_)
            cvx_fcs_list.append(fc_)
            fc_no_list.append(face_no)
            vtx_no_list.append(vtx_no)

        # add more shelf
        shelf_params = np.random.uniform([0.15,0.15,0.15,0.02], [0.4,0.4,0.4,0.04], size=(2,4))
        # for parm in [[0.4,0.4,0.15,0.03], [0.15,0.3,0.2, 0.03], [0.2,0.2,0.15,0.02]]:
        for parm in shelf_params:
            shelf_obj = cxutil.create_shelf_dc(32, 64, height=parm[0], width=parm[1], depth=parm[2], thinkness=parm[3], include_bottom=True)
            shelf_obj = shelf_obj.translate(shelf_obj.pos)
            cvx_obj_list.append(shelf_obj.vtx_tf)
            cvx_fcs_list.append(shelf_obj.fc)

        print(f'loaded object set no: {len(cvx_obj_list)}')

        # self.cvx_obj_list = jnp.array(cvx_obj_list).astype(jnp.float32)
        # self.cvx_fcs_list = jnp.array(cvx_fcs_list).astype(jnp.int32)
        return jnp.array(cvx_obj_list).astype(jnp.float32), jnp.array(cvx_fcs_list).astype(jnp.int32)
        
    
    def data_gen_one_batch(self, jkey, cvxfc_list, save=True):
        cvx_obj_list, cvx_fcs_list = cvxfc_list
        cvx_obj, col_res, jkey = self.col_gen_jit(jkey, cvx_obj_list, cvx_fcs_list)
        cvx_obj:cxutil.CvxObjects
        vtc, fc = map(lambda x : einops.rearrange(x, 'i j ... -> (i j) ...'), (cvx_obj.vtx_tf, cvx_obj.fc))
        _, jkey = jax.random.split(jkey)
        spnts, dec_idx = self.sample_jit(jkey, vtc, fc)
        _, jkey = jax.random.split(jkey)
        qpnts, occ_res = occ_ds_given_spnts(jkey, vtc, fc, spnts, increase_ratio=self.occ_increase_ratio)
        _, jkey = jax.random.split(jkey)

        # assert jnp.all(jnp.sum(occ_res, axis=-1) > 100)
        if save and not jnp.all(jnp.sum(occ_res, axis=-1) > 100):
            print(f'too small inside qpnts {jnp.min(jnp.sum(occ_res, axis=-1))}')
            return None, jkey
        
            import open3d as o3d
            print(jnp.sum(occ_res, axis=-1))
            idx = jnp.argmin(jnp.sum(occ_res, axis=-1))
            vtx__ = vtc[idx]
            vtx__ = vtx__[jnp.where(jnp.all(jnp.abs(vtx__)<1000))]
            # visualize points
            vtxpcd = o3d.geometry.PointCloud()
            spcd = o3d.geometry.PointCloud()
            qpcd = o3d.geometry.PointCloud()
            ipcd = o3d.geometry.PointCloud()
            # for idx in range(qpnts.shape[0]):
            vtxpcd.points = o3d.utility.Vector3dVector(vtx__)
            vtxpcd.paint_uniform_color([0.2, 1.0, 0.1])
            spcd.points = o3d.utility.Vector3dVector(spnts[idx])
            spcd.paint_uniform_color([1, 0.706, 0])
            qpcd.points = o3d.utility.Vector3dVector(qpnts[idx][np.logical_not(occ_res[idx])])
            ipcd.points = o3d.utility.Vector3dVector(qpnts[idx][occ_res[idx]])
            ipcd.paint_uniform_color([0.2, 0.206, 1])
            o3d.visualization.draw_geometries([vtxpcd, spcd, qpcd, ipcd])
            # visualize points

        pln_p, pln_n, pln_label, jkey = self.pln_jit(jkey, cvx_obj)
        _, jkey = jax.random.split(jkey)
        ray_p, ray_dir, ray_depth, ray_normals, jkey = self.ray_jit(jkey, cvx_obj)
        _, jkey = jax.random.split(jkey)

        # transfer to numpy

        qpnts, occ_res, spnts, dec_idx = jax.tree_map(lambda x: einops.rearrange(x, '(i j) ... -> i j ...', j=2), (qpnts, occ_res, spnts, dec_idx))
        if save:
            occ_ds = (np.array(qpnts).astype(np.float16), np.array(occ_res).astype(bool))
            obj_ds = (np.array(cvx_obj.vtx_tf).astype(np.float32), np.array(cvx_obj.fc).astype(np.int16), np.array(spnts).astype(np.float16), np.array(dec_idx).astype(np.uint8))
            pln_ds = (np.array(pln_p).astype(np.float16), np.array(pln_n).astype(np.float16), np.array(pln_label).astype(bool))
            ray_ds = (np.array(ray_p).astype(np.float16), np.array(ray_dir).astype(np.float16), 
                    np.array(ray_depth).astype(np.float16), np.array(ray_normals).astype(np.float16))
            save_tuple = (obj_ds, np.array(col_res<=0).astype(bool), occ_ds, pln_ds, ray_ds)
            assert np.logical_not(np.any(jax.tree_util.tree_flatten(jax.tree_map(lambda x: np.any(np.isnan(x)), save_tuple))[0]))
        else:
            
            pln_ds = (pln_p.astype(jnp.float32), pln_n.astype(jnp.float32), pln_label.astype(jnp.float32))
            ray_ds = (ray_p.astype(jnp.float32), ray_dir.astype(jnp.float32), 
                    ray_depth.astype(jnp.float32), ray_normals.astype(jnp.float32))
            save_tuple = [cvx_obj.vtx_tf.astype(jnp.float32), cvx_obj.fc.astype(jnp.int32), spnts.astype(jnp.float32), dec_idx.astype(jnp.int32), 
                    qpnts.astype(jnp.float32), occ_res.astype(jnp.float32), (col_res<=0).astype(jnp.float32), *pln_ds, *ray_ds]

        return save_tuple, jkey


    def data_save(self, itr, jkey, cvxfc: Tuple[jnp.ndarray, jnp.ndarray]):
        
        date_time = datetime.datetime.now().strftime("%m%d%Y_%H%M%S")
        for i in tqdm(range(itr)):
            save_tuple, jkey = self.data_gen_one_batch(jkey, cvxfc, save=True)
            if save_tuple is None:
                continue        
            # Hmm...... cannot handle this right now, but need to fix someday. 
            if jnp.any(jnp.array(jax.tree_map(lambda x: jnp.any(jnp.isinf(x)), jax.tree_util.tree_flatten(save_tuple)[0]))):
                self.total_fault += jnp.sum(jnp.array(jax.tree_map(lambda x: jnp.any(jnp.isinf(x)), jax.tree_util.tree_flatten(save_tuple)[0]))).item()
                print(f"data fault: {self.total_fault}")

            output_file_name = os.path.join(self.save_base_dir, date_time + '_' + str(i) + '.pkl')
            if not os.path.exists(self.save_base_dir):
                os.makedirs(self.save_base_dir)
            with open(output_file_name, 'wb') as f:
                pickle.dump(save_tuple, f)

            # print(i)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="NOCS")
    parser.add_argument("--category", type=str, default="all")
    parser.add_argument("--augment_cabinet", type=int, default="0")
    args = parser.parse_args()

    pretraining_data_gen = PretrainingDataGen(dataset=args.dataset, category=args.category, augment_cabinet=args.augment_cabinet)
    seed = np.random.randint(0, 100000)
    # seed = 0
    print(f'current seed {seed}')
    jkey = jax.random.PRNGKey(seed)
    cvxfc = pretraining_data_gen.init_objs(0) # Set 0 when generating the actual training set.
    # res = pretraining_data_gen.data_gen_one_batch(jkey, save=False)
    # print(1)

    # dg_func = jax.jit(partial(pretraining_data_gen.data_gen_one_batch, save=False))
    # dg_func = partial(pretraining_data_gen.data_gen_one_batch, save=False)
    # res = dg_func(jkey, cvxfc)

    # print(1)

    # cvxfc = pretraining_data_gen.init_objs(100)
    # res = dg_func(jkey, cvxfc)

    # print(1)

    pretraining_data_gen.data_save(5000, jkey, cvxfc)