import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"   
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
from pathlib import Path
import jax
import torch
import numpy as np
import jax.numpy as jnp
from torch.utils.data import Dataset
from typing import Dict, List
from functools import partial
import itertools
import random

# Setup import path
import sys
BASEDIR = Path(__file__).parent.parent
if str(BASEDIR) not in sys.path:
    sys.path.insert(0, str(BASEDIR))

import data_generation.scene_generation as sg
import util.cvx_util as cxutil
import util.transform_util as tutil
import util.structs as structs
import util.render_util as rutil


def aggregate_data_from_directory(data_dir_path: Path, category):
    """This function will load all datapoints from separated files and concatenate them"""
    # We will collapse all files. Make sure to use sorted for reproducibility
    if category == 'none':
        dataset_filenames = list(sorted(data_dir_path.iterdir()))
    else:
        dataset_filenames = []
        for cat in category.split('__'):
            dataset_filenames = dataset_filenames + list(sorted(data_dir_path.glob(f'*{cat}.pkl')))
    # Read all
    entire_dataset_list: List[sg.SceneCls.SceneData] = []
    for i, fname in enumerate(dataset_filenames):
        file_path = data_dir_path/fname
        with file_path.open("rb") as f:
            np_data = np.load(f, allow_pickle=True)
            data = np_data["item"].item()
            data["depths"] = None
        entire_dataset_list.append(data)
    # Concatenate along batch dimension
    entire_dataset_batched = jax.tree_map(lambda *x: np.concatenate(x, 0), *entire_dataset_list)
    return entire_dataset_batched


def pytree_collate(batch: List[Dict]):
  """Simple collation for numpy pytree instances"""
  data = jax.tree_map(lambda *x: np.stack(x, 0), *batch)
  return data


import flax
import pickle
def pkl_to_npz(pkl_file_path:Path):
    npz_file_path = pkl_file_path.parent / (pkl_file_path.name[:-4] + '.npz')
    if not npz_file_path.exists():
        with open(pkl_file_path, 'rb') as f:
            datapnt = pickle.load(f)
            numpy_datapoints = flax.serialization.to_state_dict(datapnt)
        with open(npz_file_path, 'wb') as f:
            np.savez_compressed(f, item=numpy_datapoints)
        print(f'save {npz_file_path}')
    else:
        print(f'pass {npz_file_path}')

def del_if_exist(data, key):
    if key in data:
        del data[key]

def preprocess_datapoint(data):
    del_if_exist(data, "depths")
    del_if_exist(data, "table_params")
    del_if_exist(data['obj_info'], "scale")
    del_if_exist(data['obj_info'], "mesh_name")
    del_if_exist(data['obj_info'], "uid_list")
    data['seg'] = data['seg']>=0
    return data

class EstimatorDataset(Dataset):
    """Estimator dataset tailored for FLAX"""
    def __init__(
            self,
            dataset:str,
            data_dir_path:Path,
            category:str,
            size_limit:int,
            ds_obj_no:int,
            ds_type='train',
    ):
        """Entire data is already loaded in the memory"""
        
        self.size_limit = size_limit
        self.ds_type = ds_type
        # if category == 'none':
        #     dataset_filenames = list(sorted(data_dir_path.glob('*.npz')))
        # else:
        dataset_filenames = []
        if dataset=='NOCS':
            dataset_filenames = list(sorted(data_dir_path.glob(f'*{dataset}*.npz')))
        else:
            for cat in category.split('__'):
                dataset_filenames = dataset_filenames + list(sorted(data_dir_path.glob(f'*{cat}.npz')))
            dataset_filenames = list(sorted(dataset_filenames))

        # filter max num
        dataset_filenames_ = []
        for df in dataset_filenames:
            base_name = str(df.name)
            ds_max_obj_no = int(base_name.split('_')[-3])
            if ds_max_obj_no == ds_obj_no:
                dataset_filenames_.append(df)
        dataset_filenames = dataset_filenames_
        # dataset_filenames = [df for df in dataset_filenames if 'seg' in str(df.name).split('_')]
        
        # if train_seg:
        #     # segmentation filtering
        #     dataset_filenames = [df for df in dataset_filenames if 'seg' in str(df.name).split('_')]
        # else:
        #     dataset_filenames = [df for df in dataset_filenames if 'seg' not in str(df.name).split('_')]

        if self.ds_type == 'test':
            dataset_filenames = [df for df in dataset_filenames if str(df.name).split('_')[0]=='val']
        else:
            dataset_filenames = [df for df in dataset_filenames if str(df.name).split('_')[0]!='val' and str(df.name).split('_')[0]!='test']
        shelf_ds_no = len([df for df in dataset_filenames if 'shelf' in str(df.name).split('_')])
        table_ds_no = len(dataset_filenames) - shelf_ds_no
        print(f"ds type: {self.ds_type} // dataset name {dataset} // fn loaded: {len(dataset_filenames)} // shelf no: {shelf_ds_no} // table no: {table_ds_no}")

        random.shuffle(dataset_filenames)
        if self.ds_type == 'test':
            self.dataset_filenames = dataset_filenames
        else:
            self.dataset_filenames = itertools.cycle(dataset_filenames)

        # Read all
        for i, fname in enumerate(self.dataset_filenames):
            print(f'load {str(fname)}')
            with fname.open("rb") as f:
                np_data = np.load(f, allow_pickle=True)
                data = np_data["item"].item()
                data = preprocess_datapoint(data)
            if i == 0:
                entire_dataset_batched = data
            else:
                entire_dataset_batched = jax.tree_map(lambda *x: np.concatenate(x, 0), entire_dataset_batched, data)
            
            if self.ds_type == 'train' and entire_dataset_batched["rgbs"].shape[0] > size_limit:
                break
        if self.ds_type == 'train':
            entire_dataset_batched = jax.tree_map(lambda x: x[-self.size_limit:], entire_dataset_batched)
        self.entire_data = entire_dataset_batched
        print(f"ds type {self.ds_type} finish")

        if self.ds_type == 'test':
            obj_info = self.entire_data["obj_info"]
            
            occ_ns = 2048
            # occ_ns = 256
            jkey = jax.random.PRNGKey(0)
            @jax.jit
            def extract_occ_ds(jkey, obj_info_):
                gt_obj_cvx = cxutil.CvxObjects().init_obj_info(obj_info_)
                # spnts, dc_idx = cxutil.sampling_from_surface_convex_dec(jkey, gt_obj_cvx.vtx_tf, gt_obj_cvx.fc, 256)
                # qpnts = spnts + jax.random.normal(jkey, spnts.shape) * 0.008
                # qpnts = cxutil.sampling_from_AABB(jkey, gt_obj_cvx.vtx_tf, 256, margin=0.020)
                qpnts = cxutil.sampling_from_AABB(jkey, obj_info_['obj_cvx_verts_padded'], occ_ns, margin=0.07)
                qpnts = tutil.pq_action(obj_info_['obj_posquats'][...,None,:3], obj_info_['obj_posquats'][...,None,3:], qpnts)
                _, jkey = jax.random.split(jkey)
                occ_label = cxutil.occ_query(gt_obj_cvx.vtx_tf, gt_obj_cvx.fc, qpnts).astype(jnp.float32)
                return qpnts, occ_label

            one_itr_ns = 16
            nds = self.entire_data['rgbs'].shape[0]
            nobj = self.entire_data['obj_info']['obj_cvx_verts_padded'].shape[1]
            itr_no = nds//one_itr_ns+1
            qpnts = np.zeros((nds, nobj, occ_ns, 3), dtype=np.float16)
            occ_label = np.zeros((nds,nobj,occ_ns), dtype=bool)
            for i in range(itr_no):
                if one_itr_ns*i >= nds:
                    break
                obj_info_ = jax.tree_map(lambda x:x[one_itr_ns*i:one_itr_ns*(i+1)], obj_info)
                qpnts_, occ_label_ = extract_occ_ds(jkey, obj_info_)
                qpnts[one_itr_ns*i:one_itr_ns*(i+1)]=np.array(qpnts_)
                occ_label[one_itr_ns*i:one_itr_ns*(i+1)]=np.array(occ_label_)
            
            self.entire_data["qpnts"] = qpnts
            self.entire_data["occ_label"] = occ_label

    def __len__(self):
        """Dataset size"""
        return self.entire_data["rgbs"].shape[0]

    def __getitem__(self, index) -> sg.SceneCls.SceneData:
        """All operations will be based on tree_map"""
        # Index an item (squeeze batch dim)
        data = jax.tree_map(lambda x: x[index], self.entire_data)
        # - If you need some online pre-processing, add them here.
        #   We will not use torch tensor here.
        # pass
        
        return data
    
    def push(self):
        for i, fname in enumerate(self.dataset_filenames):
            print(f"fn {str(fname)} start pushing")
            with fname.open("rb") as f:
                np_data = np.load(f, allow_pickle=True)
                data = np_data["item"].item()
                data = preprocess_datapoint(data)
                # data["depths"] = None
                # data['table_params'] = None
            if 'seg' not in data:
                data['seg'] = None
            self.entire_data = jax.tree_map(lambda *x: np.concatenate(x, 0), self.entire_data, data)
            if self.entire_data['rgbs'].shape[0] > self.size_limit:
                print(f"fn {str(fname)} pushed ds len: {data['rgbs'].shape[0]}")
                break
        self.entire_data = jax.tree_map(lambda x: x[-self.size_limit:], self.entire_data)
