'''
This script evaluates the performance on the scenedata.
'''
from __future__ import annotations
import os
# os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"   
# os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"    # NVISII will crash when showed multiple devices with jax.

# This will prevent clip from being crashed
import torch
import jax
import jax.numpy as jnp
import flax
import numpy as np
import random
import pybullet as pb
from pybullet_utils.bullet_client import BulletClient
from functools import partial
from tqdm import tqdm
from bidict import bidict
import argparse
import datetime

from pathlib import Path
from typing import Tuple, Dict, Callable, List



# Setup import path
import sys
BASEDIR = Path(__file__).parent.parent.parent
if str(BASEDIR) not in sys.path:
    sys.path.insert(0, str(BASEDIR))


import util.io_util as ioutil
import util.camera_util as cutil
import util.cvx_util as cxutil
import util.transform_util as tutil
import util.pb_util as pbutil
import util.environment_util as envutil
import util.franka_util as fkutil
import util.render_util as rutil
import util.inference_util as ifutil
import util.eval_util as evalutil
import util.pcd_util as pcdutil
import pickle

# Environment
from imm_pb_util.imm.pybullet_util.vision import (
    CoordinateVisualizer, visualize_point_cloud, draw_bounding_box, transform_points,
    unwrap_o3d_pcd, wrap_o3d_pcd, PybulletPCDManager
)




def load_raw_scenedata(
        data_dir_path: Path, 
) -> Tuple[Dict[str, np.ndarray|Dict], int]:
    """Load raw scene data to memory
    
    Args:
        data_dir_path (Path): Location of scene data
        split (str): val or train

    Return:
        data_batched (Dict[str, np.ndarray|Dict]): Batched data
        num_data (int): Data size
    """
    # Select split
    eval_data_file_path_list = sorted([df for df in data_dir_path.glob("*.npz") if str(df.name).split('_')[0]=='test'])
    # eval_data_file_path_list = sorted([df for df in data_dir_path.glob("*.npz") if str(df.name).split('_')[0]=='test' and 'table' in df.name])


    # Load to memory
    for i, fname in enumerate(eval_data_file_path_list):
        print(f'Loading {str(fname)}')
        with fname.open("rb") as f:
            np_data = np.load(f, allow_pickle=True)
            data = np_data["item"].item()
        # Aggregate
        if i == 0:
            data_batched = data
        else:
            data_batched = jax.tree_util.tree_map(lambda *x: np.concatenate(x, 0), data_batched, data)
    
    # Final size
    num_data = data_batched['rgbs'].shape[0]
    print(f"current_size: {num_data}")

    return data_batched, num_data


@flax.struct.dataclass
class ParsedItem:
    rgb: np.ndarray
    cam_intrinsics: np.ndarray
    cam_posquats: np.ndarray
    mesh_names: np.ndarray
    obj_posquats: np.ndarray
    obj_scales: np.ndarray
    obj_symmetricities: np.ndarray
    plane_height: float


def parse_datapoint_items(
        data_batched: Dict[str, np.ndarray|Dict], 
        data_index: int,
        num_cameras: int
) -> ParsedItem:
    # Get datapoint from batch
    datapoint = jax.tree_util.tree_map(lambda x: x[data_index], data_batched)
    # Compose class mapping
    symmetric_classes = set(['bottle', 'bowl', 'can'])
    class_dict_ours = bidict({'mug':0, 'bottle': 1, 'camera': 2, 'can': 3, 'bowl': 4, 'laptop': 5})
    # Input
    rgb = datapoint["rgbs"][:num_cameras].astype(np.uint8)
    cam_intrinsics = datapoint["cam_info"]['cam_intrinsics'][:num_cameras].astype(np.float32)
    cam_posquats = datapoint["cam_info"]['cam_posquats'][:num_cameras].astype(np.float32)
    # Output
    uid_list_raw = datapoint["obj_info"]["uid_list"]
    uid_list_valid = uid_list_raw[uid_list_raw>0]
    uid_to_class = datapoint["uid_class"][uid_list_valid]
    # Get objects information
    valid_obj_idx = datapoint["obj_info"]["uid_list"] > 0
    mesh_names = datapoint["obj_info"]["mesh_name"][valid_obj_idx]
    obj_posquats = datapoint["obj_info"]["obj_posquats"][valid_obj_idx].astype(np.float32)
    obj_scales = datapoint["obj_info"]["scale"][valid_obj_idx].astype(np.float32)
    # Plane height
    cvx_objects = cxutil.CvxObjects().init_obj_info(datapoint["obj_info"])
    obj_min_vtx = jnp.min(jnp.where(cvx_objects.vtx_valid_mask, cvx_objects.vtx_tf[...,-1:], jnp.inf), axis=(-1,-2,-3))
    plane_height = float(jnp.mean(jnp.sort(obj_min_vtx[cvx_objects.obj_valid_mask])[:2]))

    obj_class = [class_dict_ours.inverse[i] for i in uid_to_class]
    obj_symmetricities = np.asarray([c in symmetric_classes for c in obj_class])
    # Return item
    item = ParsedItem(
        rgb, cam_intrinsics, cam_posquats, 
        mesh_names, obj_posquats, obj_scales, obj_symmetricities, plane_height
    )
    return item


def eval_func(
        jkey: jax.Array,
        iteration: int,
        inf_cls: ifutil.InfCls,
        item: ParsedItem,
        iou_detection_stats: Dict[float, evalutil.DetectionStats],
        pcd_sample_func: Callable,
        chamfer_func: Callable,
        obj_dir_path: Path,
        num_sample_points: int,
        debug_bc: BulletClient|None = None,
        debug_pcd_manager: PybulletPCDManager|None = None,
):
    """ 

    Raises:
        Exception: Box convexhull failure
    """
    # Batchify inputs
    rgb_input = np.expand_dims(item.rgb, axis=0)
    cam_intrinsics_input = np.expand_dims(item.cam_intrinsics, axis=0)
    cam_posquats_input = np.expand_dims(item.cam_posquats, axis=0)
    # Forward
    pixel_size = rgb_input.shape[-3:-1]
    jkey, subkey = jax.random.split(jkey)
    pred_obj_latent_all, pred_conf_all, _, _, _ = inf_cls.estimation(
        subkey, pixel_size, rgb_input, 
        cam_intrinsics_input, cam_posquats_input, 
        # plane_params = jnp.array([[0,0,1,item.plane_height]], dtype=jnp.float32),
        out_valid_obj_only = True,
        apply_conf_filter = False,  # Make sure to not apply confidence inside for evaluation.
        verbose = 1 )
    # TODO(ssh): deprecate this...
    pred_conf_all = np.asarray(flax.linen.sigmoid(pred_conf_all))

    # Get gt pcd and bbox
    mesh_path_list = [obj_dir_path/mesh_name for mesh_name in item.mesh_names]
    gt_instance_canonical_list = evalutil.get_canonical_gt_instance_data(
        obj_path_list = mesh_path_list,
        obj_pos_list = item.obj_posquats[:, :3],
        obj_quat_list = item.obj_posquats[:, 3:],
        obj_scale_list = item.obj_scales,
        obj_symmetricity_list = item.obj_symmetricities,
        num_sample_points = num_sample_points ) 

    # Get pred pcd and bbox
    jkey, subkey = jax.random.split(jkey)
    pred_instance_list = evalutil.get_transformed_latent_pred(
            subkey, pred_obj_latent_all, pred_conf_all, pcd_sample_func)
    
    # IoU calculation
    iou_3d_match, gt_rot_angle_table = evalutil.get_pairwise_obb_iou(
        gt_instance_canonical_list, pred_instance_list)
    

    # IoU Thresholding
    for iou_threshold, stat in iou_detection_stats.items():
        # IoU matching
        matched_pair_list, unmatched_pred_list, unmatched_gt_list \
            = evalutil.get_match_by_3d_iou(
                gt_instance_canonical_list, pred_instance_list,
                iou_3d_match, iou_threshold, chamfer_func, gt_rot_angle_table, debug = False)
        # Count
        stat.update(iteration, matched_pair_list, unmatched_pred_list, unmatched_gt_list)
        if iteration % 10 == 0:
            stat.print_status(print_ap=True, print_prefix=f"[Log] Current @IoU>{iou_threshold:.2f}")
        # Debug data
        if debug_bc is not None and iou_threshold == 0.25:
            debug_matched_pair_list = matched_pair_list
            debug_unmatched_pred_list = unmatched_pred_list
            debug_unmatched_gt_list = unmatched_gt_list


    # Debug: Visualize ground truth bbox
    if debug_bc:
        # # This visualization box does not consider symmetry.
        # for gt_instance in gt_instance_canonical_list:
        #     corners = np.asarray(gt_instance.obb_o3d.get_box_points())
        #     corners = transform_points(corners, gt_instance.pos, gt_instance.quat)
        #     _ = draw_bounding_box(debug_bc, corners, [0,0,0], [0,0,0,1], line_width=5)
        # for pred_instance in pred_instance_list:
        #     corners = np.asarray(pred_instance.obb_o3d.get_box_points())
        #     _ = draw_bounding_box(debug_bc, corners, [0,0,0], [0,0,0,1], color=(0.8, 0., 0.), line_width=5)
        # # Rotated gt is drawn here.
        # for pair in debug_matched_pair_list:
        #     gt_instance = pair.gt
        #     corners = np.asarray(gt_instance.obb_o3d.get_box_points())
        #     corners = transform_points(corners, gt_instance.pos, gt_instance.quat)
        #     _ = draw_bounding_box(debug_bc, corners, [0,0,0], [0,0,0,1], color=(0, 0.7, 0.), line_width=5)


        for pair in debug_matched_pair_list:
            gt_pcd = transform_points(pair.gt.pcd_np, pair.gt.pos, pair.gt.quat)
            debug_pcd_manager.add(gt_pcd, color=np.array([0,0,1]))
            pred_pcd = pair.pred.pcd_np
            debug_pcd_manager.add(pred_pcd, color=np.array([0,1,0])*pair.pred.confidence)
        for pred in debug_unmatched_pred_list:
            debug_pcd_manager.add(pred.pcd_np, color=np.array([1,0,0])*pred.confidence)
        for gt in debug_unmatched_gt_list:
            gt_pcd = transform_points(gt.pcd_np, gt.pos, gt.quat)
            debug_pcd_manager.add(gt_pcd, color=[0,0,1])
            

        # evalutil.draw_confidence_distribution(pred_conf_all)
        print("")
        debug_bc.removeAllUserDebugItems()
        debug_pcd_manager.remove_all()


def dump_result(
        eval_log_dir: Path,
        ckpt_dir: str,
        iou_detection_stats: Dict[float, evalutil.DetectionStats],
        inf_cls: ifutil.InfCls,
        postfix: str,
):
    # Dump log file
    exp_name = ckpt_dir.split("/")[-1]
    log_fname = eval_log_dir/f"{exp_name}_{postfix}.txt"
    with log_fname.open("a") as f:
        print("Evaluation result", file=f)
        for iou_threshold, stat in iou_detection_stats.items():
            str_out = stat.print_status(print_ap=True, print_prefix=f"[Log] Current @IoU>{iou_threshold:.2f}")
            print(str_out, file=f)

        print("", file=f)
        print("Inf time", file=f)
        est_time = np.array(inf_cls.save_time_intervals['est'])
        opt_time = np.array(inf_cls.save_time_intervals['opt'])
        total_time = est_time + opt_time
        print(f'est: {np.mean(est_time[1:]):.5f} ({np.std(est_time[1:]):.5f}) opt: {np.mean(opt_time[1:]):.5f} ({np.std(opt_time[1:]):.5f})', file=f)
        print(f'total: {np.mean(total_time[1:]):.5f} ({np.std(total_time[1:]):.5f})', file=f)
    
    # Dump prediction details
    for iou_threshold, stat in iou_detection_stats.items():
        dump_fname = eval_log_dir/f"{exp_name}_{postfix}_iou{iou_threshold}_dumped.pkl"
        with dump_fname.open("wb") as f:
            pickle.dump(stat.pickable(), file=f)


def main(args):
    """Main function"""
    # Random control
    SEED = 0
    jkey = jax.random.PRNGKey(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    torch.manual_seed(SEED)
    
    # Debugger
    # bc = BulletClient(pb.GUI)
    # pcd_manager = PybulletPCDManager(bc)
    bc = None
    pcd_manager = None

    # Config
    obj_dir_path = Path('data/NOCS_val/32_64_1_v4')
    data_dir_path = Path('data/scene_data_testset')
    data_batched, num_data = load_raw_scenedata(data_dir_path)
    
    # Model
    ckpt_dir = args.ckpt_dir
    cam_z_offset = 0.0
    num_cameras = 3
    num_samples = 32    # Will be force set 1 when nondm.
    ablation_max_time_steps = args.max_time_steps
    inf_cls = ifutil.InfCls(
        ckpt_dir = ckpt_dir,
        ns = num_samples,
        conf_threshold = -100000,
        cam_offset = np.array([0,0,cam_z_offset]),
        save_images = False,
        max_time_steps = ablation_max_time_steps,
        apply_in_range = True,
        early_reduce_sample_size = 2,
        optim_lr=4e-3,
        optimization_step=100,
        gradient_guidance=False,
    )
    inf_cls.compile_jit(num_cameras)
    inf_cls.init_renderer(vis_pixel_size=(64, 112))

    # Useful functions
    num_sample_points = 1000
    pcd_sample_func = jax.jit(partial(pcdutil.get_pcd_from_latent_w_voxel, 
        num_points=num_sample_points, models=inf_cls.models, visualize=False))
    chamfer_func = jax.jit(pcdutil.chamfer_distance_batch)

    # Output
    eval_logs_dir = Path(args.eval_logs_dir)
    eval_logs_dir.mkdir(parents=True, exist_ok=True)

    # Eval setup
    iou_threshold_list = [0.25, 0.5]
    iou_detection_stats = {
        iou: evalutil.DetectionStats(iou) 
        for iou in iou_threshold_list
    }

    eval_func_compact = partial(eval_func, 
        inf_cls = inf_cls,
        pcd_sample_func = pcd_sample_func,
        chamfer_func = chamfer_func,
        obj_dir_path = obj_dir_path,
        num_sample_points = num_sample_points,
        debug_bc = bc,
        debug_pcd_manager = pcd_manager
    )

    # Eval loop
    for i in tqdm(range(num_data)):
        jkey, subkey = jax.random.split(jkey)
        item = parse_datapoint_items(data_batched, i, num_cameras)

        eval_func_compact(
            iteration = i,
            jkey = subkey,
            item = item,
            iou_detection_stats = iou_detection_stats,
        )

    # Finalize
    for iou_threshold, stat in iou_detection_stats.items():
        stat.print_status(print_ap=True, print_prefix=f"[Log] Current @IoU>{iou_threshold:.2f}")
        
    inf_cls.summary()
    print("")

    postfix = f"{ablation_max_time_steps}steps"
    dump_result(eval_logs_dir, ckpt_dir, iou_detection_stats, inf_cls, postfix)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_dir", type=str, default="checkpoints/dif_model/02262024-180216_final")
    parser.add_argument("--eval_logs_dir", type=str, default="logs_eval_estimation/debug")
    parser.add_argument("--max_time_steps", type=int, default=5)
    args = parser.parse_args()
    main(args)