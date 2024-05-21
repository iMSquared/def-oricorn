from __future__ import annotations
import torch
import jax
import jax.numpy as jnp
import flax
import numpy as np
import open3d as o3d
import einops
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Hashable, Tuple, Callable, Dict
from copy import deepcopy
import matplotlib.pyplot as plt
from pytorch3d.ops import box3d_overlap
import numpy.typing as npt
from scipy.spatial.transform.rotation import Rotation as R
import logging


import util.cvx_util as cxutil
import util.pcd_util as pcdutil

# Environment
from imm_pb_util.imm.pybullet_util.common import get_link_pose, imagine, get_transform_matrix_with_scale
from imm_pb_util.imm.pybullet_util.vision import (
    CoordinateVisualizer, visualize_point_cloud, draw_bounding_box, transform_points,
    unwrap_o3d_pcd, wrap_o3d_pcd
)

@dataclass
class GtInstanceData:
    """PCD, OBB is defined in canoncial coordinate."""
    pcd_np: object
    pcd_o3d: object
    obb_o3d: object
    pos: np.ndarray[float]  # [3]
    quat: np.ndarray[float] # [4]
    symmetricity: bool

    def picklable(self) -> GtInstanceData:
        pickable_copy = GtInstanceData(
            deepcopy(self.pcd_np),
            None,
            None,
            deepcopy(self.pos),
            deepcopy(self.quat),
            deepcopy(self.symmetricity)
        )
        return pickable_copy


@dataclass
class PredInstanceData:
    """PCD, OBB is transformed to world coordinate."""
    pcd_np: object
    pcd_o3d: object
    obb_o3d: object
    confidence: float

    def picklable(self) -> PredInstanceData:
        pickable_copy = PredInstanceData(
            deepcopy(self.pcd_np),
            None,
            None,
            self.confidence
        )
        return pickable_copy


@dataclass
class MatchedPair:
    pred: PredInstanceData
    gt: GtInstanceData
    iou: float
    chamfer_distance: float

    def picklable(self) -> MatchedPair:
        pickable_copy = MatchedPair(
            self.pred.picklable(),
            self.gt.picklable(),
            self.iou,
            self.chamfer_distance
        )
        return pickable_copy
    

class DetectionStats:

    def __init__(
            self, 
            iou_threshold: float,
    ):
        # Thresholds
        self.iou_threshold = iou_threshold
        # Records (AP)
        self.matched_pairs_per_sample: dict[int, list[MatchedPair]] = {}
        self.unmatched_preds_per_sample: dict[int, list[PredInstanceData]] = {}
        self.unmatched_gts_per_sample: dict[int, list[GtInstanceData]] = {}

        self.eps = 1e-9

    def update(
            self,
            sample_index: int,
            new_matched_pairs: list[MatchedPair],
            new_unmatched_preds: list[PredInstanceData],
            new_unmatched_gts: list[GtInstanceData],
    ):
        # For AP
        self.matched_pairs_per_sample[sample_index] = new_matched_pairs
        self.unmatched_preds_per_sample[sample_index] = new_unmatched_preds
        self.unmatched_gts_per_sample[sample_index] = new_unmatched_gts


    def get_all_preds_and_gts_classified(self):
        all_matched_pairs = [pair for v in self.matched_pairs_per_sample.values() for pair in v]
        all_unmatched_preds = [pred for v in self.unmatched_preds_per_sample.values() for pred in v]
        all_unmatched_gts = [gt for v in self.unmatched_gts_per_sample.values() for gt in v]
        return all_matched_pairs, all_unmatched_preds, all_unmatched_gts
    
    def get_all_preds_and_gts_unclassified(self):
        all_matched_pairs, all_unmatched_preds, all_unmatched_gts = self.get_all_preds_and_gts_classified()
        all_matched_preds = [pair.pred for pair in all_matched_pairs]
        all_matched_gts = [pair.gt for pair in all_matched_pairs]
        all_preds = all_unmatched_preds + all_matched_preds
        all_gts = all_unmatched_gts + all_matched_gts
        return all_preds, all_gts

    def get_best_f1_score(self):
        matched_pairs, unmatched_preds, unmatched_gts = self.get_all_preds_and_gts_classified()
        prec, rec, f1, conf = get_best_prec_rec_f1(matched_pairs, unmatched_preds, unmatched_gts)
        return prec, rec, f1, conf
    
    def get_raw_f1_score(self):
        matched_pairs, unmatched_preds, unmatched_gts = self.get_all_preds_and_gts_classified()
        EPS = 1e-9
        n_tp = len(matched_pairs)
        n_pred = len(matched_pairs) + len(unmatched_preds)
        n_gt = len(matched_pairs) + len(unmatched_gts)
        rec = n_tp / n_gt
        prec = n_tp / n_pred
        f1 = (2*prec*rec) / (prec+rec+EPS)
        return prec, rec, f1

    
    def get_average_precision(self) -> float:
        all_matched_pairs, all_unmatched_preds, all_unmatched_gts = self.get_all_preds_and_gts_classified()
        result = get_pascal_voc_metrics(all_matched_pairs, all_unmatched_preds, all_unmatched_gts)
        ap = result["AP"]
        return ap

    def get_cd_and_iou(self, confidence_threshold):
        all_matched_pairs, all_unmatched_preds, all_unmatched_gts = self.get_all_preds_and_gts_classified()
        thresholded_pair = [pair for pair in all_matched_pairs if pair.pred.confidence >= confidence_threshold]
        avg_cd = np.mean([pair.chamfer_distance for pair in thresholded_pair])
        avg_iou = np.mean([pair.iou for pair in thresholded_pair])
        return avg_cd, avg_iou

    def print_status(
            self, 
            print_ap: bool = False, 
            print_best_f1: bool = False, 
            print_raw_f1: bool = False,
            print_cd: bool = True, 
            print_prefix: str = "", 
            output_logging: bool = False
    ):
        str_out = f"{print_prefix} "
        if print_ap:
            ap = self.get_average_precision()
            str_out += f"AP = {ap:.4f}"
        if print_best_f1:
            prec, rec, f1, conf = self.get_best_f1_score()
            str_out += f", @Conf>{conf:.2f} Prec/Rec/F1/ = ({prec:1.4f} / {rec:1.4f} / {f1:1.4f}) "
        if print_raw_f1:
            prec, rec, f1 = self.get_raw_f1_score()
            str_out += f", @Raw Prec/Rec/F1/ = ({prec:1.4f} / {rec:1.4f} / {f1:1.4f}) "
        if print_cd:
            cd, iou = self.get_cd_and_iou(-1)
            str_out += f", cd = {cd:.4f}, iou = {iou:.4f}"

        print(str_out)
        if output_logging:
            logging.info(str_out)
        return str_out
    
    def pickable(self) -> DetectionStats:

        pickable_matched_pairs = {}
        pickable_unmatched_preds = {}
        pickable_unmatched_gts = {}
        for k, v in self.matched_pairs_per_sample.items():
            pickable_matched_pairs[k] = [pair.picklable() for pair in v ]
        for k, v in self.unmatched_preds_per_sample.items():
            pickable_unmatched_preds[k] = [pred.picklable() for pred in v ]
        for k, v in self.unmatched_gts_per_sample.items():
            pickable_unmatched_gts[k] = [gt.picklable() for gt in v ]

        picklable_copy = DetectionStats(self.iou_threshold)
        picklable_copy.matched_pairs_per_sample = pickable_matched_pairs
        picklable_copy.unmatched_preds_per_sample = pickable_unmatched_preds
        picklable_copy.unmatched_gts_per_sample = pickable_unmatched_gts
        return picklable_copy


def get_canonical_gt_instance_data(
        obj_path_list: List[Path],
        obj_pos_list: List[np.ndarray]|np.ndarray,
        obj_quat_list: List[np.ndarray]|np.ndarray,
        obj_scale_list: List[float]|np.ndarray,
        obj_symmetricity_list: List[bool]|np.ndarray,
        num_sample_points
) -> List[GtInstanceData]:
    """Sample canonical pcd, obb from mesh path + world pose
    
    Raises:
        Exception: Box convexhull failure
    """
    gt_instance_canonical_list = []
    loop = zip(obj_path_list, obj_pos_list, obj_quat_list, obj_scale_list, obj_symmetricity_list)
    for obj_path, obj_pos, obj_quat, obj_scale, obj_symmetricity in loop:
        # GT mesh + scaling
        mesh_transform = get_transform_matrix_with_scale([0,0,0], [0,0,0,1], obj_scale)
        mesh_o3d = o3d.io.read_triangle_mesh(str(obj_path))
        mesh_o3d.transform(mesh_transform)
        # PCD sampling
        pcd_o3d = mesh_o3d.sample_points_uniformly(num_sample_points)
        pcd_np = unwrap_o3d_pcd(pcd_o3d)
        # OBB sampling
        obb_o3d = pcd_o3d.get_minimal_oriented_bounding_box()
        gt_instance_canonical = GtInstanceData(
            pcd_np, pcd_o3d, obb_o3d, 
            obj_pos, obj_quat, obj_symmetricity)
        gt_instance_canonical_list.append(gt_instance_canonical)

    return gt_instance_canonical_list


def get_transformed_latent_pred(
        jkey: jax.Array,
        obj_latent: cxutil.LatentObjects,
        pred_conf: np.ndarray[float],
        pcd_sample_func: Callable,
        min_volume: float = 1e-6,
) -> List[PredInstanceData]:
    """Sample from latent obj
    
    Raises:
        Exception: Box convexhull failure
    """
    # Get pred pcd and bbox
    jkey, subkey = jax.random.split(jkey)
    if obj_latent.outer_shape[0] == 0:
        pred_instance_list = []
        print("Empty prediction")
        return pred_instance_list
    else:
        pcd_batch = pcd_sample_func(subkey, obj_latent)
        pcd_batch = np.asarray(pcd_batch)

    pred_instance_list = []
    for obj_pcd, conf in zip(pcd_batch, pred_conf):
        # OBB sampling
        obj_pcd_o3d = wrap_o3d_pcd(obj_pcd)
        obj_obb_o3d = obj_pcd_o3d.get_minimal_oriented_bounding_box()
        # Drop small objects
        if obj_obb_o3d.volume() < min_volume:
            print("Droping too small object prediction")
            continue
        # Collect
        pred_instance = PredInstanceData(
            obj_pcd, obj_pcd_o3d, obj_obb_o3d, conf)
        pred_instance_list.append(pred_instance)

    return pred_instance_list


def get_pairwise_obb_iou(
        gt_instance_canonical_list: List[GtInstanceData],
        pred_instance_list: List[PredInstanceData],
        iou_eps: float = 1e-6,
        symmetry_obb_aug_res: int = 180
):
    """Wrap torch3d obb iou for o3d

    Symmetric objects will best-match among the rotated obbs.

    Args:
        gt_instance_canonical_list (List[GtInstanceData]): List of gt instances, size of M
        pred_instance_list (List[PredInstanceData]): List of pred instances, size of N
        iou_eps (float): IoU epsilon for torch3d. IoU below this will raise error. Coplanar error above this will raise error.
        symmetry_obb_aug_res (int): Rotation bbox augmentation resolution along the axis for symmetric groundtruth.
    
    Raises:
        Exception: Invalid box

    Returns:
        np.ndarray: NxM array iou matrix, pred -> gt.
        np.ndarray: Rotation of the GT along the symmetric axis at the best IoU.
    """

    M = len(gt_instance_canonical_list)
    N = len(pred_instance_list)
    if N == 0:
        return np.zeros(shape=(N, M)), np.zeros(shape=(N, M))
    
    # square bbox -> only augmenting 90 degree is enough
    aug_rot_axis = np.asarray([0., 1., 0.]) # y-up in mesh frame
    rot_angles = np.linspace(0, np.pi/2., symmetry_obb_aug_res//4)
    aug_rot_aa_list = aug_rot_axis*rot_angles[...,None]
    aug_rot_list = R.from_rotvec(aug_rot_aa_list)

    # Vectorize bboxes
    gt_obb_list = []        # [M, 8, 3]
    gt_obj_indices = []     # [M]
    gt_obj_aug_angle_list = []   # [M]
    for m, gt_instance in enumerate(gt_instance_canonical_list):
        corners_canonical = np.asarray(gt_instance.obb_o3d.get_box_points())
        # Compose rotations
        if gt_instance.symmetricity == True:
            # [1] * [M] -> [M] broadcast
            gt_rot = R.from_quat(gt_instance.quat)
            gt_rot_aug = gt_rot * aug_rot_list 
            gt_quat_aug = gt_rot_aug.as_quat()
            gt_obj_aug_angle = np.copy(rot_angles)  # Will later be used...
        else:
            gt_quat_aug = np.expand_dims(gt_instance.quat, axis=0)
            gt_obj_aug_angle = np.array([0])
        # Transform obb. [8,3] -> [M, 8, 3]
        corners_aug = [transform_points(corners_canonical, gt_instance.pos, q) for q in gt_quat_aug]        
        # Concat
        gt_obb_list.append(np.array(corners_aug))
        gt_obj_indices.append(np.full(shape=[len(corners_aug)], fill_value=m))
        gt_obj_aug_angle_list.append(gt_obj_aug_angle)
        
    # Vectorized~!
    gt_obb_list = np.concatenate(gt_obb_list)
    gt_obj_indices = np.concatenate(gt_obj_indices)
    gt_obj_aug_angle_list = np.concatenate(gt_obj_aug_angle_list)

    # Vectorize pred box
    pred_obb_list = []  # [N, 8, 3]
    for pred_instance in pred_instance_list:
        corners = np.asarray(pred_instance.obb_o3d.get_box_points())
        pred_obb_list.append(corners)
    pred_obb_list = np.stack(pred_obb_list)

    # 3D Box IoU (pred -> gt)
    gt_obb_list = torch.tensor(gt_obb_list, dtype=torch.float32).cuda()
    gt_obb_list = convert_obb_corner_order_o3d_to_torch3d(gt_obb_list)
    pred_obb_list = torch.tensor(pred_obb_list, dtype=torch.float32)
    pred_obb_list = convert_obb_corner_order_o3d_to_torch3d(pred_obb_list).cuda()
    intersection_vol, iou_3d = box3d_overlap(pred_obb_list, gt_obb_list, iou_eps)
    iou_3d = iou_3d.cpu()

    # Collapse augmentation obb
    iou_3d_collapsed = np.zeros((N, M))
    gt_rot_angle_collapsed = np.zeros((N, M))
    for m in range(M):
        # Find best match
        iou_3d_pred_to_m = np.where(gt_obj_indices[None,:] == m, iou_3d, 0)     # [N, #aug] from [N, #total]
        pred_idx = np.arange(N)
        best_gt_idx = np.argmax(iou_3d_pred_to_m, axis=-1)                      # [N]
        iou_3d_collapsed[:,m] = iou_3d_pred_to_m[pred_idx, best_gt_idx]         # [N, M]
        
        # Return with augmentation angle
        gt_obj_aug_angle_pred_to_m = np.where(gt_obj_indices == m, gt_obj_aug_angle_list, 0)    # [N]
        gt_rot_angle_collapsed[:,m] = gt_obj_aug_angle_pred_to_m[best_gt_idx]                   # [N, M]

    return iou_3d_collapsed, gt_rot_angle_collapsed


def get_pairwise_pcd_cd(
        gt_instance_canonical_list: List[GtInstanceData],
        pred_instance_list: List[PredInstanceData],
        chamfer_fn: Callable
) -> np.ndarray:
    """
    Args:
        gt_instance_canonical_list (List[GtInstanceData]): List of gt instances, size of M
        pred_instance_list (List[PredInstanceData]): List of pred instances, size of N

    Returns:
        np.ndarray: NxM array iou matrix, pred -> gt.
    """

    M = len(gt_instance_canonical_list)
    N = len(pred_instance_list)

    gt_pcd_all = []  # [M, #points, 3]
    for gt in gt_instance_canonical_list:
        # Transform GT pcd to world frame
        t_mat = get_transform_matrix_with_scale(gt.pos, gt.quat)
        gt_pcd_o3d_transformed = deepcopy(gt.pcd_o3d).transform(t_mat)
        # PCDs in world frame
        gt_obj_pcd = unwrap_o3d_pcd(gt_pcd_o3d_transformed)
        gt_pcd_all.append(gt_obj_pcd)
    pred_pcd_all = np.array([pred.pcd_np for pred in pred_instance_list])   # [N, #points, 3]

    gt_pcd_all = np.tile(gt_pcd_all, reps=[N,1,1]) # (1,2, ..., 1,2, ..., 1,2, ...) N times
    pred_pcd_all = np.repeat(pred_pcd_all, repeats=M, axis=0)      # (1,1, ... 2,2, ..., 3,3, ...) M times

    match = chamfer_fn(pred_pcd_all, gt_pcd_all)
    match = np.reshape(match, newshape=(N,M))
    return match


def get_match_by_3d_iou(
        gt_instance_canonical_list: List[GtInstanceData],
        pred_instance_list: List[PredInstanceData],
        iou_3d: np.ndarray, 
        iou_threshold: float,
        chamfer_fn: Callable,
        gt_rot_angle_table: np.ndarray,
        debug: bool = False
):
    """Match iou following PASCAL VOC + calculate chamfer between matches."""

    # Exception handling
    if len(pred_instance_list) == 0:
        matched_pair_list = []
        unmatched_pred_list = []
        unmatched_gt_list = gt_instance_canonical_list
        return matched_pair_list, unmatched_pred_list, unmatched_gt_list

    # IoU thresholding
    iou_3d = np.where(iou_3d > iou_threshold, iou_3d, 0.)

    # Match Pred->GT. Drop duplicate pred.
    gt_already_matched = set()
    tp_pair_pred_gt_indices: List[Tuple[int, int]] = []
    for pred_idx, pred_to_gt_iou in enumerate(iou_3d):
        # Find best
        matched_gt_idx = np.argmax(pred_to_gt_iou)
        # Invalid case
        if np.sum(pred_to_gt_iou) == 0:
            continue
        if matched_gt_idx in gt_already_matched:
            continue
        # Match TP.
        gt_already_matched.add(matched_gt_idx)
        true_positive_pair = (pred_idx, matched_gt_idx)
        tp_pair_pred_gt_indices.append(true_positive_pair)

    # Find unmatched pred and gt.
    matched_pred_indices = [item[0] for item in tp_pair_pred_gt_indices]
    matched_gt_indices = [item[1] for item in tp_pair_pred_gt_indices]
    num_preds = iou_3d.shape[0]
    num_gts = iou_3d.shape[1]
    unmatched_pred_indices = [i for i in range(num_preds) if i not in matched_pred_indices]
    unmatched_gt_indices = [i for i in range(num_gts) if i not in matched_gt_indices]
    # Compose list of instances
    unmatched_pred_list = [pred_instance_list[i] for i in unmatched_pred_indices]
    unmatched_gt_list = [gt_instance_canonical_list[i] for i in unmatched_gt_indices]

    # Get matched metric
    matched_pair_list: List[MatchedPair] = []
    for pred_idx, gt_idx in tp_pair_pred_gt_indices:
        # Transform GT pcd to world frame
        gt_instance = gt_instance_canonical_list[gt_idx]
        t_mat = get_transform_matrix_with_scale(gt_instance.pos, gt_instance.quat)
        gt_pcd_o3d_transformed = deepcopy(gt_instance.pcd_o3d).transform(t_mat)
        # Pred pcd in already in world frame.
        pred_instance = pred_instance_list[pred_idx]
        pred_pcd_o3d = pred_instance.pcd_o3d
        # PCDs in world frame
        gt_obj_pcd = unwrap_o3d_pcd(gt_pcd_o3d_transformed)
        pred_obj_pcd = unwrap_o3d_pcd(pred_pcd_o3d)
        # Measure chamfer distance
        cd = chamfer_fn(gt_obj_pcd, pred_obj_pcd).item()
        iou = iou_3d[pred_idx, gt_idx]

        # Define rotation in canonical coord.
        rot_angle = gt_rot_angle_table[pred_idx, gt_idx]
        aug_rot_axis = np.asarray([0., 1., 0.]) # y-up in mesh frame
        aug_rot_aa = aug_rot_axis * rot_angle
        aug_rot = R.from_rotvec(aug_rot_aa).as_matrix()
        # Rotating bboxes...
        gt_instance_bbox_rotated = deepcopy(gt_instance)
        gt_instance_bbox_rotated.obb_o3d.rotate(aug_rot)

        # Compose
        pair = MatchedPair(pred_instance, gt_instance_bbox_rotated, iou, cd)
        matched_pair_list.append(pair)

        # Debug
        if debug:
            print(f"[Debug] instance conf={pred_instance.confidence} cd={cd:1.5f}, IoU={iou:.5f}")
            # visualize_point_cloud(
            #     [pred_obj_pcd_transformed, gt_obj_pcd_transformed], 
            #     lower_lim=-1, upper_lim=1, save=True, 
            #     save_path=f"eval_visualization/ours-eval{k}-{NUM_OBJ}objects.png")
            visualize_point_cloud(
                [gt_obj_pcd, pred_obj_pcd], 
                lower_lim=-1, upper_lim=1)
            print("")

    return matched_pair_list, unmatched_pred_list, unmatched_gt_list


def convert_obb_corner_order_o3d_to_torch3d(corners: torch.Tensor) -> torch.Tensor:
    """Convert convention
    
    open3d
        (2) +---------+. (5)
            | ` .     |  ` .
            | (0) +---+-----+ (3)
            |     |   |     |
        (7) +-----+---+. (4)|
            ` .   |     ` . |
            (1) ` +---------+ (6)

    torch3D
        (4) +---------+. (5)
            | ` .     |  ` .
            | (0) +---+-----+ (1)
            |     |   |     |
        (7) +-----+---+. (6)|
            ` .   |     ` . |
            (3) ` +---------+ (2)

    Args:
        corners (torch.Tensor): OBB corner in o3d format. Shape=[..., 8, 3]
    
    Return:
        torch.Tensor: OBB corners in torch3d format. Shape=[..., 8, 3]
    """
    o3d_to_torch3d = [0, 3, 6, 1, 2, 5, 4, 7]
    corners_new = corners[..., o3d_to_torch3d, :]
    return corners_new


def draw_confidence_distribution(confidences: np.ndarray):
    """For debug purpose"""
    # plt.clf()
    plt.switch_backend('TkAgg')
    plt.figure(figsize=(6,1))
    width=0.4
    # y = np.zeros(confidences.shape[0]) + (np.random.uniform(size=confidences.shape[0])*width-width/2.)
    y = np.zeros(confidences.shape[0])
    plt.scatter(confidences, y)
    plt.ylim([-width/2, width/2])
    plt.tight_layout()
    # plt.xlim([-5, 5])
    plt.show()



def get_best_prec_rec_f1(
        matched_pairs: list[MatchedPair],
        unmatched_preds: list[PredInstanceData],
        unmatched_gts: list[GtInstanceData],
):
    """asdf"""
    EPS = 1e-9
    matched_predictions = [pair.pred for pair in matched_pairs]
    matched_gts = [pair.gt for pair in matched_pairs]
    all_predictions = sorted(matched_predictions + unmatched_preds, key=lambda pred: pred.confidence, reverse=True) # Descending confidence
    all_gts = matched_gts + unmatched_gts

    # Get only ground truths of class c, use filename as key
    npos = len(all_gts)

    # Sort detections by decreasing confidence
    TP = np.zeros(len(all_predictions), dtype=int)
    FP = np.zeros(len(all_predictions), dtype=int)

    # Loop through detections
    conf_list = []
    for i, pred in enumerate(all_predictions):
        # Classify TP or FP
        if check_same_reference_existence(pred, matched_predictions):
            TP[i] = 1
        else:
            FP[i] = 1
        conf_list.append(pred.confidence)

    # Compute precision, recall and average precision
    acc_FP = np.cumsum(FP)
    acc_TP = np.cumsum(TP)
    rec = acc_TP / npos
    prec = np.divide(acc_TP, (acc_FP + acc_TP))
    f1 = (2*prec*rec) / (prec+rec+EPS)

    best_idx = np.argmax(f1)
    best_prec, best_rec, best_f1, best_conf = prec[best_idx], rec[best_idx], f1[best_idx], conf_list[best_idx]
    return best_prec, best_rec, best_f1, best_conf






# Below here is borrowed from https://github.com/rafaelpadilla/Object-Detection-Metrics/
def check_same_reference_existence(instance, list):
    for item in list:
        if instance is item:
            return True
    else:
        return False

def get_pascal_voc_metrics(
        matched_pairs: list[MatchedPair],
        unmatched_preds: list[PredInstanceData],
        unmatched_gts: list[GtInstanceData],
):
    """Get the average precision used by the VOC Pascal 2012 challenge."""
    matched_predictions = [pair.pred for pair in matched_pairs]
    matched_gts = [pair.gt for pair in matched_pairs]
    all_predictions = sorted(matched_predictions + unmatched_preds, key=lambda pred: pred.confidence, reverse=True) # Descending confidence
    all_gts = matched_gts + unmatched_gts

    # Get only ground truths of class c, use filename as key
    npos = len(all_gts)

    # Sort detections by decreasing confidence
    TP = np.zeros(len(all_predictions), dtype=int)
    FP = np.zeros(len(all_predictions), dtype=int)

    # Loop through detections
    for i, pred in enumerate(all_predictions):
        # Classify TP or FP
        if check_same_reference_existence(pred, matched_predictions):
            TP[i] = 1
        else:
            FP[i] = 1

    # Compute precision, recall and average precision
    acc_FP = np.cumsum(FP)
    acc_TP = np.cumsum(TP)
    rec = acc_TP / npos
    prec = np.divide(acc_TP, (acc_FP + acc_TP))
    # Depending on the method, call the right implementation
    ap, mpre, mrec, ii = calculate_average_precision(rec, prec)

    result = {
        'precision': prec,
        'recall': rec,
        'AP': ap,
        'interpolated precision': mpre,
        'interpolated recall': mrec,
        'total positives': npos,
        'total TP': np.sum(TP),
        'total FP': np.sum(FP)
    }
    return result


def plot_precision_recall_curve(
        matched_pairs: list[MatchedPair],
        unmatched_preds: list[PredInstanceData],
        unmatched_gts: list[GtInstanceData],
        show_interpolated_precision = True,
        name: str = None,
):
    """PlotPrecisionRecallCurve
    Plot the Precision x Recall curve for a given class.
    Args:
        show_interpolated_precision (optional): if True, it will show in the plot the interpolated
            precision (default = False);
    """
    result = get_pascal_voc_metrics(matched_pairs, unmatched_preds, unmatched_gts)

    precision = result['precision']
    recall = result['recall']
    average_precision = result['AP']
    mpre = result['interpolated precision']
    mrec = result['interpolated recall']
    npos = result['total positives']
    total_tp = result['total TP']
    total_fp = result['total FP']

    if show_interpolated_precision:
        # plt.plot(mrec, mpre, '--r', label='Interpolated precision (every point)')
        plt.plot(mrec, mpre, '--r')
    # plt.xlim([0, 0.5])
    plt.ylim([0, 1])
    label = f"Precision" if name is None else name
    plt.plot(recall, precision, label=label)
    plt.xlabel('recall')
    plt.ylabel('precision')
    ap_str = "{0:.4f}%".format(average_precision * 100)
    # plt.title('Precision x Recall curve \nAP: %s' % (ap_str))
    plt.legend(shadow=True)
    plt.grid()
    # plt.show()

    return result


def calculate_average_precision(rec, prec):
    mrec = []
    mrec.append(0)
    [mrec.append(e) for e in rec]
    mrec.append(1)
    mpre = []
    mpre.append(0)
    [mpre.append(e) for e in prec]
    mpre.append(0)
    for i in range(len(mpre) - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    ii = []
    for i in range(len(mrec) - 1):
        if mrec[1+i] != mrec[i]:
            ii.append(i + 1)
    ap = 0
    for i in ii:
        ap = ap + np.sum((mrec[i] - mrec[i - 1]) * mpre[i])
    # return [ap, mpre[1:len(mpre)-1], mrec[1:len(mpre)-1], ii]
    return [ap, mpre[0:len(mpre) - 1], mrec[0:len(mpre) - 1], ii]