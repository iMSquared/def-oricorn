import jax
import jax.numpy
import os
import numpy as np
import pybullet as pb
import jax.numpy as jnp
from typing import List
import logging


import util.io_util as ioutil
import util.inference_util as ifutil


# Environment
from imm_pb_util.imm.pybullet_util.common import get_link_pose
from imm_pb_util.imm.pybullet_util.vision import CoordinateVisualizer
from imm_pb_util.bullet_envs.env import SimpleEnv
from imm_pb_util.bullet_envs.robot import FrankaPanda, PandaGripper
from imm_pb_util.bullet_envs.manipulation import FrankaManipulation
from imm_pb_util.bullet_envs.objects import BulletObject
from imm_pb_util.bullet_envs.camera import SimpleCamera, RobotAttachedCamera
from imm_pb_util.bullet_envs.nvisii import BaseNVRenderer


class IntegratedModulesOurs:

    def __init__(
            self, 
            ckpt_dir: str,
            robot_base_pos,
            robot_base_quat, 
            use_uncertainty: bool,
            logs_dir: str, 
    ):
        # Estimator configuration
        num_samples = 32        # Batch size
        cam_z_offset = 0.5
        conf_threshold = -0.5
        max_time_steps = 5
        overlay_obj_no = 2 if use_uncertainty else 1    # For uncertainty RRT
        self.inf_cls = ifutil.InfCls(
            ckpt_dir = ckpt_dir,
            ns = num_samples,
            conf_threshold = conf_threshold,
            cam_offset = np.array([0,0,cam_z_offset]),
            save_images = True,
            max_time_steps = max_time_steps,
            apply_in_range = True,
            early_reduce_sample_size = overlay_obj_no,
            optim_lr=4e-3,
            optimization_step=100,
            gradient_guidance=False,
            guidance_grad_step=4,
            overay_obj_no = overlay_obj_no,
            scene_obj_no=7,
            log_dir = logs_dir
        )

        # Configure estimator renderer
        num_cameras = 3
        self.pixel_size = (64, 112)
        vis_pixel_size = (120, 212)
        self.inf_cls.compile_jit(num_cameras)
        self.inf_cls.init_renderer(vis_pixel_size)

        # Motion planner configuration
        RRT_NB = 1
        node_size = 500
        self.rrt_refinement = True
        self.rrt_time_measures = []
        self.franka_rrt = ifutil.FrankaRRT(
            self.inf_cls.models, 
            robot_base_pos,
            robot_base_quat, 
            logs_dir, 
            nb=RRT_NB, 
            node_size=node_size
        )
        self.franka_rrt.comfile_jit()

        # Profiling
        self.save_time_intervals = {
            "est": [],
            "opt": [],
            "rrt": [],
            "total": [],
        }


    def plan(
            self, 
            jkey: jax.Array, 
            multiview_rgb: np.ndarray,
            multiview_cam: List[SimpleCamera],
            initial_q: np.ndarray,
            joint_pos_start: np.ndarray,
            joint_pos_goal: np.ndarray,
            plane_params: np.ndarray,
            verbose: int = 1,
    ):  
        # Reformat
        cam_intrinsics = []
        cam_posquats = []
        for cam in multiview_cam:
            intrinsics = cam.intrinsic.formatted # (W, H, Fx, Fy, Cx, Cy)
            pos = cam.extrinsic.camera_pos
            quat = cam.extrinsic.camera_quat
            cam_intrinsics.append(intrinsics)
            cam_posquats.append(np.concatenate([pos, quat]))
        cam_intrinsics = np.stack(cam_intrinsics, axis=0)
        cam_posquats = np.stack(cam_posquats, axis=0)


        # Estimate
        print("estimation start")
        jkey, subkey = jax.random.split(jkey)
        obj_pred_sorted, conf_sorted, _, aux_time, _ = self.inf_cls.estimation(
            subkey, self.pixel_size, multiview_rgb, cam_intrinsics, cam_posquats, 
            out_valid_obj_only=False, # False for benchmark
            filter_before_opt=True,
            verbose = 0
        )

        # RRT
        print("rrt start")
        with ioutil.context_profiler("rrt_time", print_log=False) as rrt_time_data:
            jkey, subkey = jax.random.split(jkey)
            traj, origin_traj, goal_reached, aux = self.franka_rrt.execution(
                jkey = subkey, 
                env_objs = obj_pred_sorted, 
                initial_q = jnp.array(joint_pos_start), 
                goal_pq = None, 
                goal_q = jnp.array(joint_pos_goal), 
                plane_params=plane_params, 
                gripper_width = 0.,
                refinement = self.rrt_refinement, 
                video_fn = None
            )

        # Benchmarking
        jax.block_until_ready(traj)
        self.__log_rrt_time(
            est_time = aux_time[0],
            opt_time = aux_time[1],
            rrt_time = rrt_time_data.duration, 
            verbose = verbose)

        return obj_pred_sorted, conf_sorted, traj
    

    def __log_rrt_time(self, est_time: float, opt_time: float, rrt_time: float, verbose: int):
        """Log inference times"""
        # Log time
        total_time = est_time + opt_time + rrt_time
        self.save_time_intervals['est'].append(est_time)
        self.save_time_intervals['opt'].append(opt_time)
        self.save_time_intervals['rrt'].append(rrt_time)
        self.save_time_intervals['total'].append(total_time)
        if verbose > 0:
            print(f'total time: {total_time:.3f} // estimation time: {est_time:.3f} // optimization time: {opt_time:.3f} // rrt time: {rrt_time:.3f}')
            logging.info(f'total time: {total_time:.3f} // estimation time: {est_time:.3f} // optimization time: {opt_time:.3f} // rrt time: {rrt_time:.3f}')
        
        # Log averaged time
        if len(self.save_time_intervals['total']) > 1:
            time_summary = [
                f"{k} avg={np.mean(self.save_time_intervals[k][1:]):.3f}, std={np.std(self.save_time_intervals[k][1:]):.3f} "
                for k in ['total', 'est', 'opt', 'rrt']
            ]
            time_summary = f"cummulative time: {time_summary[0]} // {time_summary[1]} // {time_summary[2]} // {time_summary[3]}"
            if verbose > 0:
                print(time_summary)
                logging.info(time_summary)

    
    def summary(self):
        est_time = np.array(self.save_time_intervals['est'])
        opt_time = np.array(self.save_time_intervals['opt'])
        rrt_time = np.array(self.save_time_intervals['rrt'])
        total_time = est_time + opt_time + rrt_time

        # NOTE: exclude jit time
        print(f'avg est: {np.mean(est_time[1:]):.3f} opt: {np.mean(opt_time[1:]):.3f} rrt: {np.mean(rrt_time[1:]):.3f}')
        print(f'std est: {np.std(est_time[1:]):.3f} opt: {np.std(opt_time[1:]):.3f} rrt: {np.std(rrt_time[1:]):.3f}')
        print(f'total avg: {np.mean(total_time[1:]):.3f}')
        print(f'total std: {np.std(total_time[1:]):.3f}')

