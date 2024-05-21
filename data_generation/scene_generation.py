from __future__ import annotations
import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"   
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
import glob
import numpy as np
import pybullet as p
from scipy.spatial.transform import Rotation as sciR
import ray
import jax
import jax.numpy as jnp
import time
import copy
import flax
import sys
from pathlib import Path
import csv

BASEDIR = Path(__file__).parent.parent
if str(BASEDIR) not in sys.path:
    sys.path.insert(0, str(BASEDIR))

import util.obj_util as outil
import util.transform_util as tutil
import util.camera_util as cutil
import util.cvx_util as cxutil
import itertools

import pkgutil

# Typing
from typing import Tuple, List
import numpy.typing as npt
# from imm.pybullet_util.typing_extra import UidT
UidT = int



def cam_posquat_sample(
        target_center: np.ndarray = np.zeros((3,)),
        cambody_id: UidT = None, 
        r_max: float = 1.2
) -> Tuple[npt.NDArray, npt.NDArray]:
    """Sample random camera pose from the upper hemisphere.

    Args:
        cambody_id (UidT, optional): Check collsion of the camera if provided. Defaults to None.
        r_max (float, optional): Radius of the sphere. Defaults to 1.7.

    Returns:
        cam_pos (npt.NDArray): New camera position
        cam_quat (npt.NDArray): New camera orientation
    """
    cam_invalid_pose = True
    while cam_invalid_pose:
        r = np.random.uniform(0.15, r_max)
        randz = np.zeros((3,))
        while randz[2] < 0.10:
            randz = np.random.normal(size=(3,))
            randz = randz / np.linalg.norm(randz)
        # target = np.random.uniform([-0.1,-0.1,-0.1], [0.1,0.1,0.1], size=(3,)) + target_center
        target = np.random.uniform([-0.07,-0.07,-0.07], [0.07,0.07,0.07], size=(3,)) + target_center
        # randy = np.random.normal(size=(3,))
        randy = np.array([0,0,1]) + np.random.normal(size=(3,)) * 0.10
        xaxis = np.cross(randy, randz)
        xaxis = xaxis / (np.linalg.norm(xaxis) + 1e-6)
        yaxis = np.cross(randz, xaxis)

        cam_R = np.stack([xaxis, yaxis, randz],axis=-1)
        cam_pos = randz * r + target
        cam_pos[:2] += np.random.normal(scale=0.1, size=(2,))

        if cam_pos[2] - target_center[2] < 0.04:
            continue
        cam_quat = sciR.from_matrix(cam_R).as_quat()

        if cambody_id is not None:
            p.resetBasePositionAndOrientation(cambody_id, cam_pos, cam_quat)
            p.performCollisionDetection()
            if len(p.getContactPoints(cambody_id)) == 0:
                cam_invalid_pose = False
        else:
            cam_invalid_pose = False

    return cam_pos, cam_quat


def cam_posquat_eval(
        target_center: np.ndarray = np.zeros((3,)),
        cambody_id: UidT = None, 
        r_max: float = 1.5
) -> Tuple[npt.NDArray, npt.NDArray]:
    """Sample random camera pose from the upper hemisphere, but sample space is now limited"""

    
    theta_limit = [np.pi/6, np.pi/4]
    phi_limit = [-np.pi/6, np.pi/6]
    theta = np.random.uniform(*theta_limit)
    phi = np.random.uniform(*phi_limit)
    target = target_center
    r = r_max

    cam_invalid_pose = True
    while cam_invalid_pose:

        # Spherical coordinate
        # target = np.random.uniform([-0.07,-0.07,-0.07], [0.07,0.07,0.07], size=(3,)) + target_center
        x = r * np.cos(phi) * np.cos(theta)
        y = r * np.sin(phi) * np.cos(theta)
        z = r * np.sin(theta)
        pos_vector = np.array([x, y, z])
        cam_pos = pos_vector + target

        # Rotation GL
        rot_z = pos_vector/np.linalg.norm(pos_vector)
        ref_y = np.array([0,0,1])
        rot_x = np.cross(ref_y, rot_z)
        rot_x = rot_x / (np.linalg.norm(rot_x) + 1e-6)
        rot_y = np.cross(rot_z, rot_x)
        cam_R = np.stack([rot_x, rot_y, rot_z],axis=-1)
        cam_quat = sciR.from_matrix(cam_R).as_quat()

        if cambody_id is not None:
            p.resetBasePositionAndOrientation(cambody_id, cam_pos, cam_quat)
            p.performCollisionDetection()
            if len(p.getContactPoints(cambody_id)) == 0:
                cam_invalid_pose = False
        else:
            cam_invalid_pose = False

    return cam_pos, cam_quat



def cam_posquat_sample_shelf(
        shelf_offset,
        cambody_id=None
) -> Tuple[npt.NDArray, npt.NDArray]:
    """Sample random camera pose based on the shelf object.

    """
    target = shelf_offset + np.random.uniform([-0.15,-0.05,0], [0.15,0.05,0.25], size=(3,))
    target = np.zeros(3,)
    cam_z_dir = sciR.from_euler('z', np.random.uniform(-np.pi/3,np.pi/3)) * sciR.from_euler('x', np.random.uniform(np.pi/2-np.pi/4,np.pi/2))
    cam_z_dir = cam_z_dir.as_matrix()[:,2]
    rand_r = np.random.uniform(0.45,1.3)
    cam_pos = target + cam_z_dir *rand_r
    
    randy = np.array([0,0,1.]) + 0.1*np.random.normal(size=(3,))
    # randy[2] = 2
    xaxis = np.cross(randy, cam_z_dir)
    xaxis = xaxis / (np.linalg.norm(xaxis) + 1e-6)
    yaxis = np.cross(cam_z_dir, xaxis)

    cam_R = np.stack([xaxis, yaxis, cam_z_dir],axis=-1)

    # cam_pos[:2] += np.random.normal(scale=0.1, size=(2,))

    cam_quat = sciR.from_matrix(cam_R).as_quat()


    if cambody_id is not None:
        p.resetBasePositionAndOrientation(cambody_id, cam_pos, cam_quat)
        p.performCollisionDetection()
        if len(p.getContactPoints(cambody_id)) == 0:
            cam_invalid_pose = False
    else:
        cam_invalid_pose = False

    return cam_pos, cam_quat


def intrinsic_sample(pixel_size: Tuple[int, int], type='all') -> cutil.IntrinsicT:
    """Sample feasible camera intrinsic

    Args:
        pixel_size (Tuple[int, int]): (height, width).
        type (str, optional): Camera type. Defaults to 'all'.

    Returns:
        IntrinsicT: Sampled intrinsic
    """
    if type=='d455':
        fov =np.random.uniform(59., 81.) # d455 / pixy 2.1
    elif type=='d435':
        fov =np.random.uniform(40., 45.) # d435
    elif type=='all':
        fov =np.random.uniform(40., 70.) # d455 / d435 / pixy 2.1
    intrinsic = cutil.pbfov_to_intrinsic(pixel_size, fov)
    return intrinsic



class SceneCls:

    @flax.struct.dataclass
    class SceneData:
        @flax.struct.dataclass
        class CamInfo:
            cam_posquats: npt.NDArray           # [..., #cam, 7]
            cam_intrinsics: npt.NDArray         # [..., #cam, 6]. (W, H, Fx, Fy, Cx, Cy)
        @flax.struct.dataclass
        class ObjInfo:
            obj_posquats: npt.NDArray           # [..., #obj, 7]
            obj_cvx_verts_padded: npt.NDArray   # [..., #obj, #cvx, #max_verts, 3], padded with 1e6
            obj_cvx_faces_padded: npt.NDArray   # [..., #obj, #cvx, #max_faces, 3], padded with -1
            scale: float
            uid_list: List[int]
            mesh_name: str|None
        @flax.struct.dataclass
        class NVRenInfo:
            obj_posquats: npt.NDArray           # [..., #obj, 7]
            scales: npt.NDArray
            mesh_name: str
            canonical_verts: npt.NDArray   # [..., #obj, #cvx, #max_verts, 3], padded with 1e6
            canonical_faces: npt.NDArray   # [..., #obj, #cvx, #max_faces, 3], padded with -1
        rgbs: npt.NDArray|None                  # [..., #cam, H, W, 3]
        depths: npt.NDArray|None                # [..., #cam, H, W]
        seg: npt.NDArray|None                # [..., #cam, H, W]
        uid_class :npt.NDArray|None
        cam_info: CamInfo                       # [..., #cam, ...]
        obj_info: ObjInfo                       # [..., #obj, ...]
        table_params: npt.NDArray               # [..., 3]
        robot_params: npt.NDArray
        nvren_info: NVRenInfo|None

    def __init__(
            self, 
            # object_set_dir_list: List[str], 
            dataset='NOCS',
            camera_type='all',
            max_obj_no: int = 3,
            fix_pos: bool = False, 
            shift_com: bool = True, 
            no_rgb: bool = False, 
            use_texture: bool = True, 
            col_obj_set_dir_list: List[str] = None, 
            robot_gen: bool = False, 
            pixel_size: Tuple[int, int] = [48,48], 
            gui: bool = False, 
            validation: bool = False, 
            scene_type: str = None
    ):
        self.camera_type = camera_type
        # self.obj_dir = object_set_dir_list
        if validation and dataset=='NOCS':
            obj_dir = glob.glob(os.path.join(BASEDIR, f'data/NOCS_val/32_64_1_v4/*.obj'))
        else:
            obj_dir = glob.glob(os.path.join(BASEDIR, f'data/{dataset}/32_64_1_v4_textured/*.obj'))

        self.class_unique_id = {'mug':0, 'bottle':1, 'camera':2, 'can':3, 'bowl':4, 'laptop':5}
        base_len = 0.65
        self.categorical_scale = {'can':1/base_len*0.15, 'bottle':1/base_len*0.15, 'bowl':1/base_len*0.2, 
                                    'camera':1/base_len*0.15, 'laptop':1/base_len*0.3, 'mug':1/base_len*0.12}
        meshname_scale = {}
        with open('data/class_id.csv', newline='') as csvfile:
            spamreader = csv.reader(csvfile)
            for i, row in enumerate(spamreader):
                meshname_scale[row[1]] = row[2]
        
        if validation and dataset=='NOCS':
            original_mesh_dir = glob.glob('data/NOCS_val/vis/*.obj')
        else:
            original_mesh_dir = glob.glob('data/NOCS/vis/*.obj')
        mesh_names = [Path(omd).name for omd in original_mesh_dir]

        self.scale_list = []
        self.obj_dir = []
        self.vis_obj_dir = []
        for od in obj_dir:
            meshname = os.path.basename(od)
            if validation and dataset=='NOCS':
                self.obj_dir.append(od)
                cat_name = meshname.split('-')[0]
                self.scale_list.append(float(self.categorical_scale[cat_name]))
                assert meshname in mesh_names
                self.vis_obj_dir.append(os.path.join(os.path.dirname(os.path.dirname(od)), 'vis', meshname))
            elif meshname in meshname_scale:
                self.obj_dir.append(od)
                self.scale_list.append(float(meshname_scale[meshname]))
                assert meshname[10:] in mesh_names
                self.vis_obj_dir.append(os.path.join(os.path.dirname(os.path.dirname(od)), 'vis', meshname[10:]))
                # self.vis_obj_dir.append(original_meshid_dir[meshname.split('-')[1][:-4]])
                # self.vis_obj_dir.append(os.path.join(os.path.dirname(os.path.dirname(od)), 'vis', 'bottle-70172e6afe6aff7847f90c1ac631b97f.obj'))
                
        self.max_obj_no = max_obj_no
        self.pixel_size = pixel_size
        self.no_rgb = no_rgb
        self.fix_pos = fix_pos
        self.shift_com = shift_com
        self.gui = gui
        self.robot_gen = robot_gen
        self.validation = validation
        self.scene_type = scene_type # shelf
        self.texture_dir_list = (
            glob.glob('data/texture/*/*/*.jpg') if use_texture 
            else None)
        self.col_obj_set_dir_dict = (
            {os.path.basename(cod):cod for cod in col_obj_set_dir_list} if col_obj_set_dir_list is not None
            else None)

        if self.gui:
            p.connect(p.GUI)
        else:
            p.connect(p.DIRECT)

        self.init()
    
    def init(self):
        """_summary_"""
        self.grasped_object_id = None
        self.table_id = None
        self.objcls_list = []
        self.cambody_id = None
        self.cam_pose_randomized_params = None
        p.setGravity(0,0,-9.81)

    def reset(self, reset=False):
        """_summary_

        Args:
            reset (bool, optional): Re-init all entities when `True`. Defaults to False.
        """
        if reset:
            p.resetSimulation()
            self.init()
        
        # Init camera body
        if self.cambody_id is None:
            self.cambody_id = p.createMultiBody(
                baseMass = 0.0, 
                baseCollisionShapeIndex = p.createCollisionShape(
                    p.GEOM_BOX, 
                    # halfExtents = [0.05,0.05,0.100]),
                    halfExtents = [0.05,0.05,0.160]),
                basePosition = [0,0,-1])
        p.resetBasePositionAndOrientation(self.cambody_id, [0,0,-1], [0,0,0,1])

        # Table
        #   Offset? randomization
        if self.fix_pos:
            self.table_offset = -0.20
        else:
            if self.validation:
                self.table_offset = np.random.uniform(-0.020, 0.020)
            else:
                self.table_offset = np.random.uniform(-0.080, 0.080)
        #   Table size randomization
        table_d = 2.0 # if set 0.03 -> cause egl render error : there is weird shadow in the floor
        if self.validation:
            table_half_width = np.random.uniform(0.5, 0.7, size=(2,))
        else:
            table_half_width = np.random.uniform(0.4, 1.5, size=(2,))
        if self.table_id is not None:
            p.removeBody(self.table_id)
        if self.scene_type == 'shelf':
            
            self.shelf_offset = np.random.uniform(-0.080, 0.080, size=(3,))
            # self.shelf_offset = np.zeros(3)

            mesh_pick = np.random.randint(0,6)
            if mesh_pick==0:
                shelf_name, shelf_scale, self.shelf_xlen, shelf_height = 'shelf-040', 1.8, 0.55, 0
            elif mesh_pick==1:
                shelf_name, shelf_scale, self.shelf_xlen, shelf_height = 'shelf-040', 1.8, 0.55, -0.48 # top layers
            elif mesh_pick==2:
                shelf_name, shelf_scale, self.shelf_xlen, shelf_height = 'shelf-045', 1.8, 0.4, -0.36 # second
            elif mesh_pick==3:
                shelf_name, shelf_scale, self.shelf_xlen, shelf_height = 'shelf-045', 1.8, 0.4, -0.65 # top
            elif mesh_pick==4:
                shelf_name, shelf_scale, self.shelf_xlen, shelf_height = 'wall-shelf-028', 1.8, 0.55, -0.04 # bottom
            elif mesh_pick==5:
                shelf_name, shelf_scale, self.shelf_xlen, shelf_height = 'wall-shelf-028', 1.8, 0.60, -0.52 # bottom

            shelf_position = self.shelf_offset + np.array([0,0,shelf_height])
            shelf_rotation = sciR.from_euler('x',np.pi/2).as_quat()

            visual_mesh_fn = f'data/shelf/{shelf_name}-obj/{shelf_name}.obj'
            self.table_id = p.createMultiBody(baseMass=0.0, basePosition=shelf_position, baseOrientation=shelf_rotation,
                                                baseCollisionShapeIndex=p.createCollisionShape(p.GEOM_MESH, fileName=f'data/shelf/32_64_1_v4/{shelf_name}.obj', meshScale=[shelf_scale,shelf_scale,shelf_scale]),
                                                baseVisualShapeIndex=p.createVisualShape(p.GEOM_MESH, fileName=visual_mesh_fn, meshScale=[shelf_scale,shelf_scale,shelf_scale]))

            self.table_params = {'shelf_name':visual_mesh_fn, 'shelf_position':shelf_position, 'shelf_rotation':shelf_rotation, 'shelf_scale':(shelf_scale,shelf_scale,shelf_scale)}
        
        else:
            self.table_id = p.createMultiBody(
                baseMass = 0.0, 
                basePosition = [0,0,-table_d],
                baseCollisionShapeIndex = p.createCollisionShape(p.GEOM_BOX, halfExtents=[*table_half_width,table_d]),
                baseVisualShapeIndex = p.createVisualShape(p.GEOM_BOX, halfExtents=[*table_half_width,table_d]))
            #   Table height randomization
            p.resetBasePositionAndOrientation(
                self.table_id, 
                posObj = [0,0,-table_d+self.table_offset], 
                ornObj = [0,0,0,1])
            self.table_params = np.array([self.table_offset, *list(table_half_width)])

        # ????
        if self.validation:
            p.changeDynamics(self.table_id, -1, lateralFriction = 0.4)
        else:
            p.changeDynamics(self.table_id, -1, lateralFriction = 1.0)

        if len(self.objcls_list) != 0:
            for oc in self.objcls_list:
                p.removeBody(oc.pbobj)
            self.objcls_list = []
        
        if self.max_obj_no >= 5:
            if np.random.uniform()<0.6:
                self.cur_obj_no = np.random.randint((self.max_obj_no*2)//3+1, self.max_obj_no+1)
            else:
                self.cur_obj_no = np.random.randint(1, (self.max_obj_no*2)//3+1)
        else:
            if np.random.uniform()<0.5:
                self.cur_obj_no = self.max_obj_no
            else:
                self.cur_obj_no = np.random.randint(1, self.max_obj_no+1)
        if self.validation: 
            self.cur_obj_no = self.max_obj_no
            center_offset = np.random.uniform([-0.05,-0.05, 0.0], [0.05,0.05, 0], size=(3,))
        else:
            center_offset = np.random.uniform([-0.25,-0.25, 0.0], [0.25,0.25, 0], size=(3,))

        if self.robot_gen:
            # add stick
            stick_len =np.random.uniform(0.8,0.9)
            stick_radius = np.random.uniform(0.015, 0.028)/2.
            stick_dir = (sciR.from_euler('z', np.random.uniform(0, 2*np.pi)) * sciR.from_euler('x', np.random.uniform(0, 45*np.pi/180.))).as_matrix()[:,2]
            stick_ori = sciR.from_matrix(tutil.line2Rm_np(stick_dir)).as_quat()
            stick_pos = center_offset + np.random.uniform([-0.2,-0.2, 0.003], [0.2,0.2,0.07], size=(3,))
            stick_pos[2] += self.table_offset + stick_len *0.5 * stick_dir[2] + np.linalg.norm(stick_dir[:2]) * stick_radius
            if self.grasped_object_id is not None:
                p.removeBody(self.grasped_object_id)    
            self.grasped_object_id = p.createMultiBody(baseMass=0.0, 
                                    basePosition=stick_pos,
                                    baseOrientation=stick_ori,
                                    baseCollisionShapeIndex=p.createCollisionShape(p.GEOM_CYLINDER, radius=stick_radius, height=stick_len),
                                    baseVisualShapeIndex=p.createVisualShape(p.GEOM_CYLINDER, radius=stick_radius, length=stick_len),
                                )
            self.grasped_object_params = np.array([stick_len, stick_radius, *list(stick_pos), *list(stick_ori)])
            p.changeDynamics(self.grasped_object_id, -1, lateralFriction = 0.8)
        else:
            self.grasped_object_params = None

        self.nvren_pq_list = np.zeros((self.max_obj_no,7))
        self.nvren_scales_list = np.zeros((self.max_obj_no,3))
        self.nvren_mesh_name_list = np.array([None for _ in range(self.max_obj_no)])
        while len(self.objcls_list) < self.cur_obj_no:
            obj_idx = np.random.randint(len(self.obj_dir))
            col_obj_fn = self.obj_dir[obj_idx]
            base_scale = self.scale_list[obj_idx]
            vis_obj_fn = self.vis_obj_dir[obj_idx]
            # if self.col_obj_set_dir_dict is not None:
            #     vis_obj_fn = obj_fn
            #     obj_name = os.path.basename(vis_obj_fn)
            #     col_obj_fn = self.col_obj_set_dir_dict[obj_name]
            # else:
            #     vis_obj_fn = obj_fn
            #     col_obj_fn = obj_fn

            # if self.scene_type =='shelf':
            #     scale = base_scale*np.random.uniform(0.6, 1.7, size=())
            #     # scale = scale * np.random.uniform(0.85, 1.2, size=(3,))
            #     ocls = outil.ObjCls(scale=scale, pbcolobj=col_obj_fn, pbvisobj=vis_obj_fn, shift_com=False)
            # elif self.validation:
            #     scale = base_scale*np.random.uniform(0.6, 1.7, size=())
            #     # scale = scale * np.random.uniform(0.85, 1.2, size=(3,))
            #     ocls = outil.ObjCls(scale=scale, pbcolobj=col_obj_fn, pbvisobj=vis_obj_fn, shift_com=self.shift_com)
            # else:
            #     scale = base_scale*np.random.uniform(0.6, 1.7, size=())
            #     # scale = scale * np.random.uniform(0.85, 1.2, size=(3,))
            #     ocls = outil.ObjCls(scale=scale, pbcolobj=col_obj_fn, pbvisobj=vis_obj_fn, shift_com=self.shift_com)
            
            scale = base_scale*np.random.uniform(0.6, 1.7, size=())
            ocls = outil.ObjCls(scale=scale, pbcolobj=col_obj_fn, pbvisobj=vis_obj_fn, shift_com=False)

            oid = ocls.pbobj
            if self.validation:
                p.changeDynamics(oid, -1, lateralFriction = 0.4)
            else:
                p.changeDynamics(oid, -1, lateralFriction = 1.0)
            invalid_pose = True
            cnt = 0
            aabb = np.stack(p.getAABB(oid), axis=0)
            while invalid_pose:
                if self.fix_pos:
                    opos, oquat = np.zeros((3,)), tutil.qrand(outer_shape=())
                else:
                    if np.random.uniform()<0.7:
                        z_ang = sciR.from_euler('z', np.random.uniform(-np.pi, np.pi, size=()))*sciR.from_quat(np.array([0.707, 0, 0, 0.707]))
                    else:
                        z_ang = sciR.from_euler('z', np.random.uniform(-np.pi, np.pi, size=()))*sciR.from_quat(np.array([0,0,0,1]))
                    oquat = z_ang.as_quat()
                    if self.scene_type == 'shelf':
                        opos = self.shelf_offset + np.random.uniform([-self.shelf_xlen,-0.1, 0.0], [self.shelf_xlen,0.1,0.25], size=(3,))
                    else:
                        opos = center_offset + np.random.uniform([-0.18,-0.18, 0.0], [0.18,0.18,1.5], size=(3,))
                if self.validation and self.scene_type != 'shelf':
                    # opos, oquat = np.random.uniform([-0.18,-0.18, 0.0], [0.18,0.18,1.5], size=(3,)), tutil.qrand(outer_shape=())
                    opos, oquat = np.random.uniform([-0.11,-0.11, 0.0], [0.11,0.11,0.4], size=(3,)), tutil.qrand(outer_shape=())
                p.resetBasePositionAndOrientation(oid, opos, oquat)
                p.performCollisionDetection()
                if len(p.getContactPoints(oid)) == 0:
                    invalid_pose = False
                cnt += 1
                if cnt >= 10:
                    break
            if cnt >= 10:
                p.removeBody(oid)
                continue
            
            self.nvren_scales_list[len(self.objcls_list)]=scale
            self.nvren_mesh_name_list[len(self.objcls_list)]=vis_obj_fn
            self.objcls_list.append(ocls)

        # step simulation
        pep = p.getPhysicsEngineParameters()
        for ts in range(int(5.0/pep['fixedTimeStep'])):
            p.stepSimulation()
        
        # check obj below plane
        new_obj = []
        new_scales = np.zeros((self.max_obj_no,3))
        new_mesh_name = np.array([None for _ in range(self.max_obj_no)])
        for ii, oid in enumerate([oc.pbobj for oc in self.objcls_list]):
            pos_, _ = p.getBasePositionAndOrientation(oid)
            if pos_[2] <= -0.2:
                if self.validation and self.scene_type!='shelf':
                    self.reset()
                    return
                p.removeBody(oid)
                self.cur_obj_no -= 1
            else:
                new_scales[len(new_obj)]=self.nvren_scales_list[ii]
                new_mesh_name[len(new_obj)]=self.nvren_mesh_name_list[ii]
                new_obj.append(self.objcls_list[ii])
        self.objcls_list = copy.deepcopy(new_obj)
        self.nvren_scales_list = copy.deepcopy(new_scales)
        self.nvren_mesh_name_list = copy.deepcopy(new_mesh_name)
        if self.cur_obj_no <= 0:
            self.reset()
            return
        
        # check obj rich contact
        if self.scene_type =='flat':
            cdist = []
            for ocl1, ocl2 in itertools.combinations(self.objcls_list, 2):
                closest_pnts = p.getClosestPoints(ocl1.pbobj, ocl2.pbobj, distance=0.001)
                if len(closest_pnts)!=0:
                    cdist.append(closest_pnts[0][8])

            if len(cdist) > 1:
                self.reset()

        if not self.validation and self.scene_type !='shelf':
            cdist = []
            for ocl1, ocl2 in itertools.combinations(self.objcls_list, 2):
                closest_pnts = p.getClosestPoints(ocl1.pbobj, ocl2.pbobj, distance=0.030)
                if len(closest_pnts)!=0:
                    cdist.append(closest_pnts[0][8])

            if len(self.objcls_list) >= 3 and len(cdist) < 1:
                self.reset()
            # contact_infos = p.getClosestPoints()
            # contact_dict = {}
            # for ci in contact_infos:
            #     if ci[0]!=self.table_id and ci[1]!=self.table_id:
            #         if self.robot_gen and (ci[0]==self.grasped_object_id or ci[1]==self.grasped_object_id):
            #             continue
            #         contact_dict[[ci[0],ci[1]]] = 1
        

        # get obj informations
        posquat_list = []
        cvx_pd_list = []
        cvx_fc_list = []
        canonical_cvx_pd_list = []
        canonical_cvx_fc_list = []
        for i, oc in enumerate(self.objcls_list):
            posquat_list.append(np.concatenate(p.getBasePositionAndOrientation(oc.pbobj), axis=-1))
            cvx_pd_list.append(oc.cvx_pd[0])
            cvx_fc_list.append(oc.cvx_pd[1])
            canonical_cvx_pd_list.append(oc.canonical_vtx_fc[0])
            canonical_cvx_fc_list.append(oc.canonical_vtx_fc[1])
            nvpos = sciR.from_quat(posquat_list[-1][-4:]).apply(-oc.baseInertialFramePosition) + posquat_list[-1][:3]
            self.nvren_pq_list[i]=np.concatenate([nvpos, posquat_list[-1][-4:]], axis=-1)
        for i in range(self.max_obj_no - len(self.objcls_list)):
            posquat_list.append(np.zeros_like(posquat_list[-1]))
            cvx_pd_list.append(1e6*np.ones_like(cvx_pd_list[-1]))
            cvx_fc_list.append(-np.ones_like(cvx_fc_list[-1]))
            canonical_cvx_pd_list.append(1e6*np.ones_like(canonical_cvx_pd_list[-1]))
            canonical_cvx_fc_list.append(-np.ones_like(canonical_cvx_fc_list[-1]))
        self.obj_posquat = np.stack(posquat_list, 0)
        self.obj_cvx_pd = np.stack(cvx_pd_list, 0)
        self.obj_cvx_fc = np.stack(cvx_fc_list, 0)
        self.canonical_cvx_pd = np.stack(canonical_cvx_pd_list, 0)
        self.canonical_cvx_fc = np.stack(canonical_cvx_fc_list, 0)

        # domain randomization
        p.changeVisualShape(self.table_id, -1, rgbaColor=list(np.random.uniform(0,1,size=(3,)))+[1])
        if self.robot_gen:
            p.changeVisualShape(self.grasped_object_id, -1, rgbaColor=list(np.random.uniform(0,1,size=(3,)))+[1])
        for oc in self.objcls_list:
            if np.random.uniform() < (0.65 if self.texture_dir_list is not None else 2.0):
                p.changeVisualShape(oc.pbobj, -1, rgbaColor=list(np.random.uniform(0,1,size=(3,)))+[1])
            else:
                tex_id = p.loadTexture(np.random.choice(self.texture_dir_list))
                p.changeVisualShape(oc.pbobj, -1, rgbaColor=[1,1,1,1], textureUniqueId=tex_id)

        # light dir works even with egl!
        # self.light_dir = (sciR.from_euler('x', np.random.uniform(-np.pi/3, np.pi/3))*sciR.from_euler('z', np.random.uniform(-np.pi, np.pi))).as_matrix()[:,2]
        light_mag = np.random.uniform(8, 12)
        self.light_dir = light_mag*(sciR.from_euler('x', np.random.uniform(-np.pi/3, np.pi/3))*sciR.from_euler('z', np.random.uniform(-np.pi, np.pi))).as_matrix()[:,2]
        return

    def get_onescene_data(
            self, 
            itr = 1, 
            intrinsic_in = None, 
            near = 0.010, 
            far = 3.0
    ) -> SceneData:

        rgb_list = []
        depth_list = []
        seg_list = []
        cam_posquat_list = []
        cam_intrinsic_list = []

        obj_pos_mean = [self.nvren_pq_list[i,:3] for i in range(len(self.objcls_list))]
        obj_pos_mean = np.mean(obj_pos_mean, 0)

        oid = np.array([oc.pbobj for oc in self.objcls_list])
        uid_list = -100*np.ones((self.max_obj_no,), dtype=np.int16)
        uid_list[:len(oid)] = oid

        for i in range(itr):
            # Generate dataset
            if self.scene_type == 'shelf':
                cam_posquat = cam_posquat_sample_shelf(self.shelf_offset, self.cambody_id)
            elif self.validation:
                # cam_pos = sciR.from_euler('z', np.random.uniform(0, 2*np.pi))*sciR.from_euler('x', np.array(45/180*np.pi))
                # cam_pos = cam_pos.as_matrix()[:,-1] * 0.6
                # target = np.random.uniform([-0.05,-0.05,-0.05], [0.05,0.05,0], size=(3,))
                cam_pos = sciR.from_euler('z', np.random.uniform(0, 2*np.pi))*sciR.from_euler('x', np.random.uniform(30/180*np.pi, 70/180*np.pi))
                cam_pos = cam_pos.as_matrix()[:,-1] * np.random.uniform(0.5, 0.8)
                target = np.random.uniform([-0.1,-0.1,-0.05], [0.1,0.1,0.05], size=(3,)) + obj_pos_mean
                cam_z = copy.deepcopy(cam_pos)
                cam_quat = np.array(tutil.line2q(cam_z, np.array([0,0,1])))
                cam_pos = cam_pos + target
                cam_posquat = (cam_pos, cam_quat)
            elif self.fix_pos:
                cam_posquat = cam_posquat_sample(obj_pos_mean, cambody_id=self.cambody_id, r_max=0.7)
            else:
                cam_posquat = cam_posquat_sample(obj_pos_mean, cambody_id=self.cambody_id)

            # Configure camera
            camH = np.zeros((4,4))
            camH[:3,:3] = sciR.from_quat(cam_posquat[1]).as_matrix()
            camH[:3,3] = cam_posquat[0]
            camH[3,3] = 1
            view_matrix = np.linalg.inv(camH).T
            view_matrix = view_matrix.reshape(-1)
            if intrinsic_in is None:
                intrinsic = intrinsic_sample(self.pixel_size, self.camera_type)
            else:
                intrinsic = intrinsic_in

            # Capture image
            projection_matrix = p.computeProjectionMatrix(
                *cutil.intrinsic_to_pb_lrbt(intrinsic, near=near), 
                nearVal = near, 
                farVal = far)
            img_out = p.getCameraImage(
                width = self.pixel_size[1], 
                height = self.pixel_size[0], 
                viewMatrix = view_matrix.reshape(-1), 
                projectionMatrix = projection_matrix, 
                shadow = 1,
                lightDirection = self.light_dir,
                renderer = p.ER_BULLET_HARDWARE_OPENGL)
            # seg = np.any(np.array(img_out[4])[...,None] == oid, axis=-1).astype(bool)
            seg = img_out[4]
            seg = np.where(np.any(np.array(img_out[4])[...,None] == oid, axis=-1), seg, -1)
            seg = np.where(jnp.logical_or(np.array(img_out[4]) == self.table_id, seg>=0), seg, -2).astype(np.int16)
            depth = np.array(img_out[3])
            depth = far * near / (far - (far - near) * depth)
            depth = depth.astype(np.float16)
            if not self.no_rgb:
                rgb = img_out[2]
                rgb = np.array(rgb)[...,:3].astype(np.uint8)
            else:
                rgb = np.array([None])
                # depth = np.array([None])
                # seg = np.array([None])
            
            cam_posquat = np.concatenate(cam_posquat, axis=-1)
            cam_intrinsic = np.array(intrinsic)
            
            # Aggregate
            rgb_list.append(rgb)
            depth_list.append(depth)
            seg_list.append(seg)
            cam_posquat_list.append(cam_posquat)
            cam_intrinsic_list.append(cam_intrinsic)

        # uid_class = np.array([self.class_unique_id[Path(mn).name.split('-')[0]] if mn is not None else None for mn in self.nvren_mesh_name_list])
        uid_class = -np.ones(shape=(self.max_obj_no+5))
        uid_class[-2] = -2
        for seq_, uid_ in enumerate(oid):
            uid_class[uid_] = self.class_unique_id[Path(self.nvren_mesh_name_list[seq_]).name.split('-')[0]]
        uid_class = uid_class.astype(np.int16)

        # import matplotlib.pyplot as plt
        # plt.figure()
        # plt.subplot(1,3,1)
        # plt.imshow(rgb_list[0])
        # plt.subplot(1,3,2)
        # plt.imshow(uid_class[np.stack(seg_list, 0)][0])
        # plt.subplot(1,3,3)
        # plt.imshow(depth_list[0])
        # plt.show()

        # Compose data
        datapoint = self.SceneData(
            rgbs = np.stack(rgb_list, 0),
            seg = np.stack(seg_list, 0),
            depths = np.stack(depth_list, 0),
            uid_class = uid_class,
            cam_info = self.SceneData.CamInfo(
                cam_posquats = np.stack(cam_posquat_list, 0).astype(np.float16),
                cam_intrinsics = np.stack(cam_intrinsic_list, 0).astype(np.float16)
            ),
            obj_info = self.SceneData.ObjInfo(
                obj_posquats = copy.deepcopy(self.obj_posquat).astype(np.float16),
                obj_cvx_verts_padded = copy.deepcopy(self.obj_cvx_pd).astype(np.float32),
                obj_cvx_faces_padded = copy.deepcopy(self.obj_cvx_fc).astype(np.int32),
                scale = copy.deepcopy(self.nvren_scales_list).astype(np.float16),
                mesh_name = np.array([Path(mn).name if mn is not None else None for mn in self.nvren_mesh_name_list]),
                uid_list = uid_list
            ),
            table_params = copy.deepcopy(self.table_params),
            nvren_info = self.SceneData.NVRenInfo(
                obj_posquats = copy.deepcopy(self.nvren_pq_list).astype(np.float16),
                scales = copy.deepcopy(self.nvren_scales_list).astype(np.float16),
                mesh_name = copy.deepcopy(self.nvren_mesh_name_list),
                canonical_verts = copy.deepcopy(self.canonical_cvx_pd).astype(np.float32),
                canonical_faces = copy.deepcopy(self.canonical_cvx_fc).astype(np.int32),
            ),
            robot_params = copy.deepcopy(self.grasped_object_params),
        )

        return datapoint

    def gen_batch_dataset(self, batch_size, view_no):
        data_list = []
        for i in range(batch_size):
            self.reset(i==0)
            data_list.append(self.get_onescene_data(itr=view_no))
        
        return jax.tree_map(lambda *x: np.stack(x, 0), *data_list)



# def main():
if __name__ == '__main__':
    import pickle
    # Setup import path
    import sys
    
    pixel_size = [64, 112]
    # odr_list = glob.glob('data/DexGraspNet/32_64_1_v4/*.obj')
    # odr_list = glob.glob('data/NOCS/32_64_1_v4_textured/*.obj')
    scene_ins = SceneCls(
        dataset='NOCS',
        camera_type='d435',
        pixel_size = pixel_size, 
        max_obj_no = 3, 
        gui = True,
        no_rgb = False, 
        use_texture = False,
        robot_gen = False,
        shift_com = False,
        validation = True, 
        scene_type = 'shelf')
    
    img, cam_info, obj_info, table_params, robot_params = scene_ins.gen_batch_dataset(1000, 5)

    env_no = 12
    batch_size = 64
    rgb_size = 3
    obj_no = 1
    fix_pos = False

    import datetime
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    ds_dir = f'scene_data_{rgb_size}_{obj_no}'
    if not os.path.exists(ds_dir):
        os.mkdir(ds_dir)
    
    datapnts_list = []
    for i in range(100000000):
        if i % 100 == 0:
            try:
                ray.shutdown()
            except:
                pass
            ray_actors = [ray.remote(SceneCls).remote(odr_list, obj_no, pixel_size=pixel_size, egl=False, gui=False, robot_gen=True, fix_pos=fix_pos) for _ in range(env_no)]
        imgs_list = ray.get([ra.gen_batch_datset.remote(batch_size, rgb_size) for ra in ray_actors])
        datapoints = jax.tree_util.tree_map(lambda *x: np.concatenate(x, 0), *imgs_list)
        datapnts_list.append(datapoints)
        if i%20==0 and i!=0:
            datapnts_list = jax.tree_util.tree_map(lambda *x: np.concatenate(x, 0), *datapnts_list)
            with open(os.path.join(ds_dir,f'{current_time}_{i}.pkl'), 'wb')as f:
                pickle.dump(datapnts_list, f)
            print(f'save datapoints {i}')
            datapnts_list = []
