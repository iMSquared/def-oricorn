'''
General API for motion planning environments
'''
import jax.numpy as jnp
import numpy as np
import jax
import flax
import numpy.typing as npt
from abc import ABC, abstractmethod
import dataclasses
from pathlib import Path
from typing import List

import util.cvx_util as cxutil
import util.model_util as mutil
import util.io_util as ioutil
import util.transform_util as tutil


def create_cvx_objects(
        jkey: jax.Array,
        obj_file_path_list: List[Path], 
        build: ioutil.BuildMetadata,
        scale = None,
        random_scale: bool = False
) -> cxutil.CvxObjects:
    """Create multiple objects at canonical origin. 
    
    Default pos will be the "volumetric center" relative to the mesh canoncial origin as [0,0,0].

    Args:
        jkey (jax.Array): Random generators
        obj_file_path_list (List[Path]): Path to the files to load
        build (ioutil.BuildMetadata): Mesh build
        random_scale (bool): Randomize scale when True.
    
    Returns:
        cxutil.CvxObjects: Created random objects with outer_shape=[n,]
    """

    # Sample random n objects with random scaling from the dataset
    jkey, scale_key = jax.random.split(jkey)
    num_objects = len(obj_file_path_list)
    
    # Load objects
    cvx_verts_list = []
    cvx_faces_list = []
    for obj_file_path in obj_file_path_list:
        verts, faces, verts_no, faces_no = cxutil.vex_obj_parsing(
            filename = str(obj_file_path), 
            max_dec_size = build.max_dec_num,
            max_vertices = build.max_vertices)
        cvx_verts_list.append(verts)
        cvx_faces_list.append(faces)
    cvx_verts_list = np.asarray(cvx_verts_list)
    cvx_faces_list = np.asarray(cvx_faces_list)

    # Random scaling
    valid_vtx_mask = cvx_verts_list<5e5
    if random_scale:
        scale = jax.random.uniform(scale_key, [num_objects, 1, 1, 3], minval=0.8, maxval=1.2)
        cvx_verts_list = jnp.where(valid_vtx_mask, cvx_verts_list*scale, cvx_verts_list)
    else:
        if scale is None:
            scale = 1
        # scale = jnp.full([num_objects, 1, 1, 3], 0.8)
        # cvx_verts_list = cvx_verts_list*scale
        cvx_verts_list = jnp.where(valid_vtx_mask, cvx_verts_list*scale, cvx_verts_list)

    # Init CvxObjects
    jkey, color_key = jax.random.split(jkey)
    color = jax.random.uniform(color_key, [num_objects, 3])
    objects = cxutil.CvxObjects().init_vtx(cvx_verts_list, cvx_faces_list)
    objects = objects.replace(color=color)

    return objects


def create_latent_objects(
        jkey: jax.Array,
        obj_file_path_list: List[Path],
        build: ioutil.BuildMetadata,
        models: flax.linen.Module,
        random_scale: bool = False,
) -> cxutil.LatentObjects:
    """Create multiple objects at canonical origin. 
    
    Default pos will be the "volumetric center" relative to the mesh canoncial origin as [0,0,0].

    Args:
        jkey (jax.Array): Random generators
        obj_file_path_list (List[Path]): Path to the files to load
        build (ioutil.BuildMetadata): Mesh build
        models (flax.linen.Module): Representation model
        random_scale (bool): Randomize scale when True.
    
    Returns:
        cxutil.LatentObjects: Created random objects with outer_shape=[n,]
    """
    jkey, zkey = jax.random.split(jkey)
    # Load objects
    objects = create_cvx_objects(jkey, obj_file_path_list, build, random_scale)
    # Init z vectors.
    objects = objects.set_z_with_models(zkey, models, keep_gt_info=True)

    return objects


def create_panda(
        jkey: jax.Array,
        panda_dir_path: Path,
        build: ioutil.BuildMetadata,
        models: flax.linen.Module,
) -> cxutil.LatentObjects:
    """Create panda links at canonical origin. 
    
    Default pos will be the "volumetric center" relative to the mesh canoncial origin as [0,0,0].

    Args:
        jkey (jax.Array): Random generators
        panda_dir_path (Path): Path to the panda directory
        build (ioutil.BuildMetadata): Mesh build
        models (flax.linen.Module): Representation model

    Returns:
        cxutil.LatentObjects: Created panda objects with outer_shape=[#links,]
    """
    # Parse franka
    panda_cvx_verts_list = []
    panda_cvx_faces_list = []
    panda_link_order = ['link0', 'link1', 'link2', 'link3', 'link4', 'link5', 'link6', 'link7', 'hand', 'finger', 'finger']
    for linkname in panda_link_order:
        verts, faces, verts_no, faces_no = cxutil.vex_obj_parsing(
            filename = str(panda_dir_path/f"{linkname}.obj"), 
            max_dec_size = build.max_dec_num,  # match convex size
            max_vertices = build.max_vertices)
        panda_cvx_verts_list.append(verts)
        panda_cvx_faces_list.append(faces)
    panda_cvx_verts_list = np.array(panda_cvx_verts_list)
    panda_cvx_faces_list = np.array(panda_cvx_faces_list)

    panda_link_obj = cxutil.CvxObjects().init_vtx(panda_cvx_verts_list, panda_cvx_faces_list)
    if models is None:
        return panda_link_obj
    panda_link_obj = panda_link_obj.set_z_with_models(jkey, models, keep_gt_info=True)
    panda_link_obj = dataclasses.replace(panda_link_obj, color=jnp.array([0.3,0.3,0.3]))
    panda_link_obj = panda_link_obj.broadcast(panda_link_obj.outer_shape)

    return panda_link_obj


def create_shelf(
        jkey: jax.Array,
        models: flax.linen.Module,
        build: ioutil.BuildMetadata
) -> cxutil.LatentObjects:
    """Wrapper for cxutil.create_shelf for latent inference.

    Args:
        jkey (jax.Array): Random generators
        models (flax.linen.Module): Representation model
        build (ioutil.BuildMetadata): Mesh build metadata

    Returns:
        cxutil.LatentObjects: Created objects with outer_shape=[#objs,]
    """
    shelf_obj = cxutil.create_shelf(
        num_dec=build.max_dec_num, 
        num_verts=build.max_vertices,
        height=0.4, width=0.4, depth=0.15, thinkness=0.08)
    shelf_obj = shelf_obj.set_z_with_models(jkey, models, keep_gt_info=True)

    return shelf_obj


def create_tray(
        jkey: jax.Array,
        models: flax.linen.Module,
        build: ioutil.BuildMetadata
) -> cxutil.LatentObjects:
    """Wrapper for cxutil.create_tray for latent inference.

    Args:
        jkey (jax.Array): Random generators
        models (flax.linen.Module): Representation model
        build (ioutil.BuildMetadata): Mesh build metadata

    Returns:
        cxutil.LatentObjects: Created objects with outer_shape=[#objs,]
    """
    tray_obj = cxutil.create_tray(
        num_dec=build.max_dec_num,
        num_verts=build.max_vertices)
    tray_obj = tray_obj.set_z_with_models(jkey, models, keep_gt_info=True)

    return tray_obj


def set_color(obj: cxutil.LatentObjects, color: jnp.ndarray) -> cxutil.LatentObjects:
    """Set color of the object

    Args:
        obj (cxutil.LatentObjects): Any shapes
        color (jnp.ndarray): [..., 3] or [3]

    Returns:
        cxutil.LatentObjects: Colored obj
    """
    obj = dataclasses.replace(obj, color=color)
    obj = obj.broadcast(obj.outer_shape)
    return obj




class BaseCollisionChecker(ABC):
    """Path continous collision detection(CCD)"""
    def __init__(
            self,
            models: mutil.Models,
    ):
        """Init params

        Args:
            models (mutil.Models): Collision predictor.
        """
        self.models = models

    def check_path(
            self,
            jkey: jax.Array,
            query_point: jnp.ndarray, 
            node_point: jnp.ndarray, 
            col_res_no: int = 100,
    ) -> jnp.ndarray:
        """Check path collision.

        Args:
            jkey (jax.Array): Random generator
            query_point (jnp.ndarray): New configuration (7)
            node_point (jnp.ndarray): Node configuration in RRT (7)
            col_res_no (int): CCD resolution
        """
        # Shape validation
        assert query_point.ndim == 1 and node_point.ndim == 1

        # Extension
        cdqpnts_ = query_point + np.linspace(0, 1, num=col_res_no)[...,None] * (node_point - query_point)
        # Get batched jkeys for collision detection
        jkey, *batched_col_jkeys = jax.random.split(jkey, col_res_no+1)
        batched_col_jkeys = jax.tree_map(lambda *x: jnp.stack(x, axis=0), *batched_col_jkeys)
        # CCD
        cd_res_, col_cost_ = jax.vmap(self.check_q)(batched_col_jkeys, cdqpnts_)
        cd_res_ = jnp.any(cd_res_, axis=-1)
        return cd_res_, jnp.max(col_cost_, -1)

    @abstractmethod
    def check_q(self, jkey: jax.Array, q: jnp.ndarray) -> jnp.ndarray:
        """Check collision at a configuration

        Try using vmap with this function for batched inference.

        Args:
            jkey (jax.Array): Random generator
            q (jnp.ndarray): A configuration to check. (7)
        Returns:
            jnp.ndarray
        """
        raise NotImplementedError("")