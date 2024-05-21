import numpy as np
from typing import Dict
from pybullet_utils.bullet_client import BulletClient
import os


import util.cvx_util as cxutil


def looking_good_color(outer_shape=[]):
    while True:
        color = np.random.uniform(size=list(outer_shape)+[3,])
        if np.mean(color) > 0.4:
            break
    return color


def create_pb_multibody_from_cvx_object(bc: BulletClient, obj: cxutil.CvxObjects) -> int:
    """No outer shape is allowed.
    
    Use this function for visualization only. Fixed object will be made.
    """
    if len(obj.outer_shape) != 0:
        raise ValueError("Outer shape must be None")
    
    cvx_vtx = np.asarray(obj.rel_vtx[obj.vtx_valid_mask.repeat(3, axis=2)]).reshape(-1, 3)
    cvx_fcs = np.asarray(obj.fc[obj.fc_valid_mask.repeat(3, axis=2)]).reshape(-1)
    vshape = bc.createVisualShape(bc.GEOM_MESH, vertices=cvx_vtx, indices=cvx_fcs, rgbaColor=obj.color.tolist()+[1,])
    cshape = bc.createCollisionShape(bc.GEOM_MESH, vertices=cvx_vtx, indices=cvx_fcs)
    uid = bc.createMultiBody(
        0, cshape, vshape, [0,0,0], [0,0,0,1])    # link frame = inertial frame
    bc.resetBasePositionAndOrientation(uid, obj.pos, [0,0,0,1]) # this resets inertial frame

    return uid


def create_pb_multibody_from_file(
        bc: BulletClient, 
        file_path: str, 
        pos: np.ndarray, 
        orn: np.ndarray, 
        color: np.ndarray, 
        scale: np.ndarray,        
        fixed_base=False,
        use_visual_mesh=False,
) -> int:
    """This function is simulation compatible.
    
    """
    if use_visual_mesh:
        # Split the file path into its parts
        path_parts = file_path.split(os.sep)

        # Replace the second-to-last part with the new parent directory
        path_parts[-2] = 'vis'

        # Join the parts back into a modified file path
        vis_file_path = os.sep.join(path_parts)
        vshape = bc.createVisualShape(bc.GEOM_MESH, fileName=vis_file_path, rgbaColor=color.tolist()+[1,], meshScale=scale)
    else:
        vshape = bc.createVisualShape(bc.GEOM_MESH, fileName=file_path, rgbaColor=color.tolist()+[1,], meshScale=scale)
    cshape = bc.createCollisionShape(bc.GEOM_MESH, fileName=file_path, meshScale=scale)
    mass = 0 if fixed_base else 0.1
    uid = bc.createMultiBody(
        mass, cshape, vshape, [0,0,0], [0,0,0,1])    # link frame = inertial frame
    bc.resetBasePositionAndOrientation(uid, pos, orn) # this resets inertial frame

    return uid


def load_cvx_to_nvisii(bc: BulletClient, body_uid: int, cvx_obj: cxutil.CvxObjects, obj_fn=None) -> Dict:
    """Load pb object created from CvxObjects into nvisii.
    
    Args:
        cvx_obj: No outer shape is allowed.
    """
    import nvisii as nv
    nv_objects = {}

    for vis_index, vis_data in enumerate(bc.getVisualShapeData(body_uid)):

        # Unpack visual shape data
        link_index                     = vis_data[1]
        visual_geometry_type           = vis_data[2]
        dimensions                     = vis_data[3]
        mesh_asset_file_name           = vis_data[4]
        local_visual_frame_position    = vis_data[5]
        local_visual_frame_orientation = vis_data[6]
        rgba_color                     = vis_data[7]

        # Compose the pose of link frame in world frame.
        if link_index == -1:    # Base link
            # Parse link frame and inertial frame pose
            dynamics_info = bc.getDynamicsInfo(body_uid,-1)
            inertial_frame_position    = dynamics_info[3]   # Relative to the link frame.
            inertial_frame_orientation = dynamics_info[4]
            base_state = bc.getBasePositionAndOrientation(body_uid)
            world_link_frame_position    = base_state[0]
            world_link_frame_orientation = base_state[1]
            # Compose world position
            m1 = nv.translate(
                nv.mat4(1), 
                nv.vec3(
                    inertial_frame_position[0], 
                    inertial_frame_position[1], 
                    inertial_frame_position[2]))
            m1 = m1 * nv.mat4_cast(
                nv.quat(
                    inertial_frame_orientation[3], 
                    inertial_frame_orientation[0], 
                    inertial_frame_orientation[1], 
                    inertial_frame_orientation[2]))
            m2 = nv.translate(
                nv.mat4(1), 
                nv.vec3(
                    world_link_frame_position[0], 
                    world_link_frame_position[1], 
                    world_link_frame_position[2]))
            m2 = m2 * nv.mat4_cast(
                nv.quat(
                    world_link_frame_orientation[3], 
                    world_link_frame_orientation[0], 
                    world_link_frame_orientation[1], 
                    world_link_frame_orientation[2]))
            # m = nv.inverse(m1) * m2
            m = m2 * nv.inverse(m1)
            q = nv.quat_cast(m)
            world_link_frame_position = m[3]
            world_link_frame_orientation = q
        else:   # Other links
            link_state = bc.getLinkState(body_uid, link_index)
            world_link_frame_position    = link_state[4]
            world_link_frame_orientation = link_state[5]

        # Name to use for components
        object_name = f"{body_uid}_{link_index}_{vis_index}"
        if (nv.transform.get(object_name) is not None or 
            object_name in nv_objects):
            raise ValueError(f"Uid={body_uid}, link={link_index}, vis_index={vis_index} already exists in NVISII.")        

        # Create mesh 
        if obj_fn is None:
            cvx_vtx = np.asarray(cvx_obj.rel_vtx[cvx_obj.vtx_valid_mask.repeat(3, axis=2)]).reshape(-1)
            cvx_fcs = np.asarray(cvx_obj.fc[cvx_obj.fc_valid_mask.repeat(3, axis=2)]).reshape(-1)
            mesh = nv.mesh_create_from_data(
                name = object_name,
                positions = cvx_vtx,
                indices = cvx_fcs
            )
        else:
            mesh = nv.mesh_create_from_file(
                name = object_name, path=str(obj_fn)
            )
        nv_objects[object_name] = nv.entity.create(
            name      = object_name,
            mesh      = mesh,
            transform = nv.transform.create(object_name),
            material  = nv.material.create(object_name))
    
        # Link transform
        m1 = nv.translate(
            nv.mat4(1), 
            nv.vec3(
                world_link_frame_position[0], 
                world_link_frame_position[1], 
                world_link_frame_position[2]))
        m1 = m1 * nv.mat4_cast(
            nv.quat(
                world_link_frame_orientation[3], 
                world_link_frame_orientation[0], 
                world_link_frame_orientation[1], 
                world_link_frame_orientation[2]))
        # Visual frame transform
        m2 = nv.translate(
            nv.mat4(1), 
            nv.vec3(
                local_visual_frame_position[0], 
                local_visual_frame_position[1], 
                local_visual_frame_position[2]))
        m2 = m2 * nv.mat4_cast(
            nv.quat(
                local_visual_frame_orientation[3], 
                local_visual_frame_orientation[0], 
                local_visual_frame_orientation[1], 
                local_visual_frame_orientation[2]))

        # For entities created for primitive shapes
        nv_objects[object_name].get_transform().set_transform(m1 * m2)
        if visual_geometry_type == bc.GEOM_MESH:
            nv_objects[object_name].get_transform().set_scale(dimensions)
        nv_objects[object_name].get_material().set_base_color((
            rgba_color[0] ** 2.2,
            rgba_color[1] ** 2.2,
            rgba_color[2] ** 2.2,))
        
    return nv_objects
