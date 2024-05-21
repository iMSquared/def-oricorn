import nvisii as nv
import numpy as np
import math
from typing import List, Tuple
import os
import pathlib
import re

from pybullet_utils.bullet_client import BulletClient
from bullet_envs.env import BulletEnv
from bullet_envs.camera import BulletCamera
from bullet_envs.objects import BulletObject
from bullet_envs.robot import BulletRobot
from PIL import Image


def parse_texture_path(obj_path) -> str:
    """Parse texture file from obj file"""
    obj_path = pathlib.Path(obj_path)
    mtl_filename = str(obj_path.stem)+".mtl"
    mtl_path = obj_path.parent/mtl_filename

    with open(mtl_path, 'r') as file:
        content = file.read()

    match = re.search(r'map_Kd\s+(.*?)\n', content)
    if match:
        return obj_path.parent/match.group(1)
    else:
        None


class BaseNVRenderer:
    """
    NVISII Renderer to make photo-realistic image from pybullet image.
    For the image, the robot is also rendered, but they are filtered during segmentation.
    """
    nv_initialized = False

    def __init__(self, bc: BulletClient, show_gui=False):
        """Prevent double initialization of NVISII instance."""
        
        # TODO: Make this singleton...
        if not BaseNVRenderer.nv_initialized:
            if show_gui:
                nv.initialize(window_on_top=True)
            else:
                nv.initialize(headless=True)
            nv.enable_denoiser()
            BaseNVRenderer.nv_initialized = True
        else:
            self.clear_all()

        self.bc = bc
        self.list_cam_entities   : List = None
        self.list_light_entities : List = None
        self.list_env_entities   : List = None
        self.list_robot_entities : List = None
        self.list_object_entities: List = None
        self.list_cam_width_height: List[Tuple[int, int]] = None # width, height
        self.SAMPLES_PER_PIXEL   : int = 300
        self.DOME_LIGHT_TEXTURE  : Tuple[float, float, float] = (0, 0, 0)
        self.LIGHT_INTENSITY     : float = 20.


    def configure_cameras(self, cameras: List[BulletCamera]):
        """Configure cameras in NVISII
        
        Destroy all previous cameras if already exists.

        Args:
            cameras (List[BulletCamera]): -
        """
        # Destroy
        if self.list_cam_entities is not None:
            for cam_entity in self.list_cam_entities:
                nv_cam = cam_entity.get_camera()
                pose = cam_entity.get_transform()
                nv_cam.remove(nv_cam.get_name())
                pose.remove(pose.get_name())
                cam_entity.remove(cam_entity.get_name())
            self.list_cam_entities = []
            self.list_cam_width_heights = []

        # Configure
        self.list_cam_entities     = []
        self.list_cam_width_height = []
        for i, cam in enumerate(cameras):
            # Camera config            
            name = f"entity_camera_{i}"
            # nv_cam = nv.camera.create_from_fov(
            #     name          = f"camera_{i}",
            #     field_of_view = math.radians(cam.intrinsic.fov),
            #     aspect        = cam.intrinsic.aspect)
            nv_cam = nv.camera.create_from_intrinsics(
                name = f"camera_{i}",
                fx = cam.intrinsic.formatted[2], 
                fy = cam.intrinsic.formatted[3], 
                cx = cam.intrinsic.formatted[4], 
                cy = cam.intrinsic.formatted[5], 
                width = cam.intrinsic.width, 
                height = cam.intrinsic.height, 
                near = cam.intrinsic.gl_near_val, 
                far = cam.intrinsic.gl_far_val
            )

            nv_cam.set_focal_distance(cam.intrinsic.focusing_distance)
            # OpenGL convention pose!
            position = nv.vec3(
                cam.extrinsic.camera_pos[0],
                cam.extrinsic.camera_pos[1],
                cam.extrinsic.camera_pos[2],)
            rotation = nv.quat(
                cam.extrinsic.camera_quat[3],
                cam.extrinsic.camera_quat[0],
                cam.extrinsic.camera_quat[1],
                cam.extrinsic.camera_quat[2])
            pose = nv.transform.create(
                name = f"pose_camera_{i}",
                position = position,
                rotation = rotation)
            # NVISII requires entity instance
            cam_entitiy = nv.entity.create(
                name      = name,
                camera    = nv_cam,
                transform = pose)
            self.list_cam_entities.append(cam_entitiy)
            self.list_cam_width_height.append(
                (cam.intrinsic.width, cam.intrinsic.height))


    def configure_lights(self, use_sky_dome: bool = False):
        """Create lights"""
        # Destroy
        if self.list_light_entities is not None:
            raise NotImplementedError()
        # Configure
        if use_sky_dome:
            nv.set_dome_light_sky(sun_position = (10, 10, 1), saturation = 1)
            nv.set_dome_light_exposure(1)
        else:
            nv.set_dome_light_color(self.DOME_LIGHT_TEXTURE)
        
        light1 = nv.entity.create(
            name      = "light1",
            transform = nv.transform.create("light1"),
            mesh      = nv.mesh.create_plane("light1", flip_z = True),
            light     = nv.light.create("light1"))
        light2 = nv.entity.create(
            name      = "light2",
            transform = nv.transform.create("light2"),
            mesh      = nv.mesh.create_plane("light2", flip_z = True),
            light     = nv.light.create("light2"))
        light3 = nv.entity.create(
            name = "light3",
            transform = nv.transform.create("light3"),
            mesh = nv.mesh.create_plane("light3", flip_z = True),
            light = nv.light.create("light3"))
        light4 = nv.entity.create(
            name      = "light4",
            transform = nv.transform.create("light4"),
            mesh      = nv.mesh.create_plane("light4", flip_z = True),
            light     = nv.light.create("light4"))
        # light5 = nv.entity.create(
        #     name      = "light5",
        #     transform = nv.transform.create("light5"),
        #     mesh      = nv.mesh.create_plane("light5", flip_z = True),
        #     light     = nv.light.create("light5"))
        # light6 = nv.entity.create(
        #     name      = "light6",
        #     transform = nv.transform.create("light6"),
        #     mesh      = nv.mesh.create_plane("light6", flip_z = True),
        #     light     = nv.light.create("light6"))
        # light7 = nv.entity.create(
        #     name      = "light7",
        #     transform = nv.transform.create("light7"),
        #     mesh      = nv.mesh.create_plane("light7", flip_z = True),
        #     light     = nv.light.create("light7"))
        # light8 = nv.entity.create(
        #     name      = "light8",
        #     transform = nv.transform.create("light8"),
        #     mesh      = nv.mesh.create_plane("light8", flip_z = True),
        #     light     = nv.light.create("light8"))
        # light9 = nv.entity.create(
        #     name      = "light9",
        #     transform = nv.transform.create("light9"),
        #     mesh      = nv.mesh.create_plane("light9", flip_z = True),
        #     light     = nv.light.create("light9"))
        light1.get_light().set_intensity(self.LIGHT_INTENSITY)
        light2.get_light().set_intensity(self.LIGHT_INTENSITY)
        light3.get_light().set_intensity(self.LIGHT_INTENSITY)
        light4.get_light().set_intensity(self.LIGHT_INTENSITY)
        # light5.get_light().set_intensity(self.LIGHT_INTENSITY)
        # light6.get_light().set_intensity(self.LIGHT_INTENSITY)
        # light7.get_light().set_intensity(self.LIGHT_INTENSITY)
        # light8.get_light().set_intensity(self.LIGHT_INTENSITY)
        # light9.get_light().set_intensity(self.LIGHT_INTENSITY)
        
        light1.get_transform().set_position((0, 0, 2.5))
        light1.get_transform().set_scale((0.5, 0.5, 0.5))
        light2.get_transform().set_position((0, 1, 2.5))
        light2.get_transform().set_scale((0.5, 0.5, 0.5))
        light3.get_transform().set_position((0, -1, 2.5))
        light3.get_transform().set_scale((0.5, 0.5, 0.5))
        light4.get_transform().set_position((-3, 0, -1))
        light4.get_transform().set_scale((0.5, 0.5, 0.5))
        # light5.get_transform().set_position((-1, -3, 2.5))
        # light5.get_transform().set_scale((0.25, 0.25, 0.25))
        # light6.get_transform().set_position((1, -3, 2.5))
        # light6.get_transform().set_scale((0.25, 0.25, 0.25))
        # light7.get_transform().set_position((0, -5, 2.5))
        # light7.get_transform().set_scale((0.25, 0.25, 0.25))
        # light8.get_transform().set_position((-1, -5, 2.5))
        # light8.get_transform().set_scale((0.25, 0.25, 0.25))
        # light9.get_transform().set_position((1, -5, 2.5))
        # light9.get_transform().set_scale((0.25, 0.25, 0.25))
        # light7.get_transform().set_position((0, -10, 2.5))
        # light7.get_transform().set_scale((0.25, 0.25, 0.25))
        # light8.get_transform().set_position((-1, -10, 2.5))
        # light8.get_transform().set_scale((0.25, 0.25, 0.25))
        # light9.get_transform().set_position((1, -10, 2.5))
        # light9.get_transform().set_scale((0.25, 0.25, 0.25))

        self.list_light_entities = [light1, light2, light3, light4]


    def capture_rgb(self):
        
        list_rgb = []

        for cam_entity, (width, height) in zip(self.list_cam_entities, self.list_cam_width_height):
            nv.set_camera_entity(cam_entity)
            os.makedirs("tmp", exist_ok=True)
            path = f"tmp/nvisii_output.png"
            nv.render_to_file(width=width, height=height, samples_per_pixel=self.SAMPLES_PER_PIXEL, file_path=path)

            rgb = np.asarray(Image.open(path).convert('RGB'))
            os.remove(path)

            list_rgb.append(rgb)

        return list_rgb


    def clear_all(self):
        nv.clear_all()
        self.list_cam_entities    = None
        self.list_light_entities  = None
        self.list_env_entities    = None
        self.list_robot_entities  = None
        self.list_object_entities = None


    def configure_environment(self, env: BulletEnv):
        for env_uid in env.env_uids:
            env_entity = self._load_body_visual_shape(env_uid)
            self.list_env_entities = list(env_entity.items())

    
    def configure_robot(self, robot: BulletRobot):
        robot_entity = self._load_body_visual_shape(robot.uid)
        self.list_robot_entities = list(robot_entity.items())

    
    def configure_objects(self, objects: List[BulletObject]):
        self.list_object_entities = []
        for obj in objects:
            obj_entity = self._load_body_visual_shape(obj.uid)
            self.list_object_entities += list(obj_entity.items())
        
    
    def _load_body_visual_shape(self, body_uid: int):
        """Load uid into nvisii.
        
            - Current implementation do not reuse the mesh when reloading the same body, thus is slow.
            - It has memory leak... make sure to call nv.clear_all() occasionally.

        Args:
            body_uid (int): -
        """

        nv_objects = {}

        for vis_index, vis_data in enumerate(self.bc.getVisualShapeData(body_uid)):

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
                dynamics_info = self.bc.getDynamicsInfo(body_uid,-1)
                inertial_frame_position    = dynamics_info[3]   # Relative to the link frame.
                inertial_frame_orientation = dynamics_info[4]
                base_state = self.bc.getBasePositionAndOrientation(body_uid)
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
                link_state = self.bc.getLinkState(body_uid, link_index)
                world_link_frame_position    = link_state[4]
                world_link_frame_orientation = link_state[5]

            # Name to use for components
            object_name = f"{body_uid}_{link_index}_{vis_index}"
            if (nv.transform.get(object_name) is not None or 
                object_name in nv_objects):
                raise ValueError(f"Uid={body_uid}, link={link_index}, vis_index={vis_index} already exists in NVISII.")        

            # Create mesh 
            mesh_asset_file_name = mesh_asset_file_name.decode("UTF-8")
            try:
                texture_asset_file_name = parse_texture_path(mesh_asset_file_name)
            except:
                texture_asset_file_name = None

            if visual_geometry_type == self.bc.GEOM_MESH:
                mesh = nv.mesh.create_from_file(
                    object_name,
                    os.path.abspath(mesh_asset_file_name))
                nv_objects[object_name] = nv.entity.create(
                    name      = object_name,
                    mesh      = mesh,
                    transform = nv.transform.create(object_name),
                    material  = nv.material.create(object_name))
                if texture_asset_file_name is not None:
                    texture = nv.texture.create_from_file(name=object_name+"_texture", path=str(texture_asset_file_name))
                    nv_objects[object_name].get_material().set_base_color_texture(texture)
            elif visual_geometry_type == self.bc.GEOM_BOX:
                assert len(mesh_asset_file_name) == 0
                nv_objects[object_name] = nv.entity.create(
                    name = object_name,
                    mesh = nv.mesh.create_box(
                        name = object_name,
                        size = nv.vec3(dimensions[0] / 2, dimensions[1] / 2, dimensions[2] / 2)), # half dim in NVISII v.s. pybullet
                    transform = nv.transform.create(object_name),
                    material  = nv.material.create(object_name))
            elif visual_geometry_type == self.bc.GEOM_CYLINDER:
                assert len(mesh_asset_file_name) == 0
                length = dimensions[0]
                radius = dimensions[1]
                nv_objects[object_name] = nv.entity.create(
                    name = object_name,
                    mesh = nv.mesh.create_cylinder(
                        name   = object_name,
                        radius = radius,
                        size   = length / 2), # size in nvisii is half of the length in pybullet
                    transform = nv.transform.create(object_name),
                    material  = nv.material.create(object_name))
            elif visual_geometry_type == self.bc.GEOM_SPHERE:
                assert len(mesh_asset_file_name) == 0
                nv_objects[object_name] = nv.entity.create(
                    name = object_name,
                    mesh = nv.mesh.create_sphere(
                        name   = object_name,
                        radius = dimensions[0]),
                    transform = nv.transform.create(object_name),
                    material  = nv.material.create(object_name))
            else:
                continue

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
            if visual_geometry_type == self.bc.GEOM_MESH:
                nv_objects[object_name].get_transform().set_scale(dimensions)
            nv_objects[object_name].get_material().set_base_color((
                rgba_color[0] ** 2.2,
                rgba_color[1] ** 2.2,
                rgba_color[2] ** 2.2,))
            
                
        return nv_objects
