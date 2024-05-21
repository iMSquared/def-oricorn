import os 
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"  # add this
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform" 
os.environ["CUDA_VISIBLE_DEVICES"] = "0"    # NVISII will crash when showed multiple devices with jax.

import nvisii
import random
import numpy as np
import glob
import jax
from scipy.spatial.transform import Rotation as sciR
from PIL import Image

import util.cvx_util as cxutil
import util.transform_util as tutil
import util.camera_util as cutil
import util.render_util as rutil
import data_generation.scene_generation as sg


from typing import Tuple, List
import numpy.typing as npt
from dataclasses import dataclass


class NvisiiRender(object):

    @dataclass(frozen=True, eq=True)
    class RenderOption:
        spp: int
        width: int
        height: int
        use_denoiser: bool
        out_filename: str

    def __init__(
            self, 
            pixel_size: Tuple[int, int], 
            texture_dir: str,
            hdr_dir: str, 
    ):
        self.option = self.RenderOption(
            spp = 16,
            width = pixel_size[1],
            height = pixel_size[0],
            use_denoiser = False,
            out_filename = 'nvisii_output_img'
        )
        self.texture_dir = texture_dir
        # self.hdr_dir = hdr_dir
        self.hdr_fn = glob.glob(os.path.join(hdr_dir, '*.hdr'))

        self.init_variables()

        nvisii.initialize(headless=True, lazy_updates=True)
        if not self.option.use_denoiser is True: 
            nvisii.enable_denoiser()

    def init_variables(self):
        self.sun = None
        self.table_entity = None
        self.grasped_obj = None
        self.obj_entity_list = []


    def set_scene(
            self, 
            nvren_info: sg.SceneCls.SceneData.NVRenInfo,
            table_params: npt.NDArray,
            robot_params: npt.NDArray,
            add_distractor: bool
    ):
        # scene_objects = cxutil.CvxObjects().init_obj_info(
        #     (obj_info.obj_posquats, obj_info.obj_cvx_verts_padded, obj_info.obj_cvx_faces_padded))
        scene_objects = cxutil.CvxObjects().init_obj_info(
            (np.concatenate([np.zeros(3),np.array([0,0,0,1.])],-1), nvren_info.canonical_verts, nvren_info.canonical_faces))

        max_obj_no = nvren_info.obj_posquats.shape[-2]

        # Change the dome light intensity
        nvisii.set_dome_light_intensity(1.0)

        hdr_name = np.random.choice(self.hdr_fn)
        self.dome_texture = nvisii.texture.get(hdr_name)
        if self.dome_texture is None:
            self.dome_texture = nvisii.texture.create_from_file(hdr_name, hdr_name)

        # Use "enable_cdf" for dome light textures that contain 
        # bright objects that cast shadows (like the sun). Note
        # that this has a significant impact on rendering performance,
        # and is disabled by default.
        nvisii.set_dome_light_texture(self.dome_texture, enable_cdf = True)
        nvisii.set_dome_light_rotation(nvisii.angleAxis(nvisii.pi() * .1, (0,0,1)))

        # Lets add a sun light
        if self.sun is None:
            self.sun = nvisii.entity.create(
                name = "sun",
                mesh = nvisii.mesh.create_sphere("sphere"),
                transform = nvisii.transform.create("sun"),
                light = nvisii.light.create("sun")
            )
        roll = np.random.uniform(-np.pi/4, np.pi/4)
        yaw = np.random.uniform(-np.pi, np.pi)
        sun_dir = (sciR.from_euler('x',roll) * sciR.from_euler('z',yaw)).as_matrix()[:,2]
        sun_mag = np.random.uniform(9, 16)
        self.sun.get_transform().set_position(sun_mag*sun_dir)
        self.sun.get_light().set_temperature(5780)
        self.sun.get_light().set_intensity(2000)

        texture_file_list = glob.glob(f'{self.texture_dir}/*/*/*.jpg')
        obj_texture_file_list = glob.glob(f'{self.texture_dir}/object/*.jpg')

        # Set boxes
        if nvisii.entity.get('box') is not None:
            nvisii.transform.remove('box')
            nvisii.material.remove('box')
            nvisii.mesh.remove('box')
            nvisii.entity.remove('box')
        
        if isinstance(table_params, dict):
            # shelf params

            mesh_ = nvisii.mesh.get(table_params['shelf_name'])
            if mesh_ is None:
                mesh_ = nvisii.mesh.create_from_file(name=table_params['shelf_name'], path=table_params['shelf_name'])

            self.table_entity= nvisii.entity.create(
                name = 'box',
                transform = nvisii.transform.create('box'),
                material = nvisii.material.create('box'),
                mesh=mesh_
            )

            self.table_entity.get_transform().set_scale(table_params['shelf_scale'])
            self.table_entity.get_transform().set_position(table_params['shelf_position'])
            self.table_entity.get_transform().set_rotation(table_params['shelf_rotation'])

            if np.random.uniform() < 0.4:
                box_texture_fn = np.random.choice(texture_file_list)
                tex_ = nvisii.texture.get(box_texture_fn)
                if tex_ is None:
                    tex_ = nvisii.texture.create_from_file(name=box_texture_fn, path=box_texture_fn)
                self.table_entity.get_material().set_base_color_texture(tex_)
            else:
                self.table_entity.get_material().set_base_color(
                    nvisii.vec3(*np.random.uniform(size=(3,))))
                self.table_entity.get_material().set_roughness(np.random.uniform(0, 0.5))
                self.table_entity.get_material().set_specular(np.random.uniform(0, 0.2))
                self.table_entity.get_material().set_metallic(0)
        else:
            self.table_entity = nvisii.entity.create(
                name="box",
                mesh = nvisii.mesh.create_box("box"),
                transform = nvisii.transform.create("box"),
                material = nvisii.material.create("box")
            )
            # Lets set the box1 up
            floor_height = table_params[0]
            self.table_entity.get_transform().set_scale((*table_params[1:], 1))
            self.table_entity.get_transform().set_position(
                nvisii.vec3(0,0,-1.0+floor_height)
            )
            if np.random.uniform() < 0.4:
                box_texture_fn = np.random.choice(texture_file_list)
                tex_ = nvisii.texture.get(box_texture_fn)
                if tex_ is None:
                    tex_ = nvisii.texture.create_from_file(name=box_texture_fn, path=box_texture_fn)
                self.table_entity.get_material().set_base_color_texture(tex_)
            else:
                self.table_entity.get_material().set_base_color(
                    nvisii.vec3(*np.random.uniform(size=(3,))))
                self.table_entity.get_material().set_roughness(np.random.uniform(0, 0.5))
                self.table_entity.get_material().set_specular(np.random.uniform(0, 0.2))
                self.table_entity.get_material().set_metallic(0)

        if nvisii.entity.get('grasped_obj') is not None:
            nvisii.entity.remove('grasped_obj')
            nvisii.transform.remove('grasped_obj')
            nvisii.material.remove('grasped_obj')

        if robot_params is not None and np.random.uniform() < 0.5:
            # stick
            # if nvisii.entity.get('grasped_obj') is None:
            self.grasped_obj = nvisii.entity.create(
                name="grasped_obj",
                # mesh = nvisii.mesh.create_capped_cylinder("grasped_obj", radius=robot_params[1], size=robot_params[0]/2),
                transform = nvisii.transform.create("grasped_obj"),
                material = nvisii.material.create("grasped_obj")
            )
            # if nvisii.mesh.get('grasped_obj') is not None:
            nvisii.mesh.remove('grasped_obj')
            mesh = nvisii.mesh.create_capped_cylinder("grasped_obj", radius=robot_params[1], size=robot_params[0]/2)
            self.grasped_obj.set_mesh(mesh)
            # lets set the box1 up

            self.grasped_obj.get_transform().set_position(
                nvisii.vec3(robot_params[2],robot_params[3],robot_params[4])
                )
            self.grasped_obj.get_transform().set_rotation(
                nvisii.quat(robot_params[8], robot_params[5], robot_params[6], robot_params[7])
                )
            self.grasped_obj.get_material().set_base_color(
                nvisii.vec3(*np.random.uniform(size=(3,))))
            self.grasped_obj.get_material().set_roughness(np.random.uniform(0, 0.5))
            self.grasped_obj.get_material().set_specular(np.random.uniform(0, 0.2))
            self.grasped_obj.get_material().set_metallic(0)
        # else:
        #     if nvisii.entity.get('grasped_obj') is None:
        #         nvisii.entity.remove('grasped_obj')

        # Enroll obj
        if len(self.obj_entity_list) != 0:
            for oe in self.obj_entity_list:
                if oe is not None:
                    nvisii.transform.remove(oe.get_name())
                    nvisii.material.remove(oe.get_name())
                    nvisii.entity_remove(oe.get_name())

        self.obj_entity_list = []
        for j in range(max_obj_no):
            obj_et = self.set_object(*jax.tree_map(lambda x: x[j], (scene_objects, nvren_info)), obj_id=j, colors=None, obj_texture_file_list=obj_texture_file_list)
            self.obj_entity_list.append(obj_et)

        # Add distractor
        if add_distractor:
            for j in range(max_obj_no):
                distractor_obj: cxutil.CvxObjects = jax.tree_map(lambda x: x[j], scene_objects)
                distractor_nv = jax.tree_map(lambda x: x[j], nvren_info)
                rand_pos = np.random.normal(size=(2,))
                # rand_r = np.random.uniform(1.2, 1.6)
                rand_r = np.random.uniform(2.0, 2.5)
                rand_pos = rand_pos/np.linalg.norm(rand_pos) * rand_r
                # distractor_obj = distractor_obj.replace(pos=distractor_obj.pos.at[:2].set(rand_pos))
                rand_quat = np.random.normal(size=(4,))
                rand_quat = rand_quat/np.linalg.norm(rand_quat)
                distractor_nv = distractor_nv.replace(obj_posquats=np.concatenate([rand_pos, distractor_nv.obj_posquats[2:3], rand_quat],-1))
                dist_obj_et = self.set_object(distractor_obj, distractor_nv, 2*max_obj_no+j, colors=None, obj_texture_file_list=obj_texture_file_list)
                self.obj_entity_list.append(dist_obj_et)

    def set_object(
            self, 
            scene_objects: cxutil.CvxObjects, 
            nvren_info:sg.SceneCls.SceneData.NVRenInfo,
            obj_id, 
            colors, 
            obj_texture_file_list
    ):
        vtx = scene_objects.vtx_tf[np.where(scene_objects.vtx_valid_mask[...,0])]
        indices = scene_objects.fc[np.where(scene_objects.fc_valid_mask[...,0])]
        if len(indices) == 0:
            return None
        
        # uv = rutil.texcoord_parameterization(vtx)
        # if nvren_info.mesh_name not in self.mesh_dict:
        mesh_ = nvisii.mesh.get(nvren_info.mesh_name)
        if mesh_ is None:
            # mesh_ = nvisii.mesh.create_from_data(name=nvren_info.mesh_name, positions=vtx.reshape((-1,)), indices=indices.reshape((-1,)), texcoords=uv.reshape((-1,)))
            mesh_ = nvisii.mesh.create_from_file(name=nvren_info.mesh_name, path=nvren_info.mesh_name)

        name = f"mesh_{obj_id}"
        if nvisii.entity.get(name) is not None:
            nvisii.transform.remove(name)
            nvisii.material.remove(name)
            nvisii.entity.remove(name)
        obj= nvisii.entity.create(
            name = name,
            transform = nvisii.transform.create(name),
            material = nvisii.material.create(name),
            mesh=mesh_
        )
        # obj.set_mesh(mesh_)

        obj.get_transform().set_scale(nvren_info.scales)
        obj.get_transform().set_position(nvren_info.obj_posquats[...,:3])
        obj.get_transform().set_rotation(nvren_info.obj_posquats[...,3:])

        r = random.randint(0,3) # plastic (0-1) / metalic / glass

        # Material setting
        obj_mat = obj.get_material()
        if colors is not None: 
            rgb = colors
            obj_mat.set_base_color(rgb)
        else:
            if np.random.uniform() < 0.8 and r<=1:
                obj_texture_fn = np.random.choice(obj_texture_file_list)
                tex_ = nvisii.texture.get(obj_texture_fn)
                if tex_ is None:
                    tex_ = nvisii.texture.create_from_file(name=obj_texture_fn, path=obj_texture_fn)
                obj_mat.set_base_color_texture(tex_)
            else:
                rgb = [random.uniform(0,1),random.uniform(0,1),random.uniform(0,1)]
                obj_mat.set_base_color(rgb)
                if r <= 1:  
                    # Plastic / mat
                    obj_mat.set_metallic(0)  # should 0 or 1      
                    obj_mat.set_transmission(0)  # should 0 or 1      
                    obj_mat.set_roughness(random.uniform(0,1)) # default is 1  
                if r == 2:  
                    # metallic
                    obj_mat.set_metallic(random.uniform(0.9,1))  # should 0 or 1      
                    obj_mat.set_transmission(0)  # should 0 or 1      
                if r == 3:  
                    # glass
                    obj_mat.set_metallic(0)  # should 0 or 1      
                    obj_mat.set_transmission(random.uniform(0.9,1))  # should 0 or 1      
                    if np.random.uniform() < 0.8:
                        obj_mat.set_base_color(np.random.uniform(0.8, 1.0, size=(3,)))
                if r > 1: # for metallic and glass
                    r2 = random.randint(0,1)
                    if r2 == 1: 
                        obj_mat.set_roughness(random.uniform(0,.1)) # default is 1  
                    else:
                        obj_mat.set_roughness(random.uniform(0.9,1)) # default is 1  
        return obj

    def take_pictures(
            self, 
            cam_info: sg.SceneCls.SceneData.CamInfo, 
            cidx: int
    ) -> Image.Image:
        """Take picture of the current scene.

        Args:
            cam_info (sg.SceneCls.SceneData.CamInfo): Camera info
            cidx (int): Unique camera id. Any arbitrary integer is okay.

        Returns:
            PIL.Image.Image: PIL image file
        """

        camera = nvisii.entity.get('camera')
        if camera is None:
            # Create a camera
            camera = nvisii.entity.create(
                name = "camera",
                transform = nvisii.transform.create("camera"),
                # camera = nvisii.camera.create_from_fov("camera", *cutil.intrinsic_to_fov(cam_info.cam_intrinsics[cidx]))
            )
        if nvisii.camera.get("camera") is not None:
            nvisii.camera.remove("camera")
        camera.set_camera(nvisii.camera.create_from_fov("camera", *cutil.intrinsic_to_fov(cam_info.cam_intrinsics[cidx])))
        cpos = cam_info.cam_posquats[cidx,:3]
        cRm = tutil.q2R(cam_info.cam_posquats[cidx,3:])
        camera.get_transform().look_at(
            at = cpos - cRm[:,2],
            up = cRm[:,1],
            eye = cpos,
        )
        nvisii.set_camera_entity(camera)

        render_bump = np.maximum(200/self.option.height, 1)
        render_pixel_size_ij = [int(self.option.height*render_bump), int(self.option.width*render_bump)]
        img = nvisii.render(
            height=render_pixel_size_ij[0], 
            width=render_pixel_size_ij[1], 
            samples_per_pixel=int(self.option.spp),
        )
        img = np.sqrt(np.array(img).reshape((*render_pixel_size_ij, 4))[...,::-1,:,:3])
        img = (img*255).clip(0,255).astype(np.uint8)
        img = Image.fromarray(img).resize((int(self.option.width), int(self.option.height)), Image.BICUBIC)

        return img

    def get_rgb_for_datapoint(
            self, 
            datapoint: sg.SceneCls.SceneData, 
            add_distractor: bool
    ) -> npt.NDArray:
        """TODO

        Args:
            datapoint (sg.SceneCls.SceneData): _description_
            add_distractor (bool): _description_

        Returns:
            npt.NDArray: [#cam, H, W, 3]
        """
        # nvisii.clear_all()
        self.set_scene(
            datapoint.nvren_info,
            datapoint.table_params,
            datapoint.robot_params,
            add_distractor)
        
        num_cams = datapoint.cam_info.cam_posquats.shape[0]
        img_list = []
        for cidx in range(num_cams):
            img = self.take_pictures(datapoint.cam_info, cidx)
            img_list.append(img)
        
        return np.stack(img_list, 0)
