from typing import NamedTuple, Tuple
import jax
import jax.numpy as jnp
from functools import partial
from IPython.display import display, Image, HTML
import numpy as np
# import PIL
from PIL import Image as pImage
import io
import time
import subprocess
import base64
import einops
import dm_pix as pix

import util.camera_util as cutil
import util.transform_util as tutil
import util.cvx_util as cxutil

CvxObjects = cxutil.CvxObjects

def np2pil(a):
    if a.dtype in [np.float32, np.float64]:
        a = np.uint8(np.clip(a, 0, 1) * 255)
    return pImage.fromarray(a)


def imwrite(f, a, fmt=None):
    a = np.asarray(a)
    if isinstance(f, str):
        fmt = f.rsplit(".", 1)[-1].lower()
        if fmt == "jpg":
            fmt = "jpeg"
        f = open(f, "wb")
    np2pil(a).save(f, fmt, quality=95)


def imencode(a, fmt="jpeg"):
    a = np.asarray(a)
    if len(a.shape) == 3 and a.shape[-1] == 4:
        fmt = "png"
    f = io.BytesIO()
    imwrite(f, a, fmt)
    return f.getvalue()


def imshow(a, fmt="jpeg", display=display):
    return display(Image(data=imencode(a, fmt)))


class VideoWriter:
    def __init__(self, filename="_autoplay.mp4", fps=30.0):
        self.ffmpeg = None
        self.filename = filename
        self.fps = fps
        self.view = display(display_id=True)
        self.last_preview_time = 0.0

    def add(self, img):
        img = np.asarray(img)
        h, w = img.shape[:2]
        if self.ffmpeg is None:
            self.ffmpeg = self._open(w, h)
        if img.dtype in [np.float32, np.float64]:
            img = np.uint8(img.clip(0, 1) * 255)
        if len(img.shape) == 2:
            img = np.repeat(img[..., None], 3, -1)
        self.ffmpeg.stdin.write(img.tobytes())
        t = time.time()
        if self.view and t - self.last_preview_time > 1:
            self.last_preview_time = t
            imshow(img, display=self.view.update)

    def __call__(self, img):
        return self.add(img)

    def _open(self, w, h):
        cmd = f"""ffmpeg -y -f rawvideo -vcodec rawvideo -s {w}x{h}
      -pix_fmt rgb24 -r {self.fps} -i - -pix_fmt yuv420p 
      -c:v libx264 -crf 20 {self.filename}""".split()
        return subprocess.Popen(cmd, stdin=subprocess.PIPE)

    def close(self):
        if self.ffmpeg:
            self.ffmpeg.stdin.close()
            self.ffmpeg.wait()
            self.ffmpeg = None

    def __enter__(self):
        return self

    def __exit__(self, *kw):
        self.close()
        if self.filename == "_autoplay.mp4":
            self.show()

    def show(self):
        self.close()
        if not self.view:
            return
        b64 = base64.b64encode(open(self.filename, "rb").read()).decode("utf8")
        s = f"""<video controls loop>
 <source src="data:video/mp4;base64,{b64}" type="video/mp4">
 Your browser does not support the video tag.</video>"""
        self.view.update(HTML(s))


def animate(f, duration_sec, fps=20):
    with VideoWriter(fps=fps) as vid:
        for t in jnp.linspace(0, 1, int(duration_sec * fps)):
            vid(f(t))

def sdf_ratio_calculator(models, jkey, vtx_list, fcs_list, ns=10, visualize=False):

    oidx = jax.random.randint(jkey, (ns,), 0, vtx_list.shape[0])
    _, jkey = jax.random.split(jkey)
    scale = jax.random.uniform(jkey, [ns, 1, 1, 3], minval=0.6, maxval=1.7)
    _, jkey = jax.random.split(jkey)
    cvx = vtx_list[oidx]*scale
    fc = fcs_list[oidx]

    gt_cvx_object = cxutil.CvxObjects().init_vtx(cvx, fc)
    latent_object = gt_cvx_object.set_z_with_models(jkey, models)

    surface_pnts, surface_nmls, _ = cxutil.sampling_from_surface_convex_dec(jkey, gt_cvx_object.vtx_tf, gt_cvx_object.fc, ns=4, normal_out=True)
    _, jkey = jax.random.split(jkey)
    distance_dist = jax.random.uniform(jkey, shape=surface_pnts[...,0].shape, minval=0.01, maxval=0.08) # (... NS 1)
    distance_dist = distance_dist
    dist_pnts = surface_pnts + surface_nmls * distance_dist[...,None]
    occ_values = models.apply('occ_predictor', latent_object, dist_pnts)
    ratio = jnp.maximum(-occ_values, 0) / distance_dist * 2.5

    if visualize:
        import open3d as o3d
        eidx = 0
        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(surface_pnts[eidx]))
        pcd.normals = o3d.utility.Vector3dVector(surface_nmls[eidx])
        pcd.colors = o3d.utility.Vector3dVector(einops.repeat(np.array([1,0,0]), 'i -> r i', r=ns))

        pcd2 = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(dist_pnts[eidx]))
        pcd2.colors = o3d.utility.Vector3dVector(einops.repeat(np.array([0,0,1]), 'i -> r i', r=ns))

        o3d.visualization.draw_geometries([pcd, pcd2], point_show_normal=True)
        
    return jnp.mean(ratio)


def scene_sdf(objects, p, models, up_idx=2, sdf_ratio=2200, floor_offset=-0.8):
    '''
    positive: outside
    negative: inside
    '''
    occ_values = models.apply('occ_predictor', objects, p[...,None,:])
    dists = jnp.squeeze(-occ_values, (-1,))/sdf_ratio
    id = jnp.argmin(dists)
    obj_dist = dists[id]
    if floor_offset is not None:
        floor_dist = p[up_idx] - floor_offset  # floor is at y==-0.8
        id = jnp.where(obj_dist<floor_dist, id, -1)
        return jnp.minimum(obj_dist, floor_dist), id
    else:
        return obj_dist, id

def _sdf_raycast(sdf, objects, p0, dir, step_n=50, floor_offset=-0.8):
    '''
    sdf : function of pnts -> scalar
    '''
    def f(_, p):
        v, _ = sdf(objects, p, floor_offset=floor_offset)
        return p + v * dir
    return jax.lax.fori_loop(0, step_n, f, p0)

def sdf_raycast(sdf, objects, p0, dir, world_up_idx=2, threshold=0.01, step_n=200, floor_offset=-0.8, normal=False, models=None):

    if models is not None:
        k = np.minimum(2, objects.outer_shape[0])
        p0, hit_id_ = rdf_raycast(objects, p0, dir, models, k=k, world_up_idx=world_up_idx, 
                                  normal=False, depth=False, id_depth=False, floor_offset=floor_offset, soft=False)
        if k==1:
            hit_id_ = hit_id_[None]
            # objects = jax.tree_map(lambda x: x[hit_id_][None], objects)
        # else:
        objects = jax.tree_map(lambda x: x[hit_id_], objects)
        # p0 = p0 - 0.03*dir
        p0 = p0 - 0.05*dir
        # hit_id_ = hit_id_[None]
    hit_p = _sdf_raycast(sdf, objects, p0, dir, step_n=step_n, floor_offset=floor_offset)
    v, hit_id = sdf(objects, hit_p, floor_offset=floor_offset)
    if models is not None:
        hit_id = jnp.where(hit_id>=0, hit_id_[hit_id], hit_id)
    floor_hit_p = p0 + dir*(floor_offset - p0[world_up_idx]/dir[world_up_idx])
    floor_hit_p = jnp.where(dir[world_up_idx]>-1e-4, p0 + dir*1e5, floor_hit_p)
    hit_p = jnp.where(jnp.logical_or(v<threshold, hit_id==-1), hit_p, floor_hit_p)
    hit_id = jnp.where(jnp.logical_or(v<threshold , hit_id==-1), hit_id, -1)
    if normal:
        raw_normal, _ = jax.grad(partial(sdf, objects, floor_offset=floor_offset), has_aux=True)(hit_p)
        return hit_p, hit_id, raw_normal
    else:
        return hit_p, hit_id

def sdf_cast_shadow(sdf, objects, light_dir, p0, world_up_idx=2, floor_offset=-0.8, step_n=50, hardness=15.0, models=None, out_depth=False):

    if models is not None:
        # use this just reduce object number
        k = np.minimum(2, objects.outer_shape[0])
        _, hit_id_ = rdf_raycast(objects, p0, light_dir, models, k=k, world_up_idx=world_up_idx, 
                                  normal=False, depth=False, id_depth=False, floor_offset=floor_offset, soft=False)
        if k==1:
            hit_id_ = hit_id_[None]
        objects = jax.tree_map(lambda x: x[hit_id_], objects)

    def f(_, carry):
        t, shadow = carry
        h, hit_id_ = sdf(objects, p0 + light_dir * t, floor_offset=floor_offset)
        return t + h, jnp.clip(hardness * h / t, 0.0, shadow)
        # return t + h, jnp.where(hit_id_==-1, 1.0, jnp.clip(hardness * h / t, 0.0, shadow))
    hit_depth, shadow = jax.lax.fori_loop(0, step_n, f, (1e-2, 1.0))
    if out_depth:
        return hit_depth, shadow
    else:
        return shadow

def cvx_cast_shadow(objects, light_dir, p0, hit_id=None, world_up_idx=2, floor_offset=-0.8):
    '''
    1: no shadow
    0: shadow
    '''
    hit_p, id = cvx_raycast(objects, p0, light_dir, world_up_idx=world_up_idx, floor_offset=floor_offset)
    if hit_id is not None:
        shadow = (jnp.logical_or(id==-2, id==hit_id)).astype(jnp.float32)
    else:
        shadow = (id==-2).astype(jnp.float32)
    return shadow

def rdf_cast_shadow(objects, light_dir, p0, hit_id, models, world_up_idx=2, floor_offset=-0.8):
    '''
    1: no shadow
    0: shadow
    '''
    # hit_p, id = rdf_raycast(objects, p0, light_dir, rdf=rdf, world_up_idx=world_up_idx, floor_offset=floor_offset)
    # shadow = (id==-2).astype(jnp.float32)
    # shoot ray start from hitting points toward light source.
    id, depth = rdf_raycast(objects, p0, -light_dir, models=models, world_up_idx=world_up_idx, floor_offset=floor_offset, id_depth=True)
    # shadow = (depth > -2e-3).astype(jnp.float32)
    self_shadow = (depth > -5e-2) # if depth have negative value, it means p0 is inside object -> self shadow
    obj_interaction_shadow = (hit_id == id) # if hit id different -> occluded by other objects
    floor_shadow = jnp.logical_or(hit_id>=0, id<0) # add floor shadow
    shadow = jnp.logical_and(self_shadow, obj_interaction_shadow) # add obj shadow
    shadow = jnp.logical_and(shadow, floor_shadow)

    # import matplotlib.pyplot as plt
    # plt.figure()
    # plt.imshow(floor_shadow.val.reshape(240,424,1))
    # plt.show()

    # plt.figure()
    # plt.imshow(self_shadow.val.reshape(240,424,1))
    # plt.show()  

    # plt.figure()
    # plt.imshow(obj_interaction_shadow.val.reshape(240,424,1))
    # plt.show()

    # plt.figure()
    # plt.imshow(shadow.val.reshape(240,424,1))
    # plt.show()


    # id, depth = rdf_raycast(objects, p0, light_dir, models=models, world_up_idx=world_up_idx, floor_offset=floor_offset, id_depth=True)
    # ids, depthes = jax.vmap(partial(rdf_raycast, p0, light_dir, models=models, world_up_idx=world_up_idx, floor_offset=floor_offset, id_depth=True))(objects)

    return shadow


def mixed_raycast(sdf, objects:CvxObjects, p0, dir, render_type_indices, world_up_idx=2, threshold=0.01, step_n=200, floor_offset=-0.8, normal=False, models=None):
    '''
    objects: outer shape : (B, ...) -> B: object number
    '''
    
    objects_cvx = jax.tree_map(lambda x: x[np.where(render_type_indices==1)], objects)
    objects_latent = jax.tree_map(lambda x: x[np.where(render_type_indices==0)], objects)

    cvx_ids = jnp.arange(objects.outer_shape[0])[np.where(render_type_indices==1)]
    latent_ids = jnp.arange(objects.outer_shape[0])[np.where(render_type_indices==0)]


    sdfres = sdf_raycast(sdf, objects_latent, p0, dir, world_up_idx=world_up_idx, threshold=threshold, step_n=step_n, 
                floor_offset=floor_offset, normal=normal, models=models)
    latent_ids = jnp.where(sdfres[1]<0, sdfres[1], latent_ids[sdfres[1]])
    sdfres = (sdfres[0], latent_ids, *sdfres[2:])
    cvxres = cvx_raycast(objects_cvx, p0, dir, world_up_idx=world_up_idx, normal=normal, depth=False, floor_offset=floor_offset)
    cvx_ids = jnp.where(cvxres[1]<0, cvxres[1], cvx_ids[cvxres[1]])
    cvxres = (cvxres[0], cvx_ids, *cvxres[2:])

    decide_basis = jnp.sum((sdfres[0]-p0)**2) > jnp.sum((cvxres[0]-p0)**2)
    return jax.tree_map(lambda x,y: jnp.where(decide_basis, x, y), cvxres, sdfres)
    


def rdf_cast_shadow_alpha(objects, light_dir, points, depths, models, world_up_idx=2, floor_offset=-0.8):
    '''
    return 1: pixel in shadow - 0
    '''
    # hit_p, id = rdf_raycast(objects, p0, light_dir, rdf=rdf, world_up_idx=world_up_idx, floor_offset=floor_offset)
    # shadow = (id==-2).astype(jnp.float32)
    # p0 = points[objects.alpha_one_mask] # TODO: ground truth object that has alpha=1
    # points, depths = rdf_raycast_alpha(objects, p0, -light_dir, models=models, world_up_idx=world_up_idx, floor_offset=floor_offset, id_depth=True)
    # depth = rdf_raycast(objects, p0, -light_dir, rdf=rdf, world_up_idx=world_up_idx, floor_offset=floor_offset, depth=True)
    # shadow = (id==-1).astype(jnp.float32)
    # shadow = (depth > -2e-3).astype(jnp.float32)
    # shadow = (hit_id == id).astype(jnp.float32)
    shadow = jnp.zeros_like(depths)
    return shadow


def shade_f(surf_color, shadow, raw_normal, ray_dir, light_dir):
    '''
    shadow : values between 0 - 1 - 1 means entirely blocked
    '''
    ambient = tutil.safe_norm(raw_normal, axis=-1, eps=1e-10)
    normal = raw_normal / ambient
    diffuse = normal.dot(light_dir).clip(0.0) * shadow
    half = tutil.normalize(light_dir - ray_dir)
    spec = 0.3 * shadow * half.dot(normal).clip(0.0) ** 200.0
    light = 0.7 * diffuse + 0.2 * ambient
    return surf_color * light + spec

def shade_f_alpha(color_pack, shadow, depths, normals, ray_dir, light_dir):
    '''
    shadow : values between 0 - 1 - 1 means entirely blocked
    '''

    # ray dir = (3)
    # surf_color, raw_normal = (NO (object number), 4)
    # output = (3)
    # light_dir = (3)
    # shadow = (NO)
    # apply shadow only when alpha=1 

    ambient = tutil.safe_norm(normals, axis=-1, eps=1e-10, keepdims=True) # (NO, ...)
    normal = normals / ambient
    diffuse = (normal*light_dir).sum(axis=-1, keepdims=True).clip(0.0) # (NO, ...)
    half = tutil.normalize(light_dir - ray_dir)
    spec = 0.3 * (normal*half).sum(axis=-1, keepdims=True).clip(0.0) ** 200.0
    obj_colors = 0.7 * diffuse + 0.2 * ambient + spec

    depths = jnp.squeeze(depths)
    depth_order_idx = jnp.argsort(depths)
    validity = depths < 20.
    validity_2 = depths > 0.
    alphas = color_pack[:-1, 3]
    alphas = jnp.where(validity, alphas, 0.)
    alphas = jnp.where(validity_2, alphas, 0.)
    remaining_light = 1.0
    final_color = jnp.zeros(3)

    for i in range(color_pack.shape[0]-1):
        obj_idx = depth_order_idx[i]
        valid = validity[obj_idx]
        # if depth > 20.: break
        alpha = alphas[obj_idx]
        final_color += remaining_light * alpha * obj_colors[obj_idx] * color_pack[obj_idx, :3]
        remaining_light = remaining_light * (1. - alpha)
        # if alpha == 1.: break
    
    final_color += remaining_light * color_pack[-1, :3]

    return final_color

def cvx_render_scene(
    objects:cxutil.CvxObjects,
    models=None,
    sdf=None,
    intrinsic=None,
    render_type_indices=None,
    pixel_size=(400, 640),
    target_pos=jnp.array([0.0, 0.0, 0.0]),
    camera_pos=jnp.array([4.0, 4.0, 3.0])/5,
    camera_quat=None,
    camera_up=jnp.array([0,0,1.]),
    light_dir = jnp.array([0.733, 0.133, 0.666]),
    sky_color=jnp.array([0.3, 0.4, 0.7]),
    world_up_idx=2,
    floor_offset=-0.8,
    seg_out=False,
    debug=False,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    '''
    objects - outer_shape (NO,)
    all other values have no batch dimension

    '''
    if len(objects.outer_shape) == 2:
        objects:cxutil.CvxObjects = jax.tree_map(lambda x: jnp.squeeze(x, axis=0), objects)
    if len(objects.outer_shape) == 0:
        objects:cxutil.CvxObjects = jax.tree_map(lambda x: x[None], objects)
    light_dir = tutil.normalize(light_dir)

    if objects.color is None:
        objects = objects.random_color(jax.random.PRNGKey(2))

    def render_ray(ray_dir):
        if models is not None:
            if sdf is not None:
                # use rdf broad-check to improve efficiency
                # hit_pos, hit_id, raw_normal = sdf_raycast(sdf, objects, camera_pos, ray_dir, threshold=0.020, step_n=20, world_up_idx=world_up_idx, floor_offset=floor_offset, normal=True, models=models)
                hit_pos, hit_id, raw_normal = sdf_raycast(sdf, objects, camera_pos, ray_dir, threshold=0.020, step_n=100, world_up_idx=world_up_idx, floor_offset=floor_offset, normal=True, models=models)
                shadow = sdf_cast_shadow(sdf, objects, light_dir, hit_pos, world_up_idx, floor_offset=floor_offset, models=models)
            else:
                # render with rdf
                hit_pos, hit_id, raw_normal = rdf_raycast(objects, camera_pos, ray_dir, models=models, world_up_idx=world_up_idx, floor_offset=floor_offset, normal=True)
                shadow = rdf_cast_shadow(objects, light_dir, hit_pos, hit_id, models, world_up_idx, floor_offset=floor_offset)
        elif sdf is not None:
            # render with ray marching
            hit_pos, hit_id, raw_normal = sdf_raycast(sdf, objects, camera_pos, ray_dir, world_up_idx=world_up_idx, floor_offset=floor_offset, normal=True)
            shadow = sdf_cast_shadow(sdf, objects, light_dir, hit_pos, world_up_idx, floor_offset=floor_offset)
        else:
            # render with cvx raycast
            hit_pos, hit_id, raw_normal = cvx_raycast(objects, camera_pos, ray_dir, world_up_idx=world_up_idx, floor_offset=floor_offset, normal=True)
            shadow = cvx_cast_shadow(objects, light_dir, hit_pos, world_up_idx, floor_offset=floor_offset)

        # floor color
        x, y, z = jnp.tanh(jnp.sin(hit_pos*2*jnp.pi)*20.0)
        floor_color = (0.5+(x*y)*0.1)*jnp.ones(3)
        color_pack = jnp.stack([sky_color, floor_color], 0)
        color_pack = jnp.concatenate([objects.color[...,:3], color_pack], axis=0)

        surf_color = color_pack[hit_id]
        color = shade_f(surf_color, shadow, raw_normal, ray_dir, light_dir)
        escape = jnp.tanh(jnp.abs(hit_pos).max() - 10.0) * 0.5 + 0.5
        color = color + (sky_color - color) * escape
        if debug:
            return ray_dir, hit_pos, hit_id, raw_normal, shadow, (color** (1.0 / 2.2)).clip(0,1.0)
        else:
            return color ** (1.0 / 2.2), hit_id  # gamma correction

    if intrinsic is None:
        intrinsic = cutil.default_intrinsic(pixel_size)
    _,_,ray_dir = cutil.pixel_ray(pixel_size, cam_pos=camera_pos, 
                cam_quat=camera_quat if camera_quat is not None else tutil.line2q(camera_pos-target_pos, camera_up), 
                intrinsic=intrinsic, near=0.001, far=3.0)
    ray_dir = ray_dir.reshape((-1,3))
    if debug:
        debug_results = jax.vmap(render_ray)(ray_dir)
        return jax.tree_map(lambda x: x.reshape(tuple(pixel_size) + x.shape[1:]), debug_results)
    color, hit_id = jax.vmap(render_ray)(ray_dir)
    rgb = color.reshape(*pixel_size, 3)
    rgb = jnp.where(jnp.isnan(rgb), 0, rgb)
    if seg_out:
        return rgb.clip(0,1.0), hit_id.reshape(*pixel_size)
    else:
        return rgb.clip(0,1.0)


def cvx_render_scene_depth(
    objects:cxutil.CvxObjects,
    models=None,
    sdf=None,
    intrinsic=None,
    pixel_size=(400, 640),
    target_pos=jnp.array([0.0, 0.0, 0.0]),
    camera_pos=jnp.array([4.0, 4.0, 3.0])/5,
    camera_quat=None,
    camera_up=jnp.array([0,0,1.]),
    world_up_idx=2,
    floor_offset=-0.8,
):
    if len(objects.outer_shape) == 2:
        objects:cxutil.CvxObjects = jax.tree_map(lambda x: jnp.squeeze(x, axis=0), objects)

    def render_ray(ray_dir):
        if models is not None:
            hit_depth = rdf_raycast(objects, camera_pos, ray_dir, models=models, world_up_idx=world_up_idx, floor_offset=floor_offset, depth=True, soft=True)
        elif sdf is not None:
            hit_depth = sdf_raycast(sdf, objects, camera_pos, ray_dir, world_up_idx=world_up_idx, floor_offset=floor_offset, depth=True, soft=True)
        else:
            hit_depth = cvx_raycast(objects, camera_pos, ray_dir, world_up_idx=world_up_idx, floor_offset=floor_offset, depth=True, soft=True)

        return hit_depth

    if intrinsic is None:
        intrinsic = cutil.default_intrinsic(pixel_size)
    _,_,ray_dir = cutil.pixel_ray(pixel_size, cam_pos=camera_pos, 
                cam_quat=camera_quat if camera_quat is not None else tutil.line2q(camera_pos-target_pos, camera_up), 
                intrinsic=intrinsic, near=0.001, far=3.0)
    ray_dir = ray_dir.reshape((-1,3))
    depth = jax.vmap(render_ray)(ray_dir)
    depth = depth.reshape(*pixel_size, 1)
    depth = jnp.where(jnp.isnan(depth), 0, depth)
    return depth


def cvx_render_scene_alpha(
    objects:cxutil.CvxObjects,
    models=None,
    sdf=None,
    intrinsic=None,
    pixel_size=(400, 640),
    target_pos=jnp.array([0.0, 0.0, 0.0]),
    camera_pos=jnp.array([4.0, 4.0, 3.0])/5,
    camera_quat=None,
    camera_up=jnp.array([0,0,1.]),
    light_dir = jnp.array([0.733, 0.133, 0.666]),
    sky_color=jnp.array([0.3, 0.4, 0.7, 1.0]),
    floor_color=jnp.array([0.481,0.548,0.707, 1.]),
    world_up_idx=2,
    floor_offset=-0.8,
):
    light_dir = tutil.normalize(light_dir)

    color_pack = jnp.stack([floor_color, sky_color], 0)
    # obj_rgb = objects.color # (5, 3)
    # obj_num = obj_rgb.shape[0]
    # obj_a = float(3./obj_num) * jnp.ones(obj_num)[...,None] # (5, 1)
    # obj_rgba = jnp.concatenate([obj_rgb, obj_a], axis=-1)
    obj_rgba = objects.color
    color_pack = jnp.concatenate([obj_rgba, color_pack], axis=0)

    def render_ray_alpha(ray_dir):
        if models is not None:
            points, depths, normals = rdf_raycast_alpha(objects, camera_pos, ray_dir, models=models, world_up_idx=world_up_idx, floor_offset=floor_offset, normal=True)
            shadow = rdf_cast_shadow_alpha(objects, light_dir, points, depths, models, world_up_idx, floor_offset=floor_offset)
            # shadow (NO)
        
        else: 
            print("No model")
            return

        color = shade_f_alpha(color_pack, shadow, depths, normals, ray_dir, light_dir)
        # escape = jnp.tanh(jnp.abs(hit_pos).max() - 10.0) * 0.5 + 0.5
        # color = color + (sky_color - color) * escape
        return color ** (1.0 / 2.2)  # gamma correction

    if intrinsic is None:
        intrinsic = cutil.default_intrinsic(pixel_size)
    _,_,ray_dir = cutil.pixel_ray(pixel_size, cam_pos=camera_pos, 
                cam_quat=camera_quat if camera_quat is not None else tutil.line2q(camera_pos-target_pos, camera_up), 
                intrinsic=intrinsic, near=0.001, far=3.0)
    ray_dir = ray_dir.reshape((-1,3))
    color = jax.vmap(render_ray_alpha)(ray_dir) # (NP, 3)
    rgb = color.reshape(*pixel_size, 3)
    rgb = jnp.where(jnp.isnan(rgb), 0, rgb)
    return rgb.clip(0,1.0)


def cvx_render_seg(
    objects:cxutil.CvxObjects,
    intrinsic=None,
    camera_pos=jnp.array([4.0, 4.0, 3.0])/3,
    camera_quat=None,
    models=None,
    sdf=None,
    pixel_size=(400, 640),
    target_pos=jnp.array([0.0, 0.0, 0.0]),
    camera_up=jnp.array([0,0,1.]),
    world_up_idx=2,
    floor_offset=-0.8,
):

    def render_ray(ray_dir):
        if models is not None:
            hit_pos, hit_id = rdf_raycast(objects, camera_pos, ray_dir, models=models, world_up_idx=world_up_idx, floor_offset=floor_offset, normal=False)
        elif sdf is not None:
            hit_pos, hit_id = sdf_raycast(sdf, objects, camera_pos, ray_dir, world_up_idx=world_up_idx, floor_offset=floor_offset, normal=False)
        else:
            hit_pos, hit_id = cvx_raycast(objects, camera_pos, ray_dir, world_up_idx=world_up_idx, floor_offset=floor_offset, normal=False)

        return hit_id  # gamma correction

    if intrinsic is None:
        intrinsic = cutil.default_intrinsic(pixel_size)
    _,_,ray_dir = cutil.pixel_ray(pixel_size, cam_pos=camera_pos, 
                cam_quat=camera_quat if camera_quat is not None else tutil.line2q(camera_pos-target_pos, camera_up), 
                intrinsic=intrinsic, near=0.001, far=3.0)
    ray_dir = ray_dir.reshape((-1,3))
    hit_id = jax.vmap(render_ray)(ray_dir)
    return hit_id.reshape(*pixel_size)

def rdf_seg(
    objects:cxutil.CvxObjects,
    models=None,
    intrinsic=None,
    pixel_size=(400, 640),
    target_pos=jnp.array([0.0, 0.0, 0.0]),
    camera_pos=jnp.array([4.0, 4.0, 3.0])/3,
    camera_quat=None,
    camera_up=jnp.array([0,0,1.])):

    if intrinsic is None:
        intrinsic = cutil.default_intrinsic(pixel_size)
    _,_,ray_dir = cutil.pixel_ray(pixel_size, cam_pos=camera_pos, 
                cam_quat=camera_quat if camera_quat is not None else tutil.line2q(camera_pos-target_pos, camera_up), 
                intrinsic=intrinsic, near=0.001, far=3.0)
    ray_dir = ray_dir.reshape(ray_dir.shape[:-3]+(-1,3))
    nc = intrinsic.shape[-2]
    if camera_pos.ndim != ray_dir.ndim:
        camera_pos = camera_pos[...,None,:]
        camera_pos = jnp.broadcast_to(camera_pos, jnp.broadcast_shapes(camera_pos.shape, ray_dir.shape))
        camera_pos, ray_dir = jax.tree_map(lambda x: einops.rearrange(x, '... i j k -> ... 1 (i j) k'), (camera_pos, ray_dir))
    res = models.apply('ray_predictor', objects, camera_pos, ray_dir)
    seg = jnp.squeeze(res[0], (-1,))
    depths = jnp.where(jax.nn.tanh(seg)>-0.80, jnp.squeeze(res[1], (-1,)), 20.0)
    normals =res[2]

    seg = seg.reshape(seg.shape[:-1]+(nc, pixel_size[0], pixel_size[1]))
    # seg = jnp.transpose(seg, (1, 0))

    # return 0.5*(jax.nn.tanh(seg) + 1.0).reshape((pixel_size[0], pixel_size[1]) + seg.shape[-1:])
    smooth = 5.
    return 0.5*(jax.nn.tanh(seg/smooth) + 1.0)


def cvx_raycast(objects:CvxObjects, p0, dir, world_up_idx=2, normal=False, depth=False, floor_offset=-0.8):
    # cvx_tf = tutil.pq_action(objects.pos[...,None,None,:], objects.quat[...,None,None,:], objects.cvx)
    if len(objects.outer_shape) == 0:
        objects = jax.tree_map(lambda x: x[None], objects)
    res = cxutil.ray_casting(p0, dir, objects.vtx_tf, objects.fc, min_depth=0.001, max_depth=100, normalize_dir=False)

    depths = jnp.squeeze(res[0], (-1,-2))
    normals = jnp.squeeze(res[1], -2)
    midx = jnp.argmin(depths)
    min_depth = depths[midx]
    obj_hit_p = p0 + dir*min_depth

    # floor calculation
    floor_hit_mask = obj_hit_p[world_up_idx]-floor_offset<0
    floor_hit_depth = (p0[world_up_idx]-floor_offset)/(p0[world_up_idx]-obj_hit_p[world_up_idx])*min_depth
    floor_hit_p = p0 + dir*floor_hit_depth
    hit_p = jnp.where(floor_hit_mask, floor_hit_p, obj_hit_p)
    hit_depth = jnp.where(floor_hit_mask, floor_hit_depth, min_depth)
    hit_id = jnp.where(floor_hit_mask, -1, midx)

    # hit check
    ishit = (hit_depth <9.9)
    hit_id = jnp.where(ishit, hit_id, -2)

    if depth:
        return hit_depth

    if normal:
        floor_normal = jnp.zeros((3,),dtype=jnp.float32).at[world_up_idx].set(1.0)
        normals = jnp.where(obj_hit_p[world_up_idx]-floor_offset>0, normals[midx], floor_normal)
        return hit_p, hit_id, normals
    else:
        return hit_p, hit_id


def rdf_raycast(objects:CvxObjects, p0, dir, models, k=1, world_up_idx=2, normal=False, depth=False, id_depth=False, floor_offset=-0.8, soft=False):
    '''
    objects outer_dim: (NO, )
    p0 (3,)
    dir (3,)
    use vmap for vector usage
    '''
    res = models.apply('ray_predictor', objects, p0, dir, train=False)
    floor_hit_depth = (p0[world_up_idx]-floor_offset)/(-jnp.sign(dir[world_up_idx])*jnp.abs(dir[world_up_idx]).clip(1e-5))
    floor_hit_depth = jnp.where(jnp.sign(dir[world_up_idx])>=-1e-6, 20.0, floor_hit_depth)
    if res[1].ndim == 2:
        depths = jnp.squeeze(res[1], (-1,))
        normals =res[2]
    else:
        seg = jnp.squeeze(res[0], (-1,))
        depths = jnp.squeeze(res[1], (-1,-2))
        normals = jnp.squeeze(res[2], -2)
    depths = jnp.r_[depths, [floor_hit_depth]]
    seg = jnp.r_[seg, [1e5]]
    if soft:
        seg_w = 0.5*(jax.nn.tanh(0.1*seg) + 1.)
        sort_idx = jnp.argsort(depths)
        seg_s = seg_w[sort_idx]
        depths_s = depths[sort_idx]
        cump = jnp.r_[[1.],jnp.cumprod((1-seg_s[:-1]))]
        min_depth = jnp.sum(depths_s * seg_s*cump)
    else:
        depths = jnp.where(jax.nn.tanh(seg)>-0.80, depths, 20.0)
        # pick botton-k
        midx = jnp.argsort(depths)[:k]
        min_depth = depths[midx]
    hit_p = p0 + dir*min_depth[0]
    hit_depth = min_depth
    if k==1:
        midx, hit_depth = jax.tree_map(lambda x: x.squeeze(-1), (midx, hit_depth))

    if not soft:
        hit_id = jnp.where(midx==len(depths)-1, -1, midx)

        # hit check
        ishit = (hit_depth <9.9)
        hit_id = jnp.where(ishit, hit_id, -2)

    if id_depth:
        assert not soft
        return hit_id, hit_depth

    if depth:
        return hit_depth

    if normal:
        # normals, _ = jax.grad(partial(scene_sdf, models.occ_predictor_model,  
        #             models.occ_predictor_params, models.occ_predictor_batch_stats,
        #             objects, up_idx=world_up_idx, floor_offset=floor_offset), has_aux=True)(hit_p)
        floor_normal = jnp.zeros((3,),dtype=jnp.float32).at[world_up_idx].set(1.0)
        normals = jnp.r_[normals, [floor_normal]][midx]
        # normals = jnp.where(obj_hit_p[world_up_idx]-floor_offset>0, normals[midx], floor_normal)
        normals = tutil.normalize(normals)
        # normals = jnp.where(obj_hit_p[world_up_idx]-floor_offset>0, normals, floor_normal)
        return hit_p, hit_id, normals
    else:
        return hit_p, hit_id


def rdf_raycast_alpha(objects:CvxObjects, cam_pos, dir, models, world_up_idx=2, normal=False, floor_offset=-0.8):
    res = models.apply('ray_predictor', objects, cam_pos, dir, train=False)
    floor_hit_depth = (cam_pos[world_up_idx]-floor_offset)/(-jnp.sign(dir[world_up_idx])*jnp.abs(dir[world_up_idx]).clip(1e-5))
    floor_hit_depth = jnp.where(jnp.sign(dir[world_up_idx])>=-1e-6, 200.0, floor_hit_depth)
    if res[0].ndim == 2:
        seg = jnp.squeeze(res[0], (-1,))
        # depths = jnp.where(jax.nn.tanh(seg)>-0.80, jnp.squeeze(res[1], (-1,)), 20.0)
        depths = jnp.squeeze(res[1], (-1,))
        normals =res[2]
    else:
        seg = jnp.squeeze(res[0], (-1,-2))
        # depths = jnp.where(jax.nn.tanh(seg)>-0.80, jnp.squeeze(res[1], (-1,-2)), 20.0)
        depths = jnp.squeeze(res[1], (-1,-2))
        normals = jnp.squeeze(res[2], -2)

    depths = jnp.r_[depths, [floor_hit_depth]]
    seg = jnp.r_[seg, [1e5]]

    depths = jnp.where(jax.nn.tanh(seg)>-0.0, depths, 200.0)[...,None] # (NO, 1)
    points = depths*dir+cam_pos  # (NO, 3)
    if not normal:
        return points, depths
    else:
        floor_normal = jnp.zeros((3,),dtype=jnp.float32).at[world_up_idx].set(1.0)
        normals = jnp.r_[normals, [floor_normal]]
        # normals = jnp.where(obj_hit_p[world_up_idx]-floor_offset>0, normals[midx], floor_normal)
        normals = tutil.normalize(normals)

        return points, depths, normals

from scipy.spatial.transform import Rotation as sciR
def texcoord_parameterization(x, param_type=None, rot_q=None):
    if rot_q is None:
        rot_q = tutil.qrand(())
    x_rot = sciR.from_quat(rot_q).apply(x)
    if param_type is None:
        param_type = np.random.randint(0,3)
    if param_type == 0:
        # sphere parameterization
        psi = np.arctan(np.linalg.norm(x_rot, axis=-1)/np.linalg.norm(x_rot[...,:2], axis=-1))
        theta = np.arctan2(x_rot[...,1], x_rot[...,0])
        uv_ = np.stack([theta, psi], axis=-1)
    if param_type == 1:
        # plane parameterization
        uv_ = x_rot[...,:2]
    if param_type == 2:
        # cylinder parametrization
        theta = np.arctan2(x_rot[...,1], x_rot[...,0])
        uv_ = np.stack([theta, x_rot[...,2]], axis=-1)

    return uv_%1

def gen_image(models, objects_particles, cam_info, output_pixel_size, zoom_ratio=1.0, ground_truth=False):
    '''
    particles : (NB NS ...)
    cam_info : (NB NC ...)
    '''
    if len(objects_particles.outer_shape) == 1:
        objects_particles = jax.tree_map(lambda x: x[None], objects_particles)
    if len(cam_info[0].shape) == 2:
        cam_info = jax.tree_map(lambda x: x[None], cam_info)
    
    cam_pq, cam_intrinsic = cam_info

    # reduce intrinsic for ray
    cam_pq_ext = cam_pq[...,None,:,:]
    if zoom_ratio == -1:
        cam_pq_ext = jnp.broadcast_to(cam_pq_ext, jnp.broadcast_shapes(cam_pq_ext.shape, objects_particles.pos[...,None,:1].shape)) # (NB NS NC)
        intrinsic_q = cam_intrinsic[...,None,:,:] # (NB NS NC)
    else:
        pixel_ij, out_pnts = cutil.global_pnts_to_pixel(cam_intrinsic[...,None,:,:], (cam_pq_ext[...,:3], cam_pq_ext[...,3:]), objects_particles.pos[...,None,:]) # (NB NO NC 3)
        pixel_xy = jnp.stack([pixel_ij[...,1], cam_intrinsic[...,None,:,1]-pixel_ij[...,0]], axis=-1)
        intrinsic_q = jnp.array(cam_intrinsic).at[...,1].set(output_pixel_size[0]).at[...,0].set(output_pixel_size[1])
        intrinsic_q = intrinsic_q[...,None,:,:]
        intrinsic_q = jnp.broadcast_to(intrinsic_q, jnp.broadcast_shapes(intrinsic_q.shape, pixel_xy[...,:1].shape))
        pixel_xy_bias = -(pixel_xy - 0.5*cam_intrinsic[...,None,:,:2])/zoom_ratio + jnp.array(output_pixel_size)[::-1] * 0.5
        intrinsic_q = intrinsic_q.at[...,4:6].set(pixel_xy_bias)
        intrinsic_q = intrinsic_q.at[...,2:4].set(intrinsic_q[...,2:4]/zoom_ratio)

    cam_pq_ext = jnp.broadcast_to(cam_pq_ext, jnp.broadcast_shapes(cam_pq_ext.shape, intrinsic_q[...,:1].shape)) # (NB, NO, NC, 7)
    _,_,ray_dir = cutil.pixel_ray(output_pixel_size, cam_pos=cam_pq_ext[...,:3], cam_quat=cam_pq_ext[...,3:], intrinsic=intrinsic_q, near=0.001, far=3.0)
    origin_shape = ray_dir.shape
    ray_dir = ray_dir.reshape((*ray_dir.shape[:3],-1,3)) # (NB, NO, NC, -1, 3)
    if ground_truth:
        ray_func = partial(rdf_raycast, models=models, world_up_idx=2, floor_offset=-0.5, depth=True)
    else:
        ray_func = partial(rdf_raycast, models=models, world_up_idx=2, floor_offset=-0.5, depth=True, soft=True)
    ray_func = jax.vmap(jax.vmap(jax.vmap(jax.vmap(ray_func, (None,None,0)), (None,0,0)), (0,0,0)), (0,0,0))
    depth = ray_func(objects_particles, jax.lax.stop_gradient(cam_pq_ext[...,:3]), jax.lax.stop_gradient(ray_dir)) # (NB, NO, NC)

    # ## test
    # print('test!!')
    # # depth = partial(rdf_raycast, models=models, world_up_idx=2, floor_offset=-0.5, depth=True)(objects_particles, jax.lax.stop_gradient(cam_pq_ext[0,0,0,:3]), jax.lax.stop_gradient(ray_dir[0,0,0]))
    # depth = partial(rdf_raycast, models=models, world_up_idx=2, floor_offset=-0.5, depth=True)(objects_particles, jnp.array([0,0,-1]), jnp.array([0,0,1]))
    # return depth

    NB = origin_shape[0]
    NC = cam_pq.shape[-2]
    NO = objects_particles.outer_shape[-1]
    depth = depth.reshape((NB, NO, NC, *output_pixel_size))
    return depth


def obj_ray_feat(models, origin_pixel_size, objs, img_feat, out_pixel_devider:int=4):
    '''
    test ray features
    input: cam_intrinsincs, cam_posquats, objects, pixel_size
    output: ray features
    '''
    extended = False
    if len(objs.outer_shape) == 1 and img_feat.intrinsic.ndim==2:
        extended = True
        objs, img_feat = jax.tree_map(lambda x: x[None], (objs, img_feat))

    pixel_size_to = (int(origin_pixel_size[0]//out_pixel_devider), int(origin_pixel_size[1]//out_pixel_devider))
    intrinsics = jax.vmap(cutil.resize_intrinsic, (0,None,None))(img_feat.intrinsic.astype(jnp.float32), origin_pixel_size, pixel_size_to)
    cam_posquats = img_feat.cam_posquat.astype(jnp.float32)

    # generates rays
    ray_s, ray_e, ray_dir = cutil.pixel_ray(pixel_size_to, cam_posquats[...,:3], cam_posquats[...,3:], intrinsics, near=0, far=2.0)
    
    obj_shape = ray_s.shape[-4:-1]
    ray_s, ray_e, ray_dir = jax.tree_map(lambda x: einops.rearrange(x, '... v i j k -> ... (v i j) k'), (ray_s, ray_e, ray_dir))
    ray_feat = jax.vmap(lambda x, y, z: models.apply('ray_predictor', x, y, z, feature_out=True), (1, None, None), 1)(objs, ray_s, ray_dir)
    ray_feat = einops.rearrange(ray_feat, '... (v i j) k -> ... v i j k', v=obj_shape[0], i=obj_shape[1], j=obj_shape[2])

    if extended:
        return ray_feat.squeeze(0)
    else:
        return ray_feat


def sdf_ray_segmentation(models, objects, ray_dir, p0, smooth=15.0):

    depth_resolution = [40, 40, 40]
    depth_candidate_range = [None, 0.08, 0.01]

    for i, depth_res_ in enumerate(depth_resolution):
        if i == 0:
            depth_candidates = jnp.linspace(0,1.0,depth_res_)[None,None]
        else:
            depth_candidates = jnp.linspace(depth-depth_candidate_range[i],depth+depth_candidate_range[i],depth_res_, axis=-1)
        occ_qpnts = p0[...,None,:] + ray_dir[...,None,:]*depth_candidates[...,None]
        objs_rs = objects.reshape_outer_shape((-1,))
        occ_qpnts_rs = occ_qpnts.reshape((occ_qpnts.shape[0], -1, occ_qpnts.shape[-1]))
        occ_values = models.apply('occ_predictor', objs_rs, occ_qpnts_rs)
        occ_values = occ_values.reshape((occ_values.shape[0], -1, depth_res_))
        depth_idx = jnp.argmax(occ_values, axis=-1)
        depth = jnp.take_along_axis(depth_candidates, depth_idx[...,None], axis=-1).squeeze(-1)
    seg = jnp.max(occ_values, axis=-1)
    seg = jax.nn.sigmoid((seg)/smooth)
    return depth, seg

def obj_segmentation(models, origin_pixel_size, objs, img_feat, out_pixel_devider:int=1, smooth=2, logit_bias=0.0, gaussian_filter=False, output_depth=False):
    '''
    test 
    input: img_feat (NB, NC) or (NC)
        objects : (NB, NO ...) or (NO)
        pixel_size
    output: segmentations # (NB, NO, NC, NI, NJ)
    '''
    extended = False
    if len(objs.outer_shape) == 1:
        extended = True
        objs= jax.tree_map(lambda x: x[None], objs)
    if img_feat.intrinsic.ndim==2:
        # extended = True
        img_feat = jax.tree_map(lambda x: x[None], img_feat)

    pixel_size_to = (int(origin_pixel_size[0]//out_pixel_devider), int(origin_pixel_size[1]//out_pixel_devider))
    intrinsics = jax.vmap(cutil.resize_intrinsic, (0,None,None))(img_feat.intrinsic.astype(jnp.float32), origin_pixel_size, pixel_size_to)
    cam_posquats = img_feat.cam_posquat.astype(jnp.float32)

    # generates rays
    ray_s, ray_e, ray_dir = cutil.pixel_ray(pixel_size_to, cam_posquats[...,:3], cam_posquats[...,3:], intrinsics, near=0, far=1.0)
    obj_shape = ray_s.shape[-4:-1]
    ray_s, ray_e, ray_dir = jax.tree_map(lambda x: einops.rearrange(x, '... v i j k -> ... (v i j) k'), (ray_s, ray_e, ray_dir))

    if hasattr(models.dif_args, 'implicit_baseline') and models.dif_args.implicit_baseline:
        # sdf_func = partial(scene_sdf, models=models)
        # origin_outer_shape = objs.outer_shape
        # objs_rs = objs.reshape_outer_shape((-1,1,1))
        # ns_tmp = objs_rs.outer_shape[0]
        # depth, seg = jax.vmap(jax.vmap(partial(sdf_ray_segmentation, sdf_func, floor_offset=-10.0, step_n=50, hardness=50.0)))(objs_rs.repeat_outer_shape(ray_dir.shape[1],1),
        #                                                                                                           einops.repeat(ray_dir, 'i j ... -> (r i) j ...', r=ns_tmp), 
        #                                                                                                           einops.repeat(ray_s, 'i j ... -> (r i) j ...', r=ns_tmp))
        # seg = 1-seg
        # seg = seg.reshape(origin_outer_shape + obj_shape)
        # depth = depth.reshape(origin_outer_shape + obj_shape)
        
        depth_resolution = [40, 40, 40]
        depth_candidate_range = [None, 0.08, 0.01]

        original_outer_shape = objs.outer_shape
        objs_rs = objs.reshape_outer_shape((-1,))
        for i, depth_res_ in enumerate(depth_resolution):
            if i == 0:
                depth_candidates = jnp.linspace(0,1.0,depth_res_)[None,None]
            else:
                depth_candidates = jnp.linspace(depth-depth_candidate_range[i],depth+depth_candidate_range[i],depth_res_, axis=-1)
            occ_qpnts = ray_s[...,None,:] + ray_dir[...,None,:]*depth_candidates[...,None]
            occ_qpnts_rs = occ_qpnts.reshape((occ_qpnts.shape[0], -1, occ_qpnts.shape[-1]))
            occ_values = models.apply('occ_predictor', objs_rs, occ_qpnts_rs)
            occ_values = occ_values.reshape((occ_values.shape[0], -1, depth_res_))
            depth_idx = jnp.argmax(occ_values, axis=-1)
            depth = jnp.take_along_axis(depth_candidates, depth_idx[...,None], axis=-1).squeeze(-1)
        occ_values = occ_values.reshape(original_outer_shape + occ_values.shape[-2:])
        depth = depth.reshape(original_outer_shape + depth.shape[-1:])
        seg = jnp.max(occ_values, axis=-1)
        seg = jax.nn.sigmoid((seg+logit_bias)/smooth)
        seg = einops.rearrange(seg, '... (v i j) -> ... v i j', v=obj_shape[0], i=obj_shape[1], j=obj_shape[2]) # (NB NO NC NI NJ)
        depth = einops.rearrange(depth, '... (v i j) -> ... v i j', v=obj_shape[0], i=obj_shape[1], j=obj_shape[2]) # (NB NO NC NI NJ)
    else:
        seg, depth, normal = jax.vmap(lambda x, y, z: models.apply('ray_predictor', x, y, z, feature_out=False), (1, None, None), 1)(objs, ray_s, ray_dir)
        seg = einops.rearrange(seg, '... (v i j) -> ... v i j', v=obj_shape[0], i=obj_shape[1], j=obj_shape[2]) # (NB NO NC NI NJ)
        seg = jax.nn.sigmoid((seg+logit_bias)/smooth)
        depth = einops.rearrange(depth.squeeze(-1), '... (v i j) -> ... v i j', v=obj_shape[0], i=obj_shape[1], j=obj_shape[2]) # (NB NO NC NI NJ)

    if gaussian_filter:
        origin_shape = seg.shape
        seg = seg.reshape((-1,)+origin_shape[-2:])
        seg =  pix.gaussian_blur(seg[...,None], 3, kernel_size=5)
        seg = seg.reshape(origin_shape)


    if extended:
        if output_depth:
            return seg.squeeze(0), depth.squeeze(0)
        else:
            return seg.squeeze(0)
    else:
        if output_depth:
            return seg, depth
        else:
            return seg
    


def obj_segmentation_gt(origin_pixel_size, objs, img_feat, out_pixel_devider:int=1):
    '''
    test 
    input: img_feat (NB, NC) or (NC)
        objects : (NB, NO ...) or (NO)
        pixel_size
    output: segmentations
    '''
    extended = False
    if len(objs.outer_shape) == 1:
        extended = True
        objs= jax.tree_map(lambda x: x[None], objs)
    if img_feat.intrinsic.ndim==2:
        extended = True
        img_feat = jax.tree_map(lambda x: x[None], img_feat)
    

    pixel_size_to = (int(origin_pixel_size[0]//out_pixel_devider), int(origin_pixel_size[1]//out_pixel_devider))
    intrinsics = jax.vmap(cutil.resize_intrinsic, (0,None,None))(img_feat.intrinsic.astype(jnp.float32), origin_pixel_size, pixel_size_to)
    cam_posquats = img_feat.cam_posquat.astype(jnp.float32)

    seg_func = lambda obj, intr, cpq: cvx_render_seg(
                obj,
                intrinsic=intr,
                camera_pos=cpq[...,:3],
                camera_quat=cpq[...,3:],
                models=None,
                sdf=None,
                pixel_size=pixel_size_to,
                floor_offset=-0.8,
            )
    if intrinsics.shape[0] != objs.outer_shape[0]:
        intrinsics, cam_posquats = jax.tree_map(lambda x: einops.repeat(x, 'i ... -> (r i) ...', r=objs.outer_shape[0]), (intrinsics, cam_posquats))
    
    # seg = jax.vmap(jax.vmap(seg_func, (None,0,0)))(objs, intrinsics, cam_posquats) # (NB NC NI NJ 1)

    seg = []
    for v in range(intrinsics.shape[1]):
        seg.append(jax.vmap(seg_func)(objs, intrinsics[:,v], cam_posquats[:,v])) # (NB NC NI NJ 1)
    seg = jnp.stack(seg, axis=1)

    seg = seg>=0

    if extended:
        return seg.squeeze(0)
    else:
        return seg
