import jax
import jax.numpy as jnp
import numpy as np
import mcubes
import open3d as o3d
import os, sys
import importlib
import pickle
import glob
from functools import partial

BASEDIR = os.path.dirname(os.path.dirname(__file__))
if BASEDIR not in sys.path:
    sys.path.insert(0, BASEDIR)
import util.cvx_util as cxutil
import util.transform_util as trutil

def create_mesh(args, jkey, checkpoint_dir, mesh_path, models=None, input_type='cvx', rot_aug=True, visualize=False):
    if models is None:
        with open(os.path.join(checkpoint_dir, 'saved.pkl'), 'rb') as f:
            models = pickle.load(f)['models']
    args = models.args
    enc = jax.jit(partial(models.apply, 'shape_encoder'))
    dec = jax.jit(partial(models.apply, 'occ_predictor'))

    base_len = 0.65
    categorical_scale = {'can':1/base_len*0.15, 'bottle':1/base_len*0.15, 'bowl':1/base_len*0.2, 
                                    'camera':1/base_len*0.15, 'laptop':1/base_len*0.3, 'mug':1/base_len*0.12}
    category = os.path.basename(mesh_path).split('-')[0]

    vtx_, fc_, _, _ = cxutil.vex_obj_parsing(mesh_path, max_dec_size=32, max_vertices=64, scale=categorical_scale[category])
    spnts, dc_idx = cxutil.sampling_from_surface_convex_dec(jkey, vtx_[None], fc_[None], args.npoint)
    obj = cxutil.CvxObjects().init_vtx(vtx_[None], fc_[None]).init_pcd(spnts, dc_idx)

    # obj = cxutil.create_shelf_dc(32, 64, 0.3, 0.6, 0.2, 0.05).register_pcd(jkey, models.args.npoint)

    # obj = cxutil.create_box(jnp.array([0.5,0.2,0.04]),32, 64).register_pcd(jkey, models.args.npoint)
    # obj = jax.tree_map(lambda x: x[None], obj)

    if rot_aug:
        qrand = trutil.qrand((), jkey)
        obj = obj.rotate_rel_vtxpcd(qrand)
    _, jkey = jax.random.split(jkey)
    emb = enc(obj, jkey, True)
    input_pcd = o3d.geometry.PointCloud()
    input_pcd.points = o3d.utility.Vector3dVector(np.array(obj.pcd_tf[0]))

    # mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, origin=np.array([0., 0., 0.]))
    # o3d.visualization.draw_geometries([input_pcd, mesh_frame])

    # marching cube
    occupancy_value_threshold = -0.5
    density = 202
    # density = 128
    qp_bound = 0.2
    # qp_bound = 1.0

    gap = 2*qp_bound / density
    x = np.linspace(-qp_bound, qp_bound, density+1)
    y = np.linspace(-qp_bound, qp_bound, density+1)
    z = np.linspace(-qp_bound, qp_bound, density+1)
    xv, yv, zv = np.meshgrid(x, y, z)
    grid = np.stack([xv, yv, zv]).astype(np.float32).reshape(3, -1).transpose()[None]
    grid = jnp.array(grid)
    ndiv = 50
    output = None
    dif = grid.shape[1]//ndiv
    for i in range(ndiv+1):
        _, jkey = jax.random.split(jkey)
        print(i)
        grid_ = grid[:,dif*i:dif*(i+1)]
        output_ = dec(emb, grid_, jkey)[0]
        if output is None:
            output = output_
        else:
            output = jnp.concatenate([output, output_], 0)
    volume = output.reshape(density+1, density+1, density+1).transpose(1, 0, 2)
    volume = np.array(volume)
    # print("start smoothing")
    # volume = mcubes.smooth(volume)
    # print("end smoothing")
    verts, faces = mcubes.marching_cubes(volume, occupancy_value_threshold)
    verts *= gap
    verts -= qp_bound

    mesh = o3d.geometry.TriangleMesh(vertices=o3d.utility.Vector3dVector(verts), triangles=o3d.utility.Vector3iVector(faces))
    mesh.compute_vertex_normals()
    # o3d.visualization.draw_geometries([mesh])

    # same mesh
    # o3d.io.write_triangle_mesh(os.path.join(os.path.dirname(save_dir), os.path.basename(mesh_fn)), mesh)

    # print("Cluster connected triangles")
    # with o3d.utility.VerbosityContextManager(
    #         o3d.utility.VerbosityLevel.Debug) as cm:
    #     triangle_clusters, cluster_n_triangles, cluster_area = (
    #         mesh.cluster_connected_triangles())
    # triangle_clusters = np.asarray(triangle_clusters)
    # cluster_n_triangles = np.asarray(cluster_n_triangles)
    # cluster_area = np.asarray(cluster_area)

    # import copy
    # print("Show mesh with small clusters removed")
    # mesh_0 = copy.deepcopy(mesh)
    # triangles_to_remove = cluster_n_triangles[triangle_clusters] < 10000
    # mesh_0.remove_triangles_by_mask(triangles_to_remove)

    if visualize:
        mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, origin=np.array([0., 0., 0.]))
        o3d.visualization.draw_geometries([mesh, input_pcd, mesh_frame])

    return mesh, os.path.basename(mesh_path)



def create_mesh_from_latent(jkey, models, latent_obj, density=128, qp_bound=0.20, ndiv=20, visualize=False)->o3d.geometry.TriangleMesh:
    
    dec = jax.jit(partial(models.apply, 'occ_predictor'))

    if len(latent_obj.outer_shape) == 0:
        latent_obj = jax.tree_map(lambda x: x[None], latent_obj)
        latent_obj = latent_obj.replace(pos=jnp.zeros_like(latent_obj.pos))

    # marching cube
    gap = 2*qp_bound / density
    x = np.linspace(-qp_bound, qp_bound, density+1)
    y = np.linspace(-qp_bound, qp_bound, density+1)
    z = np.linspace(-qp_bound, qp_bound, density+1)
    xv, yv, zv = np.meshgrid(x, y, z)
    grid = np.stack([xv, yv, zv]).astype(np.float32).reshape(3, -1).transpose()[None]
    grid = jnp.array(grid)
    output = None
    dif = grid.shape[1]//ndiv
    for i in range(ndiv+1):
        _, jkey = jax.random.split(jkey)
        grid_ = grid[:,dif*i:dif*(i+1)]
        output_ = dec(latent_obj, grid_, jkey)
        output_ = jnp.max(output_, axis=0) # union
        if output is None:
            output = np.array(output_)
        else:
            output = np.concatenate([output, np.array(output_)], 0)
    volume = output.reshape(density+1, density+1, density+1).transpose(1, 0, 2)
    volume = np.array(volume)
    # print("start smoothing")
    # volume = mcubes.smooth(volume)
    # print("end smoothing")
    # verts, faces = mcubes.marching_cubes(volume, 0)
    verts, faces = mcubes.marching_cubes(volume, -1)
    verts *= gap
    verts -= qp_bound

    mesh = o3d.geometry.TriangleMesh(vertices=o3d.utility.Vector3dVector(verts), triangles=o3d.utility.Vector3iVector(faces))
    # mesh.compute_vertex_normals()

    if visualize:
        mesh.compute_vertex_normals()
        mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, origin=np.array([0., 0., 0.]))
        o3d.visualization.draw_geometries([mesh, mesh_frame])

    return mesh

if __name__ == '__main__':
    # save_dir = 'checkpoints/pretraining/01082024-074428' # 64-32 with low reg
    # save_dir = 'checkpoints/pretraining/01082024-074428' # cherrypick2
    # save_dir = 'checkpoints/pretraining/12292023-095943' # None
    # save_dir = 'checkpoints/pretraining/12282023-121639' # cherrypick
    save_dir = 'checkpoints/pretraining/01152024-074410' # NOCS

    
    # mesh_path = 'data/DexGraspNet/32_64_1_v4/core-camera-cda4fc24b2a602b5b5328fde615e4a0c.obj'
    # mesh_path = 'data/DexGraspNet/32_64_1_v4/core-camera-e9e22de9e4c3c3c92a60bd875e075589.obj'
    # mesh_path = 'data/DexGraspNet/32_64_1_v4/sem-Bowl-8eab5598b81afd7bab5b523beb03efcd.obj'
    # mesh_path = 'data/DexGraspNet/32_64_1_v4/core-jar-ace45c3e1de6058ee694be5819d751ba.obj'
    # mesh_path = 'data/DexGraspNet/32_64_1_v4/sem-ToyFigure-6897b6e1470cdd4af8e8ff67ecd10c0b.obj'
    # mesh_path = 'data/DexGraspNet/32_64_1_v4/sem-Piano-7f958d2487a826bcd11510e8e4b5518b.obj'
    # mesh_path = 'data/DexGraspNet/32_64_1_v4/sem-Hammer-94dbb6874c576ee428bcb63ced82086c.obj'
    # mesh_path = 'data/DexGraspNet/32_64_1_v4/core-mug-6c379385bf0a23ffdec712af445786fe.obj'
    # mesh_path = 'data/DexGraspNet/32_64_1_v4/mujoco-Playmates_nickelodeon_teenage_mutant_ninja_turtles_shredder.obj'
    # mesh_path = 'data/DexGraspNet/32_64_1_v4/mujoco-Reebok_ULTIMATIC_2V.obj'
    # mesh_path = 'data/DexGraspNet/32_64_1_v4/sem-Cup-d8ea3aa39bcb162798910e50f05b8001.obj'
    # mesh_path = 'data/shelf/32_64_1_v4/wall-shelf-028.obj'
    # mesh_path = 'data/shelf/32_64_1_v4/shelf-045.obj'

    # mesh_path = 'data/NOCS/32_64_1_v4/bottle-1e5abf0465d97d826118a17db9de8c0.obj'
    # mesh_path = 'data/NOCS/32_64_1_v4/bowl-6816c12cc00a961758463a756b0921b5.obj'
    # mesh_path = 'data/NOCS/32_64_1_v4/mug-214dbcace712e49de195a69ef7c885a4.obj'
    # mesh_path = 'data/NOCS_val/32_64_1_v4/mug-ea33ad442b032208d778b73d04298f62.obj'
    mesh_path = 'data/NOCS_val/32_64_1_v4/mug-f1866a48c2fc17f85b2ecd212557fda0.obj'

    
    
    jkey = jax.random.PRNGKey(0)
    mesh_0, mesh_basename = create_mesh(None, jkey, save_dir, mesh_path, models=None, input_type='cvx', visualize=True)

    # o3d.io.write_triangle_mesh(os.path.join(os.path.dirname(save_dir), mesh_basename), mesh_0)