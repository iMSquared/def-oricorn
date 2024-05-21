import os, sys
if __name__ == '__main__':
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"   
    os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
import jax
import jax.numpy as jnp
import numpy as np
import einops
from functools import partial
import optax

BASEDIR = os.path.dirname(os.path.dirname(__file__))
if BASEDIR not in sys.path:
    sys.path.insert(0, BASEDIR)

import util.model_util as mutil
import util.cvx_util as cxutil
import util.transform_util as tutil

def chamfer_distance(j, carry):
    source_pnts, target_pnts, dist = carry
    source_one_pnt = source_pnts[j]
    total_dist = jnp.linalg.norm(target_pnts-source_one_pnt, axis=-1, ord=1)
    sorted_dist = jnp.sort(total_dist)
    dist += (sorted_dist[0]+sorted_dist[1]+sorted_dist[2])/3.
    return source_pnts, target_pnts, dist

def chamfer_distance_batch(source_pnts, target_pnts):
    k = 3
    total_dist = jnp.linalg.norm(target_pnts[...,None,:]-source_pnts[...,None,:,:], axis=-1)
    sorted_dist = jnp.sort(total_dist, axis=-1)
    dist = jnp.mean(jnp.sum(sorted_dist[...,:k], axis=-1)/k, axis=-1)

    total_dist = einops.rearrange(total_dist, '... i j -> ... j i')
    sorted_dist2 = jnp.sort(total_dist, axis=-1)
    dist2 = jnp.mean(jnp.sum(sorted_dist2[...,:k], axis=-1)/k, axis=-1)
    return 0.5*(dist + dist2)


def get_pcd_from_latent_w_voxel(jkey, query_objects:cxutil.LatentObjects, num_points:int, models:mutil.Models, visualize=False):
    '''
    query_objects: outer_shape - (nb, ...) or (...)
    num_points: number of output points
    
    return surface points (nb, num_points, 3)
    '''

    # preprocessing size
    if len(query_objects.outer_shape) == 0:
        query_objects = jax.tree_map(lambda x: x[None], query_objects)
    
    # define hyper parameters
    initial_density = 100
    initial_sample_num_pnts = initial_density**3
    intermediate_sample_num_pnts = 20000
    initial_bound_half_len = 0.25
    occ_boundary_value = -0.5
    occ_logit_threshold_for_surface = 8.0
    coarse_to_fine_itr_no = 2
    occ_logit_threshold_for_surface_l2 = [0.4, 0.1]
    assert coarse_to_fine_itr_no == len(occ_logit_threshold_for_surface_l2)

    nb =query_objects.outer_shape[0]

    dec = partial(models.apply, 'occ_predictor')

    # generate initial query points from grid
    x = np.linspace(-initial_bound_half_len, initial_bound_half_len, initial_density+1)
    y = np.linspace(-initial_bound_half_len, initial_bound_half_len, initial_density+1)
    z = np.linspace(-initial_bound_half_len, initial_bound_half_len, initial_density+1)
    xv, yv, zv = np.meshgrid(x, y, z)
    grid = np.stack([xv, yv, zv]).astype(np.float32).reshape(3, -1).transpose()[None]
    grid = jnp.array(grid)
    query_points_l1 = grid + query_objects.pos[...,None,:]

    # evaluate and gather valid points for level 1
    _, jkey = jax.random.split(jkey)
    occ_res = dec(query_objects, query_points_l1, jkey)
    within_mask_l1 = jnp.abs(occ_res-occ_boundary_value)<occ_logit_threshold_for_surface
    if visualize:
        print(jnp.sum(within_mask_l1))
    _, jkey = jax.random.split(jkey)
    valid_pnts_l1 = jax.vmap(lambda jk, a, p: jax.random.choice(jk, a, shape=(initial_sample_num_pnts,), p=p))(jax.random.split(jkey, nb), query_points_l1, within_mask_l1.astype(jnp.float32))
    _, jkey = jax.random.split(jkey)

    # level 2 querying
    bound_half_len_l2 = initial_bound_half_len/initial_density*2
    valid_pnts = valid_pnts_l1
    for i in range(coarse_to_fine_itr_no):
        query_points_l2 = jax.random.uniform(jkey, shape=query_objects.outer_shape + (initial_sample_num_pnts,3), minval=-bound_half_len_l2, maxval=bound_half_len_l2)
        _, jkey = jax.random.split(jkey)
        query_points_l2 = valid_pnts + query_points_l2
        occ_res_l2 = dec(query_objects, query_points_l2, jkey)
        within_mask_l2 = jnp.abs(occ_res_l2-occ_boundary_value)<occ_logit_threshold_for_surface_l2[i]
        if visualize:
            print(jnp.sum(within_mask_l2))
        valid_pnts_l2 = jax.vmap(lambda jk, a, p: jax.random.choice(jk, a, shape=(intermediate_sample_num_pnts,), p=p))(jax.random.split(jkey, nb), query_points_l2, within_mask_l2.astype(jnp.float32))
        
        # make density uniform
        resolution = 0.005 # resolution to calculate density in meter
        unique_check = (valid_pnts_l2/resolution).astype(jnp.int32)
        unique_count = jnp.sum(jnp.all(unique_check[...,None,:] == unique_check[...,None,:,:], axis=-1), axis=-1)
        assert intermediate_sample_num_pnts >= num_points
        if i==coarse_to_fine_itr_no-1:
            valid_pnts_l2 = jax.vmap(lambda jk, a, p: jax.random.choice(jk, a, shape=(num_points,), p=p))(jax.random.split(jkey, nb), valid_pnts_l2, 1/unique_count.astype(jnp.float32))
            break
        valid_pnts_l2 = jax.vmap(lambda jk, a, p: jax.random.choice(jk, a, shape=(initial_sample_num_pnts,), p=p))(jax.random.split(jkey, nb), valid_pnts_l2, 1/unique_count.astype(jnp.float32))
        _, jkey = jax.random.split(jkey)
        valid_pnts = valid_pnts_l2

    # perform optimization
    def occupancy_loss(x):
        occ = models.apply('occ_predictor', query_objects, x) # (#pnt, 1)
        return jnp.sum(jnp.abs(occ-occ_boundary_value))

    grad_func = jax.value_and_grad(occupancy_loss)
    optimizer = optax.adam(2e-5)
    surface_pnts = valid_pnts_l2
    opt_state = optimizer.init(surface_pnts)
    for _ in range(10):
        loss, grad = grad_func(surface_pnts)
        updates, opt_state = optimizer.update(grad, opt_state, surface_pnts)
        surface_pnts = optax.apply_updates(surface_pnts, updates)
        if visualize:
            print(loss)
    
    # surface_normals = jax.grad(lambda x: jnp.sum(models.apply('occ_predictor', query_objects, x)))(surface_pnts)

    if visualize:
        # visualization
        import open3d as o3d
        for i in range(nb):
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(np.array(query_points_l1[within_mask_l1]))
            o3d.visualization.draw_geometries([pcd])

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(np.array(query_points_l2[within_mask_l2]))
            o3d.visualization.draw_geometries([pcd])

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(np.array(valid_pnts_l2[i]))
            o3d.visualization.draw_geometries([pcd])

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(np.array(surface_pnts)[i])
            o3d.visualization.draw_geometries([pcd])

    return surface_pnts



def get_pcd_from_latent(jkey, query_objects:cxutil.CvxObjects, num_points:int, models:mutil.Models, visualize=False, num_itr=20):
    '''
    query_objects : (NB ...) or (...)
    '''
    extended = False
    if len(query_objects.outer_shape) == 0:
        extended = True
        query_objects = jax.tree_map(lambda x: x[None], query_objects)

    def vis_pcd(query_points, query_points2 = None):
        mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=0.1, origin=[0, 0, 0])
        numpy_query_points = np.array(query_points)
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(numpy_query_points)
        point_cloud.colors = o3d.utility.Vector3dVector(np.ones_like(numpy_query_points) * np.array([0,0,1.]))
        if query_points2 is not None:
            numpy_query_points2 = np.array(query_points2)
            point_cloud2 = o3d.geometry.PointCloud()
            point_cloud2.points = o3d.utility.Vector3dVector(numpy_query_points2)
            point_cloud2.colors = o3d.utility.Vector3dVector(np.ones_like(numpy_query_points2) * np.array([1,0,0.]))
            o3d.visualization.draw_geometries([mesh_frame, point_cloud, point_cloud2])
        else:
            o3d.visualization.draw_geometries([mesh_frame, point_cloud])

    radius = 0.5
    def one_batch_pcd_extractor(obj_particle):
        def occupancy_loss(x, obj_particle):
            occ = models.apply('occ_predictor', obj_particle, x) # (#pnt, 1)
            occ = jnp.abs(occ)
            return jnp.sum(occ)
            # return -occ
        
        def distance_loss(x, obj_pcd):
            return distance_btw_points(x+1e-5, obj_pcd)

        def distance_btw_points(pnt, query_pnts, r=0.005):
            pairwise_dist = jnp.linalg.norm(query_pnts-pnt, axis=1, ord=1) ** 2 
            k = int(query_pnts.shape[0] * r)
            k = np.maximum(k, 5)
            pairwise_dist = jnp.sort(pairwise_dist)[...,:k]
            h = jnp.median(pairwise_dist)
            h = jnp.sqrt(0.5 * h / jnp.log(pairwise_dist.shape[0]+1))
            rbf = jnp.exp(-pairwise_dist / h**2 / 2)
            total = jnp.sum(rbf)

            return total

        def gradient_of_point(point, query_pnts):
            occ_value, occ_grad = jax.value_and_grad(occupancy_loss)(point, obj_particle)
            dist_loss = jax.grad(distance_loss)(point, query_pnts)
            normal_dir = tutil.normalize(occ_grad)
            dist_loss = dist_loss - jnp.einsum('...i,...i', dist_loss, normal_dir)[...,None] * normal_dir
            dist_grad_norm = jnp.linalg.norm(dist_loss)
            dist_loss = dist_grad_norm.clip(200)*dist_loss/(dist_grad_norm+1e-5)
            # return jnp.abs(occ_value)*normal_dir + jnp.where(jnp.abs(occ_value)<0.1, 0.005, 0)*dist_loss
            return jnp.abs(occ_value)*normal_dir + 0.005*dist_loss, normal_dir


        def query_point_grad_descent(j, carry):
            query_points, _ = carry
            gradient, normal_dir = jax.vmap(gradient_of_point, (0,None))(query_points, query_points)
            # learning_rate = 7e-6*jnp.power(0.99, j) 
            # learning_rate = jnp.where(learning_rate < 5e-7, 5e-7, learning_rate)
            learning_rate = 8e-4
            query_points = query_points - learning_rate*gradient
            return query_points, normal_dir

        center = obj_particle.pos
        points = jax.random.normal(jkey, shape=(num_points, 3))
        points = radius*tutil.normalize(points)
        dirs = -points
        points = points + center
        dirs = tutil.normalize(dirs)

        segment, depth, normals = models.apply('ray_predictor', obj_particle, points, dirs)

        obj_pcd = points + dirs*(depth - 0.030)
        obj_pcd = jnp.where(segment>=0, obj_pcd, 0)

        refined_obj_pcd, surface_normal = jax.lax.fori_loop(0, num_itr, query_point_grad_descent, (obj_pcd, jnp.zeros_like(obj_pcd)))

        return obj_pcd, refined_obj_pcd, surface_normal

    rough_obj_pcd, refined_obj_pcd, surface_normal = jax.vmap(one_batch_pcd_extractor)(query_objects)
    if visualize:
        import open3d as o3d
        for i in range(rough_obj_pcd.shape[0]):
            vis_pcd(rough_obj_pcd[i], refined_obj_pcd[i])

    if extended:
        refined_obj_pcd = jnp.squeeze(refined_obj_pcd, axis=0)
        surface_normal = jnp.squeeze(surface_normal, axis=0)
    return refined_obj_pcd, surface_normal


if __name__ == '__main__':
    import time
    import util.environment_util as env_util
    import util.io_util as ioutil

    base_len = 0.65
    categorical_scale = {'can':1/base_len*0.15, 'bottle':1/base_len*0.15, 'bowl':1/base_len*0.2, 
                                    'camera':1/base_len*0.15, 'laptop':1/base_len*0.3, 'mug':1/base_len*0.12}

    BUILD = ioutil.BuildMetadata.from_str("32_64_1_v4")

    ckpt_dir = 'logs_dif/02262024-180216'
    models = mutil.get_models_from_cp_dir(ckpt_dir)

    jkey = jax.random.PRNGKey(0)

    # mesh_path, scale = 'data/NOCS_val/32_64_1_v4/bowl-e816066ac8281e2ecf70f9641eb97702.obj', categorical_scale['bowl']
    mesh_path, scale = 'data/NOCS_val/32_64_1_v4/mug-ea33ad442b032208d778b73d04298f62.obj', categorical_scale['mug']
    # mesh_path, scale = 'data/NOCS_val/32_64_1_v4/mug-f1866a48c2fc17f85b2ecd212557fda0.obj', categorical_scale['mug']
    # mesh_path, scale = 'data/NOCS_val/32_64_1_v4/mug-e16a895052da87277f58c33b328479f4.obj', categorical_scale['mug']
    # mesh_path, scale = 'data/NOCS_val/32_64_1_v4/bottle-dc687759ea93d1b72cd6cd3dc3fb5dc2.obj', categorical_scale['bottle']

    query_objects = env_util.create_cvx_objects(jkey, [mesh_path], BUILD, scale)
    _, jkey = jax.random.split(jkey)
    query_objects = query_objects.set_z_with_models(jkey, models, keep_gt_info=False)
    _, jkey = jax.random.split(jkey)

    sample_jit_func = jax.jit(partial(get_pcd_from_latent_w_voxel, num_points=5000, models=models, visualize=False))
    spoints = sample_jit_func(jkey, query_objects)
    # spoints = partial(get_pcd_from_latent_w_voxel, num_points=10000, models=models, visualize=True)(jkey, query_objects)

    start_time = time.time()
    for _ in range(100):
        spoints = sample_jit_func(jkey, query_objects)
        spoints = jax.block_until_ready(spoints)
    end_time = time.time()

    print((end_time - start_time)/100)

    import open3d as o3d
    for i in range(spoints.shape[0]):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.array(spoints)[i])
        # pcd.normals = o3d.utility.Vector3dVector(np.array(snormals)[i])
        o3d.visualization.draw_geometries([pcd])

    print(1)