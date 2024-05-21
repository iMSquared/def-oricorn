import jax.numpy as jnp
from pathlib import Path
import jax
from ott.geometry import pointcloud, costs
from ott.solvers import linear
import typing
import optax
from functools import partial
import sys

BASEDIR = Path(__file__).parent.parent
if str(BASEDIR) not in sys.path:
    sys.path.insert(0, str(BASEDIR))


import util.diffusion_util as dfutil
import util.cvx_util as cxutil
import util.render_util as rutil
import util.model_util as mutil
import util.transform_util as tutil
import util.camera_util as cutil
import util.structs as structs
import util.train_util as trutil


@jax.tree_util.register_pytree_node_class
class L1Cost(costs.CostFn):
    """distance function for convex elements."""

    def __init__(self, latent_shape: typing.Tuple, dc_pos_loss_coef, pos_loss_coef):
        super().__init__()
        self.latent_shape = latent_shape
        # self.args = args
        self.dc_pos_loss_coef = dc_pos_loss_coef
        self.pos_loss_coef = pos_loss_coef

    def pairwise(self, h1: jnp.ndarray, h2: jnp.ndarray) -> float:
        """ """
        obj1 = cxutil.LatentObjects().init_h(h1, self.latent_shape)
        obj2 = cxutil.LatentObjects().init_h(h2, self.latent_shape)
        pos_dif = jnp.sum(jnp.abs(obj1.pos - obj2.pos) ** 2, axis=-1)
        if self.dc_pos_loss_coef==0:
            cen_pwdif = 0
        else:
            cen_pwdif = jnp.sum(
                jnp.abs(obj1.dc_centers_tf - obj2.dc_centers_tf) ** 2, axis=(-1, -2)
            )
        z_pwdif = jnp.sum(jnp.abs(obj1.z_flat - obj2.z_flat) ** 2, axis=(-1, -2))
        chloss = z_pwdif + self.dc_pos_loss_coef*cen_pwdif + self.pos_loss_coef*pos_dif
        return chloss

    def tree_flatten(self):
        return (), (self.latent_shape, self.dc_pos_loss_coef, self.pos_loss_coef)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        del children
        return cls(*aux_data)


@jax.tree_util.register_pytree_node_class
class DFCost(costs.CostFn):
    """distance function for convex elements."""

    def __init__(self, latent_shape: typing.Tuple, dc_pos_loss_coef, pos_loss_coef):
        super().__init__()
        self.latent_shape = latent_shape
        self.dc_pos_loss_coef = dc_pos_loss_coef
        self.pos_loss_coef = pos_loss_coef

    def pairwise(self, h1: jnp.ndarray, h2: jnp.ndarray) -> float:
        """ """
        obj1 = cxutil.LatentObjects().init_h(h1, self.latent_shape)
        obj2 = cxutil.LatentObjects().init_h(h2, self.latent_shape)
        matching_coef = 2
        # geom_ott = pointcloud.PointCloud(obj1.get_df(matching_coef*self.dc_pos_loss_coef), obj2.get_df(matching_coef*self.dc_pos_loss_coef))
        geom_ott = pointcloud.PointCloud(obj1.dc_centers_tf, obj2.dc_centers_tf)
        ot_res = linear.solve(geom_ott)
        # ot_res = linear.solve(geom_ott, max_iterations=3)

        matrix = jax.lax.stop_gradient(ot_res.matrix)
        cd_cen_dif = jnp.sum(
            (obj1.dc_centers_tf[:, None] - obj2.dc_centers_tf[None]) ** 2, axis=-1
        )
        dc_cen_cost = jnp.sum(matrix * cd_cen_dif, axis=(-1, -2))

        # z_cost = jnp.sum((obj1.z_flat[:, None] - obj2.z_flat[None]) ** 2, axis=-1)
        # z_cost = jnp.sum(matrix * z_cost, axis=(-1, -2))
        z_cost = jnp.sum((obj1.z_flat - obj2.z_flat) ** 2, axis=(-1,-2))

        pos_cost = jnp.sum((obj1.pos - obj2.pos) ** 2, axis=-1)

        return (
            self.dc_pos_loss_coef * dc_cen_cost
            + z_cost
            + self.pos_loss_coef * pos_cost
        )

    def tree_flatten(self):
        return (), (self.latent_shape, self.dc_pos_loss_coef, self.pos_loss_coef)


    @classmethod
    def tree_unflatten(cls, aux_data, children):
        del children
        return cls(*aux_data)


@jax.tree_util.register_pytree_node_class
class CFCost(costs.CostFn):
    """distance function for convex elements."""

    def __init__(self, latent_shape: typing.Tuple, dc_pos_loss_coef, pos_loss_coef, only_pos=False, dif_func='se'):
        super().__init__()
        self.latent_shape = latent_shape
        self.dc_pos_loss_coef = dc_pos_loss_coef
        self.pos_loss_coef = pos_loss_coef
        self.only_pos = only_pos
        self.dif_func = dif_func

    def pairwise(self, h1: jnp.ndarray, h2: jnp.ndarray) -> float:
        """ """
        obj1 = cxutil.LatentObjects().init_h(h1, self.latent_shape)
        obj2 = cxutil.LatentObjects().init_h(h2, self.latent_shape)

        if self.dif_func == 'huber':
            loss_func = partial(optax.huber_loss, delta=0.010)
        else:
            loss_func = lambda x, y: (x-y)**2

        pos_dif = jnp.sum(loss_func(obj1.pos, obj2.pos), axis=-1)
        cen_pwdif = jnp.sum(
            loss_func(obj1.dc_centers_tf[..., :, None, :],
                obj2.dc_centers_tf[..., None, :, :])
            ,
            axis=-1,
        )
        if obj1.z_flat.shape[-2] == 1 or obj2.z_flat.shape[-2]==1:
            pwdif = cen_pwdif
            z_dif = jnp.sum(
                    loss_func(obj1.z_flat, obj2.z_flat), axis=-1
                )
            z_dif = jnp.mean(z_dif, axis=-1)
            chloss = (
                self.dc_pos_loss_coef * jnp.mean(jnp.min(pwdif, axis=-1) + jnp.min(pwdif, axis=-2), axis=-1)
                + self.pos_loss_coef * pos_dif + z_dif
            )
        else:
            if self.only_pos:
                z_pwdif = 0
            else:
                z_pwdif = jnp.sum(
                    (obj1.z_flat[..., :, None, :] - obj2.z_flat[..., None, :, :]) ** 2, axis=-1
                )
            pwdif = z_pwdif + self.dc_pos_loss_coef * cen_pwdif
            chloss = (
                jnp.mean(jnp.min(pwdif, axis=-1) + jnp.min(pwdif, axis=-2), axis=-1)
                + self.pos_loss_coef * pos_dif
            )
        return chloss

    def tree_flatten(self):
        return (), (self.latent_shape, self.dc_pos_loss_coef, self.pos_loss_coef, self.only_pos)


    @classmethod
    def tree_unflatten(cls, aux_data, children):
        del children
        return cls(*aux_data)


if __name__ == "__main__":
    class Args:
        pass
    args = Args()
    args.pos_loss_coef = 1.0
    args.dc_pos_loss_coef = 1.0
    latent_shape = (32,16,8)
    df_cost = DFCost(latent_shape, args)
    na = 10

    jkey = jax.random.PRNGKey(0)
    h12 = jax.random.normal(jkey, shape=(2,na,latent_shape[0]*(latent_shape[1]*latent_shape[2]+3)+3))

    geom_ott = pointcloud.PointCloud(h12[0], h12[1], cost_fn=trutil.CFCost(latent_shape=latent_shape, args=args)) # Chamfer distance

    trutil.CFCost(latent_shape=latent_shape, args=args).all_pairs_pairwise(h12[0], h12[1])
    # cost_val = df_cost.pairwise(h12[0], h12[1])

    # self.cost_fn.all_pairs_pairwise(self.x, self.y)

    geom_ott
    print(1)
