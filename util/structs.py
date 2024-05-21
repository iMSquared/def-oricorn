from flax import struct
import jax.numpy as jnp
import jax
import typing

@struct.dataclass
class ImgFeatures:
    intrinsic:jnp.ndarray # (... 6) [px, py, fx, fy, cx, cy]
    cam_posquat:jnp.ndarray # (... 7) pos-3 / quat-4
    img_feat:jnp.ndarray=None # (..., pi, pj, nf)
    img_feat_patch:jnp.ndarray=None
    img_state:jnp.ndarray=None # (..., pi, pj, nf)
    spatial_PE:jnp.ndarray=None # (..., pi, pj, nf)