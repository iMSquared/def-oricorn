import jax.numpy as jnp
import dm_pix as pix
import jax
from functools import partial 


def img_flatten(img):
    origin_shape = img.shape
    if img.ndim == 3:
        return img[None], origin_shape
    else:
        return img.reshape((-1,) + img.shape[-3:]), origin_shape

def gaussian_blur(jkey, img, apply_ratio = 0.5):
    img_, origin_shape = img_flatten(img)
    dtype_changed = False
    if img_.dtype == jnp.uint8:
        dtype_changed = True
        img_ = (img_/255.).astype(jnp.float32)
    sigma = 3 #@param {type: "slider", min: 0, max: 10}
    kernel_size = 3 #@param{type: "slider",min:0, max:10}
    sigma = jax.random.uniform(jkey, shape=(img_.shape[0],)) * 3
    _, jkey =jax.random.split(jkey)
    new_img = jax.vmap(partial(pix.gaussian_blur, kernel_size=kernel_size))(
        image=img_,
        sigma=sigma)
    new_img = new_img.reshape(origin_shape)
    if dtype_changed:
        new_img = (new_img*255).astype(jnp.uint8)

    mask = jax.random.uniform(jkey, shape=new_img[...,:1,:1,:1].shape) > apply_ratio
    _, jkey =jax.random.split(jkey)
    new_img = jnp.where(mask, new_img, img)

    return new_img


def translation(jkey, img, apply_ratio = 0.5):
    img_, origin_shape = img_flatten(img)
    gap = 2
    crop_size = (img_.shape[-3]-gap, img_.shape[-2]-gap)
    rgb_crop = pix.random_crop(
        key=jkey,
        image=img_,
        crop_sizes=(img_.shape[0], *crop_size, img.shape[-1]))
    rgb_res = jnp.zeros_like(img_)
    rgb_crop_list = []
    rgb_crop_list.append(rgb_res.at[...,1:1+crop_size[0], 0:0+crop_size[1],:].set(rgb_crop))
    rgb_crop_list.append(rgb_res.at[...,0:0+crop_size[0], 1:1+crop_size[1],:].set(rgb_crop))
    rgb_crop_list.append(rgb_res.at[...,2:2+crop_size[0], 1:1+crop_size[1],:].set(rgb_crop))
    rgb_crop_list.append(rgb_res.at[...,1:1+crop_size[0], 2:2+crop_size[1],:].set(rgb_crop))
    rgb_crop_list.append(rgb_res.at[...,1:1+crop_size[0], 1:1+crop_size[1],:].set(rgb_crop))
    rgb_crop_list = jnp.stack(rgb_crop_list, axis=1)
    rand_idx = jax.random.randint(jkey, shape=rgb_crop_list[...,:1,:1,:1,:1].shape, minval=0, maxval=rgb_crop_list.shape[1])
    _, jkey =jax.random.split(jkey)
    rgb_crop = jnp.take_along_axis(rgb_crop_list, rand_idx, axis=1)
    rgb_crop = jnp.squeeze(rgb_crop, axis=1)
    rgb_crop = rgb_crop.reshape(origin_shape)

    mask = jax.random.uniform(jkey, shape=rgb_crop[...,:1,:1,:1].shape) > apply_ratio
    _, jkey =jax.random.split(jkey)
    rgb_crop = jnp.where(mask, rgb_crop, img)

    return rgb_crop



if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import pickle
    import glob
    from pathlib import Path
    import sys
    BASEDIR = Path(__file__).parent.parent
    if str(BASEDIR) not in sys.path:
        sys.path.insert(0, str(BASEDIR))

    jkey = jax.random.PRNGKey(0)
    d_dirs = glob.glob('data/scene_data/*.pkl')
    with open(d_dirs[0], 'rb') as f:
        save_data = pickle.load(f)
    rgbs = save_data.rgbs[:10,0]

    rgb_tr = translation(jkey, rgbs)
    rgb_gb = gaussian_blur(jkey, rgbs)

    plt.figure()
    for i in range(4):
        plt.subplot(4,1,1+i)
        plt.imshow(rgb_tr[i])
    plt.show()

    plt.figure()
    for i in range(4):
        plt.subplot(4,1,1+i)
        plt.imshow(rgb_gb[i])
    plt.show()

