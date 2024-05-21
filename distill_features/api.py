# Setup CLIP path
import sys
from pathlib import Path
BASEDIR = Path(__file__).parent
if str(BASEDIR) not in sys.path:
    sys.path.insert(0, str(BASEDIR))

import torch
import jax
import jax.numpy as jnp
from typing import List, Tuple
import numpy as np
import numpy.typing as npt

import torch
from einops import rearrange
from PIL import Image
from torchvision.transforms import CenterCrop, Compose
from tqdm import tqdm
import matplotlib.pyplot as plt

from distill_features.clip import clip
from distill_features.clip import tokenize
from distill_features.clip.model import CLIP
from distill_features.clip_extract import CLIPArgs

import util.camera_util as cutil
import util.cvx_util as cxutil


@torch.no_grad()
def get_clip_model(device: str = "cuda") -> Tuple[CLIP, Compose]:
    """Load clip model"""

    print(clip.available_models())

    model, preprocess = clip.load(CLIPArgs.model_name, device=device)
    model.eval()
    print(f"Loaded CLIP model {CLIPArgs.model_name}")

    # Patch the preprocess if we want to skip center crop
    if CLIPArgs.skip_center_crop:
        # Check there is exactly one center crop transform
        is_center_crop = [isinstance(t, CenterCrop) for t in preprocess.transforms]
        assert (
            sum(is_center_crop) == 1
        ), "There should be exactly one CenterCrop transform"
        # Create new preprocess without center crop
        preprocess = Compose(
            [t for t in preprocess.transforms if not isinstance(t, CenterCrop)]
        )
        print("Skipping center crop")

    return model, preprocess


@torch.no_grad()
def extract_clip_features(
        model: CLIP, 
        preprocess: Compose, 
        images: List[Image.Image], 
        device: str, 
        normalize: bool = True,
        patch_output: bool = True,
) -> torch.Tensor:
    """Extract dense patch-level CLIP features for given images"""

    # Preprocess the images
    preprocessed_images = torch.stack([preprocess(image) for image in images])
    preprocessed_images = preprocessed_images.to(device)  # (b, 3, h, w)

    # Get CLIP embeddings for the images


    if patch_output:
        embeddings = []
        for i in tqdm(
            range(0, len(preprocessed_images), CLIPArgs.batch_size),
            desc="Extracting CLIP features",
        ):
            batch = preprocessed_images[i : i + CLIPArgs.batch_size]
            embeddings.append(model.get_patch_encodings(batch))
        embeddings = torch.cat(embeddings, dim=0)

        # Reshape embeddings from flattened patches to patch height and width
        h_in, w_in = preprocessed_images.shape[-2:]
        if CLIPArgs.model_name.startswith("ViT"):
            h_out = h_in // model.visual.patch_size
            w_out = w_in // model.visual.patch_size
        elif CLIPArgs.model_name.startswith("RN"):
            h_out = max(h_in / w_in, 1.0) * model.visual.attnpool.spacial_dim
            w_out = max(w_in / h_in, 1.0) * model.visual.attnpool.spacial_dim
            h_out, w_out = int(h_out), int(w_out)
        else:
            raise ValueError(f"Unknown CLIP model name: {CLIPArgs.model_name}")
        embeddings = rearrange(embeddings, "b (h w) c -> b h w c", h=h_out, w=w_out)
    else:
        embeddings = []
        for i in tqdm(
            range(0, len(preprocessed_images), CLIPArgs.batch_size),
            desc="Extracting CLIP features",
        ):
            batch = preprocessed_images[i : i + CLIPArgs.batch_size]
            embeddings.append(model.encode_image(batch))
        embeddings = torch.cat(embeddings, dim=0)

    if normalize:
        embeddings /= embeddings.norm(dim=-1, keepdim=True)

    return embeddings


@torch.no_grad()
def get_cvx_clip_features(
        model: CLIP, 
        preprocess: Compose, 
        images: List[Image.Image], 
        device: str, 
        cam_intrinsics: np.ndarray,
        cam_posquats: np.ndarray,
        cvx_objs: cxutil.CvxObjects,
        debug: bool = False
) -> Tuple[npt.NDArray, npt.NDArray]:
    """Get clip features of each object convexs distilled from multi-view images.
    
    This is not jit compatible. USE ONLY NUMPY HERE.
    
    Args:
        ...

    Returns:
        obj_clip_embs_avgs (npt.NDArray): Distilled cvx clip feature. [#obj, #cvx, #feat]
        clip_embs (npt.NDArray): Raw clip feature of each image. [#view, H, W, #feat]
    """
    
    # Get patched CLIP feature
    with torch.no_grad():
        clip_embs = extract_clip_features(model, preprocess, images, device, normalize=True)
        # Upsampling
        clip_embs = clip_embs.permute(0, 3, 1, 2)
        clip_embs = torch.nn.functional.interpolate(clip_embs, np.asarray(images[0]).shape[0:2], mode="nearest")
        clip_embs = clip_embs.permute(0, 2, 3, 1)
        clip_embs = clip_embs.cpu().numpy()
        
    # Reproject centers
    objs_centers = cvx_objs.dc_centers_tf
    px_coord_ij, out_pnts = cutil.global_pnts_to_pixel(
        cam_intrinsics, cam_posquats, objs_centers, expand=True)
    px_coord_ij = np.array(px_coord_ij).astype(int)

    # Distill clip features (Ask chat gpt if one cannot understand advanced indexing)
    # Extracting the i and j indices from px_coord_ij
    i_indices = px_coord_ij[..., 0]         # Shape is (#objs, #views, 32)
    j_indices = px_coord_ij[..., 1]         # Shape is (#objs, #views, 32)
    rgbs_dim1_grid = np.arange(len(images))[None, :, None]  # Creating a grid for the first dimension of 'rgbs' that matches the second dimension of 'px_coord_ij'. Shape becomes (1, #views, 1) for broadcasting.
    num_feat = clip_embs.shape[-1]
    out_pnts_expanded = out_pnts[..., None]                                         # Shape is (#objs, #views, 32, 1)
    out_pnts_broadcasted = np.repeat(~out_pnts_expanded, repeats=num_feat, axis=-1) # Shape is (#objs, #views, 32, #feat)

    # Draw projected points in each views
    if debug:
        np_images = np.stack([np.asarray(img) for img in images], axis=0)
        # Object coloring
        for k in range(len(objs_centers)):
            color = np.random.randint(0, 256, size=3, dtype=np.uint8)
            # Using boolean indexing to update np_images only where out_pnts is False
            mask = np.zeros_like(np_images, dtype=bool)
            obj_i_indices = i_indices[k][None, ...]
            obj_j_indices = j_indices[k][None, ...]
            # Only color in-points
            mask[rgbs_dim1_grid, obj_i_indices, obj_j_indices, :] = ~out_pnts_expanded[k, None, ...]
            np_images = np.where(mask, color, np_images)

        # Plot    
        debug_imgs = [Image.fromarray(np_img) for np_img in np_images]
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        for l, (ax, img) in enumerate(zip(axs, debug_imgs)):
            ax.imshow(img)
            ax.set_title(f"Camera {l}")
        plt.tight_layout()
        plt.show()

    # Distillation
    obj_clip_embs_views = clip_embs[rgbs_dim1_grid, i_indices, j_indices, :]            # Shape is (#objs, #views, 32, 128)
    obj_clip_embs_views = np.where(out_pnts_broadcasted, obj_clip_embs_views, np.inf)   # Use masks to trigger inf when incorrect out_pnts are involved to average
    obj_clip_embs_avgs = obj_clip_embs_views.mean(axis=1, where=~out_pnts_expanded)     # Shape is (#objs, 32, 128)

    return obj_clip_embs_avgs, clip_embs

@torch.no_grad()
def get_text_embeddings(model: CLIP, texts: List[str], device: str, normalize: bool = True):
    """Get text embeddings"""
    tokens = tokenize(texts).to(device)
    text_embs = model.encode_text(tokens)
    if normalize:
        text_embs /= text_embs.norm(dim=-1, keepdim=True)
    # text_embs = text_embs.cpu().numpy()
    torch.cuda.synchronize()
    return text_embs


def get_normalized_similarities(clip_embs: npt.NDArray, text_embs: npt.NDArray):
    """"""
    sims = clip_embs @ text_embs.T
    
    # Keep text dim
    axes = tuple(range(len(clip_embs.shape)-1))
    mins = np.min(sims, axis=axes)
    maxs = np.max(sims, axis=axes)

    sims_norm = (sims - mins) / (maxs - mins)
    return sims_norm