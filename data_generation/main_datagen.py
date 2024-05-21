import os
os.environ["JAX_PLATFORMS"] = "cpu"
# os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"   
# os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"    # NVISII will crash when showed multiple devices with jax.

import numpy as np
import pickle
import argparse
import ray
import jax
import flax
import nvisii
from pathlib import Path
from tqdm import tqdm

# Setup import path
import sys
BASEDIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASEDIR))

import util.io_util as ioutil
import data_generation.render_with_nvisii as rwn
import data_generation.scene_generation as sg

# Typing
from typing import Tuple, List



def main(args: argparse.Namespace):
    
    # Env config
    if args.visualize_for_debug:
        PIXEL_SIZE = (200,300)
        BATCH_SIZE = 4
    else:
        PIXEL_SIZE = ([int(v) for v in args.pixel_size.split("-")])
        BATCH_SIZE = 50
    NUM_VIEWS = args.num_views
    NUM_OBJ = args.num_objs
    NUM_ITERATIONS: int = args.num_iterations
    RAY_RESET_INTERVAL: int = args.ray_reset_interval
    NUM_RAY_ENVS: int = args.num_ray_envs
    SCENE_TYPE: str = args.scene_type
    USE_NVISII: bool = args.use_nvisii
    if args.visualize_for_debug:
        NO_RGB: bool = False
    else:
        NO_RGB: bool = USE_NVISII
    ADD_DISTRACTOR: bool = args.add_distractor
    # IO config
    HDR_DIR = Path(args.hdr_dir)
    TEXTURE_DIR = Path(args.texture_dir)
    DGN_BUILD_METADATA = ioutil.BuildMetadata.from_str(args.DGN_build)
    # if args.category == "cherrypick":
    #     DGN_OBJ_LIST = ioutil.get_filepath_from_annotation(BASEDIR/'data/annotations.csv', BASEDIR/'data/DexGraspNet/32_64_1_v4_textured')
    # elif args.category == "cherrypick2":
    #     DGN_OBJ_LIST = ioutil.get_filepath_from_annotation(BASEDIR/'data/annotations2.csv', BASEDIR/'data/DexGraspNet/32_64_1_v4_textured')
    # elif args.category == "none":
    #     DGN_OBJ_LIST = [str(fn) for fn in DGN_BUILD_PATH.glob("*.obj")]
    # else:
    #     DGN_OBJ_LIST = []
    #     for cat in args.category.split("__"):
    #         DGN_OBJ_LIST = DGN_OBJ_LIST + [str(fn) for fn in DGN_BUILD_PATH.glob(f"*{cat}*.obj")]
    SAVE_DIR = Path(args.save_dir)
    if args.validation:
        SAVE_FILE_PATH = SAVE_DIR/f"val_{ioutil.get_current_timestamp()}_{args.camera_type}_{args.scene_type}_{args.dataset}_{NUM_VIEWS}_{NUM_OBJ}_{DGN_BUILD_METADATA.max_dec_num}_{args.category}.npz"
    else:
        SAVE_FILE_PATH = SAVE_DIR/f"{ioutil.get_current_timestamp()}_{args.camera_type}_{args.scene_type}_{args.dataset}_{NUM_VIEWS}_{NUM_OBJ}_{DGN_BUILD_METADATA.max_dec_num}_{args.category}.npz"

    # DEBUG: Serial data generation
    # datapoints_list: List[sg.SceneCls.SceneData] = []
    # for i in tqdm(range(NUM_ITERATIONS)):
    #     scene = sg.SceneCls(
    #         object_set_dir_list = DGN_OBJ_LIST, 
    #         max_obj_no = NUM_OBJ, 
    #         pixel_size = PIXEL_SIZE, 
    #         scene_type = SCENE_TYPE, 
    #         no_rgb = NO_RGB, 
    #         gui = False, 
    #         robot_gen = True
    #     )
    #     datapoint = scene.gen_batch_dataset(BATCH_SIZE, NUM_VIEWS)
    #     datapoints_list.append(datapoint)

    # Parallel data generation
    datapoints_list: List[sg.SceneCls.SceneData] = []
    for i in tqdm(range(NUM_ITERATIONS)):
        if i % RAY_RESET_INTERVAL == 0:
            try:
                ray.shutdown()
            except:
                pass
            ray_actors = [ray.remote(sg.SceneCls).remote(
                    # object_set_dir_list = DGN_OBJ_LIST, 
                    dataset = args.dataset,
                    camera_type=args.camera_type,
                    max_obj_no = NUM_OBJ, 
                    pixel_size = PIXEL_SIZE, 
                    scene_type = SCENE_TYPE, 
                    no_rgb = NO_RGB, 
                    gui = False, 
                    robot_gen = False,
                    validation=args.validation==1,
                ) for _ in range(NUM_RAY_ENVS)]
        imgs_list = ray.get([ra.gen_batch_dataset.remote(BATCH_SIZE, NUM_VIEWS) for ra in ray_actors])
        datapoints = jax.tree_util.tree_map(lambda *x: np.concatenate(x, 0), *imgs_list)
        datapoints_list.append(datapoints)
    ray.shutdown()

    # Aggregate generated data
    batched_datapoints: sg.SceneCls.SceneData = jax.tree_util.tree_map(lambda *x: np.concatenate(x, 0), *datapoints_list)
    num_generated = batched_datapoints.rgbs.shape[0]

    # Save without image first.
    if not args.visualize_for_debug:
        SAVE_DIR.mkdir(exist_ok=True)
        with SAVE_FILE_PATH.open('wb') as f:
            numpy_datapoints = flax.serialization.to_state_dict(batched_datapoints)
            np.savez_compressed(f, item=numpy_datapoints)
    print(f'File initalized without rgbs.')

    # Convert to nvisii
    if USE_NVISII:
        print("Re-rendering using NVISII")
        nvisii_ren = rwn.NvisiiRender(
            pixel_size = PIXEL_SIZE, 
            hdr_dir = str(HDR_DIR),
            texture_dir = str(TEXTURE_DIR))
        
        # Naive iteration...
        nvisii_rgbs = np.zeros((num_generated, NUM_VIEWS, nvisii_ren.option.height, nvisii_ren.option.width, 3), dtype=np.uint8)
        if args.visualize_for_debug:
            origin_rgbs = batched_datapoints.rgbs
        for i in tqdm(range(num_generated)):
            # Render a datapoint
            datapoint = jax.tree_map(lambda x: x[i], batched_datapoints)    # Take one datapoint from batch
            datapoint_rgbs = nvisii_ren.get_rgb_for_datapoint(datapoint, add_distractor=ADD_DISTRACTOR)
            if args.visualize_for_debug:
                # Debug: visualize
                import matplotlib.pyplot as plt
                plt.figure(figsize=(15,10))
                for j, img_ in enumerate(datapoint_rgbs):
                    plt.subplot(2,len(datapoint_rgbs),j+1)
                    plt.imshow(img_)
                    plt.axis('off')
                    plt.subplot(2,len(datapoint_rgbs),j+len(datapoint_rgbs)+1)
                    plt.imshow(origin_rgbs[i,j])
                    plt.axis('off')
                plt.tight_layout()
                plt.show()
            # Update corresponding datapoints
            nvisii_rgbs[i] = datapoint_rgbs
            batched_datapoints = batched_datapoints.replace(rgbs=nvisii_rgbs)
            
            # if i % 5 == 0 or i == num_generated-1:
            if i == num_generated-1:
                # Update nvisii images...
                if not args.visualize_for_debug:
                    SAVE_DIR.mkdir(exist_ok=True)
                    with SAVE_FILE_PATH.open('wb') as f:
                        numpy_datapoints = flax.serialization.to_state_dict(batched_datapoints.replace(nvren_info=None, table_params=None, robot_params=None))
                        np.savez_compressed(f, item=numpy_datapoints)
                print(f'File updated at iter={i}. Time: {ioutil.get_current_timestamp()}')

        nvisii.deinitialize()
    else:
        if not args.visualize_for_debug:
            SAVE_DIR.mkdir(exist_ok=True)
            with SAVE_FILE_PATH.open('wb') as f:
                numpy_datapoints = flax.serialization.to_state_dict(batched_datapoints.replace(nvren_info=None, table_params=None, robot_params=None))
                np.savez_compressed(f, item=numpy_datapoints)
        else:
            for i in tqdm(range(num_generated)):
                # Render a datapoint
                datapoint = jax.tree_map(lambda x: x[i], batched_datapoints)
                datapoint_rgbs = datapoint.rgbs
                import matplotlib.pyplot as plt
                plt.figure(figsize=(15,10))
                for j, img_ in enumerate(datapoint_rgbs):
                    plt.subplot(1,len(datapoint_rgbs),j+1)
                    plt.imshow(img_)
                plt.tight_layout()
                plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_views', type=int, default=5)
    parser.add_argument('--num_objs', type=int, default=7)
    parser.add_argument('--save_dir', type=str, default='data/scene_data')
    parser.add_argument('--hdr_dir', type=str, default='data/hdr')
    parser.add_argument('--texture_dir', type=str, default="data/texture")
    parser.add_argument('--DGN_build', type=str, default="32_64_1_v4_textured", help="{dec_no}_{vert_no}_{vol_err_tol}_v4 ")
    parser.add_argument('--pixel_size', type=str, default="64-112", help="Training image size {height-width}")
    parser.add_argument('--num_iterations', type=int, default=60)
    parser.add_argument('--ray_reset_interval', type=int, default=10)
    parser.add_argument('--num_ray_envs', type=int, default=10)
    parser.add_argument('--use_nvisii', type=int, default=1)
    parser.add_argument('--scene_type', type=str, default="table", help="['shelf', 'flat', ...]")
    parser.add_argument('--add_distractor', type=int, default=0)
    parser.add_argument('--visualize_for_debug', type=int, default=0)
    parser.add_argument('--category', type=str, default='none')
    parser.add_argument('--dataset', type=str, default='NOCS')
    parser.add_argument('--validation', type=int, default=0)
    parser.add_argument('--camera_type', type=str, default='d435')
    args = parser.parse_args()
    main(args)



