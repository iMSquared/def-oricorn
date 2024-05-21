from typing import Tuple
from PIL import Image
import pickle
from pathlib import Path
import matplotlib.pyplot as plt

import jax
import numpy as np
import numpy.typing as npt
import cv2
from scipy.spatial.transform import Rotation


import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"


def resize(scale: float, rgb: npt.NDArray, intrinsics: npt.NDArray) -> Tuple[npt.NDArray, npt.NDArray]: 

    original_h = rgb.shape[0]
    original_w = rgb.shape[1]
    new_h = int(original_h * scale)
    new_w = int(original_w * scale)

    rgb_resized = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
    new_intrinsics = np.copy(intrinsics)
    new_intrinsics[0,:] *= scale
    new_intrinsics[1,:] *= scale

    return rgb_resized, new_intrinsics



def crop_from_center(rgb: npt.NDArray, new_w: int, new_h: int, intrinsics: npt.NDArray) -> Tuple[npt.NDArray, npt.NDArray]:

    original_h = rgb.shape[0]
    original_w = rgb.shape[1]

    if not (original_w >= new_w and original_h >= new_h):
        raise ValueError("Target crop size is larger than original size.")

    center_h = intrinsics[1,2]
    center_w = intrinsics[0,2]

    h_start = int(center_h - new_h/2)
    h_end   = int(center_h + new_h/2)
    assert (h_end-h_start == new_h)
    w_start = int(center_w - new_w/2)
    w_end   = int(center_w + new_w/2)
    assert (w_end-w_start == new_w)

    h_ratio = new_h/original_h
    w_ratio = new_w/original_w

    rgb_cropped = rgb[h_start:h_end, w_start:w_end]
    new_intrinsics = np.copy(intrinsics)
    new_intrinsics[0,2] *= w_ratio
    new_intrinsics[1,2] *= h_ratio

    return rgb_cropped, new_intrinsics



def change_focal_length(image, original_intrinsics, original_dist_coeffs, new_f_x, new_f_y):

    # Get the shape of the undistorted image
    height, width, _ = image.shape
    # Create a new camera matrix with the desired new focal length
    new_intrinsics = np.array([[new_f_x, 0, width / 2], [0, new_f_y, height / 2], [0, 0, 1]], dtype=np.float32)
    # Undistort the image using the original intriniscs
    new_image = cv2.undistort(image, original_intrinsics, original_dist_coeffs, newCameraMatrix=new_intrinsics)

    return new_image, new_intrinsics


def convert_gl_view_to_cv_extrinsics(view_matrix: npt.NDArray):

    gl_cam_pose = np.linalg.inv(view_matrix.T)
    local_T = np.array([[1,  0,  0,  0],
                        [0, -1,  0,  0],
                        [0,  0, -1,  0],
                        [0,  0,  0,  1]])

    cv2_cam_pose = gl_cam_pose@local_T
    cv2_extrinsics = np.linalg.inv(cv2_cam_pose)

    return cv2_extrinsics





def main():

    cur_dir = Path(__file__).parent.resolve()
    with open(cur_dir/"real_data"/"2_cylinder.pkl", 'rb') as f:
        save_dict = pickle.load(f)

    print(save_dict)

    for i in range(3):
        # Read RGB
        bgr = save_dict['rgb'][i]
        bgr = np.array(bgr).astype(np.uint8)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        # Read intrinsic
        pixel_size_x, pixel_size_y, fx, fy, cx, cy = save_dict['cam_intrinsic'][i]
        intrinsics = np.array([[fx,   0,  cx],
                               [ 0,  fy,  cy],
                               [ 0,   0,   1]])
        # Read extrinsic
        pos = save_dict['cam_posquat'][i][:3]
        orn = Rotation.from_quat(save_dict['cam_posquat'][i][3:]).as_matrix()
        gl_cam_pose = np.eye(4)
        gl_cam_pose[:3,:3] = orn
        gl_cam_pose[:3, 3] = pos
        gl_view_matrix = np.linalg.inv(gl_cam_pose).T
        cv2_extrinsics = convert_gl_view_to_cv_extrinsics(gl_view_matrix)

        # Convert GL view
        cv2_cam_pose = np.linalg.inv(cv2_extrinsics)
        pos = cv2_cam_pose[:3,3]
        orn = Rotation.from_matrix(cv2_cam_pose[:3,:3]).as_quat()

        # Resize and crop
        rgb, intrinsics = resize(2, rgb, intrinsics)
        rgb, intrinsics = crop_from_center(rgb, 640, 480, intrinsics)
        rgb, intrinsics = change_focal_length(
            image = rgb,
            original_intrinsics  = intrinsics,
            original_dist_coeffs = np.zeros((4,1)),
            new_f_x = 617.159,
            new_f_y = 617.159)

        # Create a figure
        plt.imshow(rgb)
        plt.show()

        print(np.linalg.inv(gl_view_matrix.T))
        print(intrinsics)
        print()



if __name__=="__main__":
    main()

        

