import copy
import numpy as np
from scipy.spatial.transform import Rotation as R
from PIL import Image
import matplotlib.pyplot as plt
try:
    import pyrealsense2 as rs
    from dt_apriltags import Detector
except:
    print('turn off real sensors')

import time

import util.camera_util as cutil


IMG_TANKING_Q = np.array([ 0.0998, -1.3284, -0.1451, -2.0762,  0.0092,  1.7475,  0.7789], dtype=np.float32)
IMG_TAKING_POS = np.array([ 0.1430, -0.0528,  0.7824], dtype=np.float32)
IMG_TAKING_QUAT = np.array([ 0.8092, -0.3363,  0.4083, -0.2557], dtype=np.float32)
RRT_INITIAL_Q = np.array([-0.0468, -1.0992, -0.0301, -2.3754, -0.0116,  1.6467,  0.7794], dtype=np.float32)
START_Q = np.array([-0.1874, -0.0380, -0.3286, -1.7475,  0.0504,  2.1404,  1.5653], dtype=np.float32)
END_Q = np.array([-1.2489, -0.3862,  1.5234, -1.8407,  0.0857,  2.7670,  0.3529], dtype=np.float32)

GRASP_INIT_Q = np.array([ 0.0372, -1.1527, -0.0738, -2.6399,  0.0314,  1.8934,  0.8879], dtype=np.float32)
GRASP_INIT_POS = np.array([ 0.2846, -0.0033,  0.4922], dtype=np.float32)
GRASP_INIT_QUAT = np.array([ 0.8813, -0.4249,  0.1602, -0.1308], dtype=np.float32)

GRASP_MID_Q = np.array([ 5.0333e-04, -7.7841e-03,  5.5246e-02, -1.9859e+00,  6.1033e-02, 1.9155e+00,  8.0344e-01], dtype=np.float32)
GRASP_MID_POS = np.array([0.5387, 0.0403, 0.3580], dtype=np.float32)
GRASP_MID_QUAT = np.array([ 0.9991,  0.0076, -0.0323, -0.0270], dtype=np.float32)

# POS_EC = R.from_euler('z',-np.pi/4).apply(np.array([0.05737, 0.0586, -0.07255]))
# POS_EC = R.from_euler('z',-np.pi/4).apply(np.array([0.05737, -0.0325+0.0285, -0.07255]))
# POS_EC = R.from_euler('z',-np.pi/4).apply(np.array([0.05737, -0.0325+0.0285, -0.07255]))
# POS_EC = R.from_euler('z',-np.pi/4).apply(np.array([0.055, -0.0325+0.0285 - 0.006, -0.07255]))
# POS_EC = R.from_euler('z',-np.pi/4).apply(np.array([0.055, -0.0325+0.0285, -0.07255 - 0.012]))
POS_EC = R.from_euler('z',-np.pi/4).apply(np.array([0.05737+0.010, -0.0325+0.0285-0.012, -0.07255 - 0.012])).astype(np.float32)
# POS_EC = R.from_euler('z',-np.pi/4).apply(np.array([0.07738, -3.713e-4, -6.051e-2]))
QUAT_EC = (R.from_euler('z', np.pi/2.0-np.pi/4) * R.from_euler('y', np.pi)).as_quat().astype(np.float32)

def way_points_to_trajectory_np_two_traj(waypnts, resolution):
    traj1, traj2 = np.split(waypnts, 2, 0)
    traj1 = way_points_to_trajectory_np(traj1, resolution//2)
    traj2 = way_points_to_trajectory_np(traj2, resolution//2)
    return np.concatenate([traj1, traj2], axis=0)

def way_points_to_trajectory_np(waypnts, resolution):
    """???"""

    # # goal reaching traj
    # goal = waypnts[-1]
    # traj_norm = np.linalg.norm(goal - waypnts, axis=-1)
    # traj_min_idx = np.min(np.arange(traj_norm.shape[0], dtype=np.int32)*(traj_norm<0.04).astype(np.int32) + 100000*(traj_norm>=0.04).astype(np.int32))
    # waypnts = waypnts[:traj_min_idx+1]
    # waypnts = np.concatenate([waypnts, goal[None]], axis=0)

    wp_len = np.linalg.norm(waypnts[1:] - waypnts[:-1], axis=-1)
    wp_len = wp_len/np.sum(wp_len).clip(1e-5)
    wp_len = np.where(wp_len<1e-5, 0, wp_len)
    wp_len = wp_len/np.sum(wp_len)
    wp_len_cumsum = np.cumsum(wp_len)
    wp_len_cumsum = np.concatenate([np.array([0]),wp_len_cumsum], 0)
    wp_len_cumsum[-1] = 1.0
    indicator = np.linspace(0, 1, resolution)
    indicator = (-np.cos(indicator*np.pi)+1)/2.
    included_idx = np.sum(indicator[...,None] > wp_len_cumsum[1:], axis=-1)
    included_idx = included_idx.clip(0, included_idx.shape[0]-2)

    upper_residual = (wp_len_cumsum[included_idx+1] - indicator)/wp_len[included_idx].clip(1e-5)
    upper_residual = np.where(wp_len[included_idx] < 1e-5, 1, upper_residual)
    upper_residual = upper_residual.clip(0.,1.)
    bottom_residual = 1.-upper_residual

    traj = waypnts[included_idx] * upper_residual[...,None] + waypnts[included_idx+1] * bottom_residual[...,None]
    traj = np.where(wp_len[included_idx][...,None] < 1e-5, waypnts[included_idx], traj)
    traj[0] = waypnts[0]
    traj[-1] = waypnts[-1]

    # check
    dif_len = np.linalg.norm(traj[1:] - traj[:-1], axis=-1)
    dif_len_max = np.max(dif_len)
    if dif_len_max > 0.1:
        print(dif_len_max)
        print(resolution)
        print(wp_len)
        mask = dif_len > 0.1
        mask = np.concatenate([[True], mask], 0)
        traj = traj[mask]


    # assert dif_len_max < 0.1, f'{dif_len_max}'
    
    return traj


def interval_based_interpolations_np(waypnts, gap):
    wp_len = np.linalg.norm(waypnts[1:] - waypnts[:-1], axis=-1)
    entire_len = np.sum(wp_len)
    assert entire_len > gap
    int_no = int(entire_len/gap)+1
    return way_points_to_trajectory_np(waypnts, int_no)

class TimeTracker(object):
    
    def __init__(self):
        self.t = {}
        self.dt = {}
    
    def set(self, name):
        if name in self.t:
            self.dt[name] = time.time()-self.t[name]
            # self.print(name)
            self.t.pop(name)
        else:
            self.t[name] = time.time()
    
    def str_dt_all(self):
        str_res = ''
        for k in self.dt:
            str_res = str_res + f'{k} time: {self.dt[k]} //'
        return str_res
    
    def print_all(self):
        for k in self.dt:
            print(k + f' time: {self.dt[k]}')

    def get_dt(self):
        return copy.deepcopy(self.dt)

    def print(self, name):
        print(name + f' time: {time.time()-self.t[name]}')

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

def matrix_from_R_T(R, T):
    T = np.reshape(T, (3))
    matrix = np.zeros((4, 4))
    matrix[:3, :3] = R
    matrix[:3, 3] = T
    matrix[3, 3] = 1.
    return matrix

def inverse_matrix_from_R_T(R, T):
    R_t = np.transpose(R)
    T_new = -1.*np.matmul(R_t, T)
    return matrix_from_R_T(R_t, T_new)

def get_camera_pose(at_detector, rgb, cam_intrinsic, tag_size=0.1591, tag_id=0, debug=False, depth=None):

    gray = rgb2gray(rgb)

    cam_param = list(cam_intrinsic[2:])
    # cam_param[-2] = cam_intrinsic[0] - cam_intrinsic[-2]
    # cam_param[-1] = cam_intrinsic[1] - cam_intrinsic[-1] # y bias should be flipped from realsense camera parameters!

    tags = at_detector.detect(gray.astype(np.uint8), True, cam_param, tag_size) # 0.1655
    tags_dict = {tag.tag_id: (tag.pose_R, tag.pose_t) for tag in tags}

    
    if len(tags_dict) == 0:
        return None
    r_cam_to_tag = tags_dict[tag_id][0]
    t_cam_to_tag = tags_dict[tag_id][1]
    r_cam_to_tag =  R.from_matrix(r_cam_to_tag) # * R.from_euler('y', np.pi)
    mat_cam = inverse_matrix_from_R_T(r_cam_to_tag.as_matrix(), t_cam_to_tag)

    offset_H = np.eye(4)
    offset_H[:3,:3] = R.from_euler('x', np.pi).as_matrix()
    mat_cam = offset_H@mat_cam
    Rm_ = mat_cam[:3,:3]
    mat_cam[:3,:3] = (R.from_matrix(Rm_) * R.from_euler('x', np.pi)).as_matrix()

    pos = mat_cam[:3, 3]
    rot = mat_cam[:3, :3]
    r = R.from_matrix(rot)

    modified_rot = r
    modified_pos = pos

    quat = modified_rot.as_quat()
    print(modified_pos)
    modified_mat = np.zeros((3, 4)) 
    modified_mat[:3, :3] = modified_rot.as_matrix()
    modified_mat[:3, 3] = modified_pos

    if debug:
        corners = tags[0].corners.astype(np.int32)
        centers = tags[0].center.astype(np.int32)[None]
        corners = np.concatenate([corners, centers], axis=0)
        gray_paint = copy.deepcopy(gray)
        rgb_paint = copy.deepcopy(rgb)

        ext = np.random.normal(size=(2000,3))
        around_pnts = (ext / np.linalg.norm(ext, axis=-1, keepdims=True))[...,-2:]
        around_pnts = around_pnts*6
        around_pnts = around_pnts.astype(np.int32)

        for cn in corners:
            for around in around_pnts:
                gray_paint[cn[1]+around[0], cn[0]+around[1]] = 0.5
                rgb_paint[cn[1]+around[0], cn[0]+around[1]] = np.array([255,0,0])

        cam_posquat = np.concatenate([modified_pos, quat], 0)
        rgb_reproduced, rgb_reproduced_resized = reproject_tag_coordinates(rgb[None], cam_posquat[None], cam_intrinsic[None], tag_size)
        
        # plt.figure()
        # plt.subplot(2,2,1)
        # plt.imshow(gray_paint[...,None])
        # plt.axis('off')
        # plt.subplot(2,2,2)
        # plt.imshow(rgb_paint)
        # plt.axis('off')
        # plt.subplot(2,2,3)
        # plt.imshow(rgb_reproduced)
        # plt.axis('off')
        # plt.subplot(2,2,4)
        # plt.imshow(rgb_reproduced_resized)
        # plt.axis('off')
        # plt.show()

        depth = np.where(depth > 2.0, 0, depth)
        cutil.visualize_pcd(rgb[None], depth[None], cam_posquat[None], cam_intrinsic[None], return_elements=False, tag_size=tag_size, area=np.array([[-0.25,-0.25,-0.1],[0.25,0.25,0.1]]))

    return modified_pos, quat, modified_mat


def flip_initrinsic(mask, intrinsic):
    if mask:
        intrinsic_fliped = copy.deepcopy(intrinsic)
        intrinsic_fliped[-2:] = -intrinsic_fliped[-2:]
        return intrinsic_fliped
    else:
        return intrinsic

def flip_img(mask, img):
    if mask:
        return img[::-1,::-1]
    else:
        return img


def reproject_tag_coordinates(colors, cam_posquat, cam_intrinsic, tag_size):
    '''
    test compatibility with our camera tools

    colors, cam_posquat, cam_intrinsic should have same batch size
    colors (NB NI NJ 3)
    cam_posquat (NB 7)
    cam_intrinsic (NB 6)
    '''
    original_pixel_size = colors.shape[-3:-1]
    output_pixel_size = (64, 112)
    colors_resized = cutil.resize_img(colors, output_pixel_size)
    colors_resized = np.array(colors_resized)
    intrinsic_resized = cutil.resize_intrinsic(cam_intrinsic, original_pixel_size, output_pixel_size)

    test_pnts = np.array([[0.,0.,0.],
                     [tag_size/2., tag_size/2., 0.],
                     [-tag_size/2., tag_size/2., 0.],
                     [tag_size/2., -tag_size/2., 0.],
                     [-tag_size/2., -tag_size/2., 0.],
                     [tag_size/2., 0., 0.],
                     [-tag_size/2., 0., 0.],
                     [0., tag_size/2., 0.],
                     [0., -tag_size/2., 0.],])

    pixel_pnts_continuous, _  = cutil.global_pnts_to_pixel(cam_intrinsic, cam_posquat, test_pnts, expand=True)
    pixel_pnts = np.floor(pixel_pnts_continuous).astype(np.int32)

    ext = np.random.normal(size=(2000,3))
    around_pnts = (ext / np.linalg.norm(ext, axis=-1, keepdims=True))[...,-2:]
    around_pnts = around_pnts*6
    around_pnts = around_pnts.astype(np.int32)

    colors_painted = copy.deepcopy(colors)
    for i, color_ in enumerate(colors_painted):
        for around in around_pnts:
            color_[pixel_pnts[i,:,0]+around[0], pixel_pnts[i,:,1]+around[1]] = np.array([255,0,0])
    
    pixel_pnts_continuous_resize, _  = cutil.global_pnts_to_pixel(intrinsic_resized, cam_posquat, test_pnts, expand=True)
    pixel_pnts_resize = np.floor(pixel_pnts_continuous_resize).astype(np.int32)

    colors_painted_resize = copy.deepcopy(colors_resized)
    for i, color_ in enumerate(colors_painted_resize):
        color_[pixel_pnts_resize[i,:,0], pixel_pnts_resize[i,:,1]] = np.array([255,0,0])

    return colors_painted[0], colors_painted_resize[0]

class RSCam(object):
    def __init__(self, sn_list, flip_mask=[True, False, False], pixel_size=(240,424), padding_no=None):
        self.sn_list = sn_list
        self.pixel_size = pixel_size
        self.fps = 30
        self.flip_mask = flip_mask
        self.padding_no = padding_no
        self.tag_size = 0.1591
        # self.tag_size = 0.1600
        self.init_cam()


    def init_cam(self):
        align_to = rs.stream.color
        self.align = rs.align(align_to)
        self.pipelines = []
        self.cam_intrinsic = []
        for flip_mask_, sn in zip(self.flip_mask, self.sn_list):
            pipeline = rs.pipeline()
            config = rs.config()
            config.enable_device(sn)
            config.enable_stream(rs.stream.depth, self.pixel_size[1], self.pixel_size[0], rs.format.z16, self.fps)
            config.enable_stream(rs.stream.color, self.pixel_size[1], self.pixel_size[0], rs.format.bgr8, self.fps)
            cfg = pipeline.start(config)
            self.pipelines.append(pipeline)
            self.cam_intrinsic.append(self.get_intrinsic(cfg, flip_mask_))
        if self.padding_no is not None:
            for _ in range(self.padding_no - len(self.sn_list)):
                self.cam_intrinsic.append(self.cam_intrinsic[0])
        self.cam_intrinsic = np.stack(self.cam_intrinsic)

        for i, (flip_mask_, pl) in enumerate(zip(self.flip_mask, self.pipelines)):
            print(f'cam {i}')
            frames = pl.wait_for_frames()
            frames = self.align.process(frames)
            color_frame = frames.get_color_frame()
            depth_image = flip_img(flip_mask_, np.array(frames.get_depth_frame().get_data())/1000.)
            color_image = flip_img(flip_mask_, np.asanyarray(color_frame.get_data())[...,::-1])
            # im = Image.fromarray(color_image)
            # im.save('1color.png')

    def get_intrinsic(self, cfg, flip_mask):
        profile = cfg.get_stream(rs.stream.color) # Fetch stream profile for depth stream
        intr = profile.as_video_stream_profile().get_intrinsics() # Downcast to video_stream_profile and fetch intrinsics
        print("intrinsic: ", intr)
        if flip_mask:
            camera_param = (intr.fx, intr.fy, self.pixel_size[1]-intr.ppx, self.pixel_size[0]-intr.ppy)
        else:
            camera_param = (intr.fx, intr.fy, intr.ppx, intr.ppy)
        cam_intrinsic = [self.pixel_size[1], self.pixel_size[0], *camera_param]
        return cam_intrinsic

    def get_rgb_depth(self):
        OG_RGB_arr = []
        depth_arr = []
        for flip_mask_, pl in zip(self.flip_mask, self.pipelines):
            frames = pl.wait_for_frames()
            frames = self.align.process(frames)
            color_frame = frames.get_color_frame()
            depth_image = flip_img(flip_mask_, np.array(frames.get_depth_frame().get_data())/1000.)
            color_image = flip_img(flip_mask_, np.asanyarray(color_frame.get_data())[...,::-1])
            OG_RGB_arr.append(color_image)
            depth_arr.append(depth_image)
        if self.padding_no is not None:
            for _ in range(self.padding_no - len(self.pipelines)):
                OG_RGB_arr.append(np.zeros_like(OG_RGB_arr[0]))
                depth_arr.append(np.zeros_like(depth_arr[0]))
        OG_RGB_arr = np.stack(OG_RGB_arr)
        depth_arr = np.stack(depth_arr)
        return OG_RGB_arr, depth_arr

    def calibrate_extrinsic(self, debug=False):
        at_detector = Detector(families='tag36h11')
        for _ in range(20):
            rgb, depth = self.get_rgb_depth()

        cam_posquats = []
        for i in range(len(self.sn_list)):
            pos_, quat_, _ = get_camera_pose(at_detector, rgb[i], self.cam_intrinsic[i], tag_size=self.tag_size, debug=debug, depth=depth[i])
            cam_posquats.append(np.concatenate([pos_, quat_], axis=-1))
        return np.stack(cam_posquats)
    

    def get_rgb_depth_holefilling(self):
        OG_RGB_arr = []
        depth_arr = []
        for flip_mask_, pl in zip(self.flip_mask, self.pipelines):
            frames = pl.wait_for_frames()
            frames = self.align.process(frames)
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            hole_filling = rs.hole_filling_filter()
            filled_depth = hole_filling.process(depth_frame)
            depth_image = flip_img(flip_mask_, np.array(filled_depth.get_data())/1000.)
            color_image = flip_img(flip_mask_, np.asanyarray(color_frame.get_data())[...,::-1])
            OG_RGB_arr.append(color_image)
            depth_arr.append(depth_image)
        OG_RGB_arr = np.stack(OG_RGB_arr)
        depth_arr = np.stack(depth_arr)
        return OG_RGB_arr, depth_arr