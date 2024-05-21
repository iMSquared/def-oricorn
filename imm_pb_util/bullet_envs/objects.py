from pathlib import Path
from typing import Tuple, Dict, List, Set, Optional
import numpy as np
import numpy.typing as npt
from scipy.spatial.transform import Rotation as sciR

import pybullet_data
from pybullet_utils.bullet_client import BulletClient
from imm.pybullet_util.typing_extra import TranslationT, QuaternionT, EulerT
from imm.pybullet_util.common import get_link_pose


class BulletObject:

    def __init__(
            self, 
            bc: BulletClient,
            uid: int,
    ):
        """Define wrapper from existing pybullet object"""
        self.bc = bc
        self.uid = uid

    @property
    def link_pos(self) -> TranslationT:
        pos, orn = get_link_pose(self.bc, self.uid, -1)
        return pos
    
    @property
    def link_orn(self) -> QuaternionT:
        pos, orn = get_link_pose(self.bc, self.uid, -1)
        return orn

    @property
    def inertial_pos(self) -> TranslationT:
        pos, orn = get_link_pose(self.bc, self.uid, -1, inertial=True)
        return pos

    @property
    def inertial_orn(self) -> TranslationT:
        pos, orn = get_link_pose(self.bc, self.uid, -1, inertial=True)
        return orn

    # @classmethod
    # def from_config(bc, object_config) -> "BulletObject":  
    #     """?"""
    #     CUSTOM_URDF_DIR_PATH: str          = object_config["project_params"]["custom_urdf_path"]
    #     OBJECT_PATH         : str          = object_config["path"]
    #     OBJECT_POS          : TranslationT = object_config["base_pos"]
    #     OBJECT_ORN          : EulerT       = object_config["base_orn"]
    #     # Path to custom URDF
    #     file_path = Path(__file__)
    #     project_path = file_path.parent.parent
    #     urdf_dir_path = project_path / CUSTOM_URDF_DIR_PATH
    #     uid = bc.loadURDF(
    #         fileName        = str(urdf_dir_path / OBJECT_PATH),
    #         basePosition    = OBJECT_POS,
    #         baseOrientation = bc.getQuaternionFromEuler(OBJECT_ORN),
    #         useFixedBase    = False)
    #     # Other configurations...
    #     color = object_config["color"]
    #     if color is not None:
    #         bc.changeVisualShape(uid, -1, rgbaColor=color)  
    #     return BulletObject(bc, uid)