from pathlib import Path
from typing import Dict, Tuple, List, Iterable, Optional
import numpy as np
import numpy.typing as npt
from pathlib import Path
from abc import ABC, abstractmethod

import pybullet_data
from pybullet_utils.bullet_client import BulletClient
from imm.pybullet_util.common import get_link_pose
from imm.pybullet_util.typing_extra import TranslationT, EulerT, QuaternionT
from imm.pybullet_util.vision import LinkBoundingBoxVisualizer, draw_coordinate


class BulletEnv:
    """
    This is an abstract class that holds the 
    common properties of bullet environment.
    """
    def __init__(self, bc: BulletClient, add_plane: bool = True):
        """Load ground plane.

        Args:
            bc (BulletClient): _
            add_plane (bool): True
        """
        self.bc = bc
        # Register environment uids here.
        self.env_assets = {}

        if add_plane:
            pybullet_data_path = Path(pybullet_data.getDataPath())
            self.plane_uid = self.bc.loadURDF(
                        fileName        = str(pybullet_data_path / "plane.urdf"), 
                        basePosition    = (0.0, 0.0, 0.0), 
                        baseOrientation = bc.getQuaternionFromEuler((0.0, 0.0, 0.0)),
                        useFixedBase    = True)
            self.env_assets = {"plane": self.plane_uid}

    def transparent(self):
        """Set the body color to transparent"""
        for name, uid in self.env_assets.items():
            if name == "plane":
                continue
            for i in range(-1, self.bc.getNumJoints(uid)):
                self.bc.changeVisualShape(uid, i, rgbaColor=[1,1,1,0.4])

    def solid(self):
        raise NotImplementedError("")



class SimpleEnv(BulletEnv):
    """
    ?
    """
    def __init__(self, bc: BulletClient, config: Dict, add_plane: bool = True):
        """Load ground plane.

        Args:
            bc (BulletClient): _
        """
        super().__init__(bc, add_plane)

        custom_urdf_path = config["project_params"]["custom_urdf_path"]
        env_params = config["env_params"]
        asset_params_list = env_params["assets"]

        # Register environment uids here.
        self.env_assets = {}

        # Path to URDFs
        project_path = Path(__file__).parent.parent
        project_urdf_data_path = project_path / custom_urdf_path
        pybullet_data_path = Path(pybullet_data.getDataPath())
        # Load assets
        if asset_params_list is not None:
            for asset_params in asset_params_list:
                # Parse
                name = asset_params["name"]
                bullet_integrated = asset_params["bullet_integrated"]
                urdf_name = asset_params["urdf_name"]
                base_pos = asset_params["base_pos"]
                base_orn = self.bc.getQuaternionFromEuler(asset_params["base_orn"])

                full_file_path = pybullet_data_path/urdf_name if bullet_integrated \
                            else project_urdf_data_path/urdf_name

                # Load environmental URDFs
                uid = self.bc.loadURDF(
                    fileName        = str(full_file_path),
                    basePosition    = base_pos,
                    baseOrientation = base_orn,
                    useFixedBase    = True)
                # Register
                self.env_assets[name] = uid

    @property
    def env_uids(self) -> List[int]:
        return list(self.env_assets.values())

    def transparent(self):
        """Set the body color to transparent"""
        for name, uid in self.env_assets.items():
            if name == "plane":
                continue
            for i in range(-1, self.bc.getNumJoints(uid)):
                self.bc.changeVisualShape(uid, i, rgbaColor=[1,1,1,0.4])

    def solid(self):
        raise NotImplementedError("")