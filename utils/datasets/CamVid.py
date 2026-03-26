import mmcv
import numpy as np
import os.path as osp
import tempfile
from PIL import Image
from mmcv.utils import print_log
from mmseg.datasets import DATASETS
from .custom import CustomDataset_video2


@DATASETS.register_module()
class CamVidDataset(CustomDataset_video2):
    """ADE20K dataset.

    In segmentation map annotation for ADE20K, 0 stands for background, which
    is not included in 150 categories. ``reduce_zero_label`` is fixed to True.
    The ``img_suffix`` is fixed to '.jpg' and ``seg_map_suffix`` is fixed to
    '.png'.
    """

    CLASSES = {
        "Animal": "0",
        "Archway": "1",
        "Bicyclist": "2",
        "Bridge": "3",
        "Building": "4",
        "Car": "5",
        "CartLuggagePram": "6",
        "Child": "7",
        "Column_Pole": "8",
        "Fence": "9",
        "LaneMkgsDriv": "10",
        "LaneMkgsNonDriv": "11",
        "Misc_Text": "12",
        "MotorcycleScooter": "13",
        "OtherMoving": "14",
        "ParkingBlock": "15",
        "Pedestrian": "16",
        "Road": "17",
        "RoadShoulder": "18",
        "Sidewalk": "19",
        "SignSymbol": "20",
        "Sky": "21",
        "SUVPickupTruck": "22",
        "TrafficCone": "23",
        "TrafficLight": "24",
        "Tree": "25",
        "Truck_Bus": "26",
        "Tunnel": "27",
        "VegetationMisc": "28",
        "Void": "29",
        "Wall": "30"
    }

    CLASSES = tuple(list(CLASSES.keys()))

    PALETTE = [
        [64, 128, 64],      # Animal
        [192, 0, 128],      # Archway
        [0, 128, 192],      # Bicyclist
        [0, 128, 64],       # Bridge
        [128, 0, 0],        # Building
        [64, 0, 128],       # Car
        [64, 0, 192],       # CartLuggagePram
        [192, 128, 64],     # Child
        [192, 192, 128],    # Column_Pole
        [64, 64, 128],      # Fence
        [128, 0, 192],      # LaneMkgsDriv
        [192, 0, 64],       # LaneMkgsNonDriv
        [128, 128, 64],     # Misc_Text
        [192, 0, 192],      # MotorcycleScooter
        [128, 64, 64],      # OtherMoving
        [64, 192, 128],     # ParkingBlock
        [64, 64, 0],        # Pedestrian
        [128, 64, 128],     # Road
        [128, 128, 192],    # RoadShoulder
        [0, 0, 192],        # Sidewalk
        [192, 128, 128],    # SignSymbol
        [128, 128, 128],    # Sky
        [64, 128, 192],     # SUVPickupTruck
        [0, 0, 64],         # TrafficCone
        [0, 64, 64],        # TrafficLight
        [128, 128, 0],      # Tree
        [192, 128, 192],    # Truck_Bus
        [64, 0, 64],        # Tunnel
        [192, 192, 0],      # VegetationMisc
        [0, 0, 0],          # Void
        [64, 192, 0]        # Wall
    ]

    def __init__(self, **kwargs):
        super(CamVidDataset, self).__init__(
            img_suffix='.png',
            seg_map_suffix='.png',
            **kwargs)


