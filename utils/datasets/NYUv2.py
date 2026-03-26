import mmcv
import numpy as np
import os.path as osp
import tempfile
from PIL import Image
from mmcv.utils import print_log
from mmseg.datasets import DATASETS
from .custom import CustomDataset_video2


@DATASETS.register_module()
class NYUv2Dataset(CustomDataset_video2):
    """ADE20K dataset.

    In segmentation map annotation for ADE20K, 0 stands for background, which
    is not included in 150 categories. ``reduce_zero_label`` is fixed to True.
    The ``img_suffix`` is fixed to '.jpg' and ``seg_map_suffix`` is fixed to
    '.png'.
    """

    CLASSES = {"others": "0", "wall": "1", "ceiling": "2", "door": "3", "stair": "4", "ladder": "5", 
    "escalator": "6", "Playground_slide": "7", "handrail_or_fence": "8", "window": "9", 
    "rail": "10", "goal": "11", "pillar": "12", "pole": "13", "floor": "14",
    "ground": "15", "grass": "16", "sand": "17", "athletic_field": "18", "road": "19", "path": "20",
    "crosswalk": "21", "building": "22", "house": "23", "bridge": "24", "tower": "25", "windmill": "26",
    "well_or_well_lid": "27", "other_construction": "28", "sky": "29", "mountain": "30", "stone": "31",
    "wood": "32", "ice": "33", "snowfield": "34", "grandstand": "35", "sea": "36", "river": "37", 
    "lake": "38", "waterfall": "39", "water": "40", "billboard_or_Bulletin_Board": "41", "sculpture": "42",
    "pipeline": "43", "flag": "44", "parasol_or_umbrella": "45", "cushion_or_carpet": "46", "tent": "47",
    "roadblock": "48", "car": "49", "bus": "50", "truck": "51", "bicycle": "52", "motorcycle": "53",
    "wheeled_machine": "54", "ship_or_boat": "55", "raft": "56", "airplane": "57", "tyre": "58",
    "traffic_light": "59", "lamp": "60", "person": "61", "cat": "62", "dog": "63", "horse": "64",
    "cattle": "65", "other_animal": "66", "tree": "67", "flower": "68", "other_plant": "69", "toy": "70",
    "ball_net": "71", "backboard": "72", "skateboard": "73", "bat": "74", "ball": "75",
    "cupboard_or_showcase_or_storage_rack": "76", "box": "77", "traveling_case_or_trolley_case": "78",
    "basket": "79", "bag_or_package": "80", "trash_can": "81", "cage": "82", "plate": "83",
    "tub_or_bowl_or_pot": "84", "bottle_or_cup": "85", "barrel": "86", "fishbowl": "87", "bed": "88",
    "pillow": "89", "table_or_desk": "90", "chair_or_seat": "91", "bench": "92", "sofa": "93",
    "shelf": "94", "bathtub": "95", "gun": "96", "commode": "97", "roaster": "98", "other_machine": "99",
    "refrigerator": "100", "washing_machine": "101", "Microwave_oven": "102", "fan": "103", "curtain": "104",
    "textiles": "105", "clothes": "106", "painting_or_poster": "107", "mirror": "108", "flower_pot_or_vase": "109",
    "clock": "110", "book": "111", "tool": "112", "blackboard": "113", "tissue": "114", "screen_or_television": "115",
    "computer": "116", "printer": "117", "Mobile_phone": "118", "keyboard": "119", "other_electronic_product": "120",
    "fruit": "121", "food": "122", "instrument": "123", "train": "124"}

    CLASSES = tuple(list(CLASSES.keys())[0:41])

    PALETTE = [[120, 120, 120], [180, 120, 120], [6, 230, 230], [80, 50, 50],
               [4, 200, 3], [120, 120, 80], [140, 140, 140], [204, 5, 255],
               [230, 230, 230], [4, 250, 7], [224, 5, 255], [235, 255, 7],
               [150, 5, 61], [120, 120, 70], [8, 255, 51], [255, 6, 82],
               [143, 255, 140], [204, 255, 4], [255, 51, 7], [204, 70, 3],
               [0, 102, 200], [61, 230, 250], [255, 6, 51], [11, 102, 255],
               [255, 7, 71], [255, 9, 224], [9, 7, 230], [220, 220, 220],
               [255, 9, 92], [112, 9, 255], [8, 255, 214], [7, 255, 224],
               [255, 184, 6], [10, 255, 71], [255, 41, 10], [7, 255, 255],
               [224, 255, 8], [102, 8, 255], [255, 61, 6], [255, 194, 7],
               [255, 122, 8], [0, 255, 20], [255, 8, 41], [255, 5, 153],
               [6, 51, 255], [235, 12, 255], [160, 150, 20], [0, 163, 255],
               [140, 140, 140], [250, 10, 15], [20, 255, 0], [31, 255, 0],
               [255, 31, 0], [255, 224, 0], [153, 255, 0], [0, 0, 255],
               [255, 71, 0], [0, 235, 255], [0, 173, 255], [31, 0, 255],
               [11, 200, 200], [255, 82, 0], [0, 255, 245], [0, 61, 255],
               [0, 255, 112], [0, 255, 133], [255, 0, 0], [255, 163, 0],
               [255, 102, 0], [194, 255, 0], [0, 143, 255], [51, 255, 0],
               [0, 82, 255], [0, 255, 41], [0, 255, 173], [10, 0, 255],
               [173, 255, 0], [0, 255, 153], [255, 92, 0], [255, 0, 255],
               [255, 0, 245], [255, 0, 102], [255, 173, 0], [255, 0, 20],
               [255, 184, 184], [0, 31, 255], [0, 255, 61], [0, 71, 255],
               [255, 0, 204], [0, 255, 194], [0, 255, 82], [0, 10, 255],
               [0, 112, 255], [51, 0, 255], [0, 194, 255], [0, 122, 255],
               [0, 255, 163], [255, 153, 0], [0, 255, 10], [255, 112, 0],
               [143, 255, 0], [82, 0, 255], [163, 255, 0], [255, 235, 0],
               [8, 184, 170], [133, 0, 255], [0, 255, 92], [184, 0, 255],
               [255, 0, 31], [0, 184, 255], [0, 214, 255], [255, 0, 112],
               [92, 255, 0], [0, 224, 255], [112, 224, 255], [70, 184, 160],
               [163, 0, 255], [153, 0, 255], [71, 255, 0], [255, 0, 163],
               [255, 204, 0], [255, 0, 143], [0, 255, 235], [133, 255, 0]
               ]

    PALETTE = PALETTE[0:41]

    def __init__(self, **kwargs):
        super(NYUv2Dataset, self).__init__(
            img_suffix='.jpg',
            seg_map_suffix='.png',
            **kwargs)


