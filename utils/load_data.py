import os
import numpy as np
from utils.kitti.kitti_utils import get_objects_from_label
from utils.kitti.calibration import Calibration
from PIL import Image

def get_image(img_file):
    assert os.path.exists(img_file)
    return Image.open(img_file)  # (H, W, 3) RGB mode

def get_label(label_file):
    assert os.path.exists(label_file)
    return get_objects_from_label(label_file)

def get_calib(calib_file):
    assert os.path.exists(calib_file)
    return Calibration(calib_file)

def get_lidar(lidar_file):
    assert os.path.exists(lidar_file)
    return np.fromfile(lidar_file, dtype=np.float32).reshape(-1, 4)

def get_basic(dataset_root, mode, index):
    if mode == 'train' or mode == 'trainval' or mode == 'val':
        folder = 'training'
    else:
        folder = 'testing'
    # loading image
    img_file = os.path.join(dataset_root, folder, 'image_2', index + '.png')
    img = get_image(img_file)

    # loading objects (ground truth or predicted boxes)
    gt_file = os.path.join(dataset_root, folder, 'label_2', index + '.txt')
    gt = get_label(gt_file)

    # loading calib
    calib_file = os.path.join(dataset_root, folder, 'calib', index + '.txt')
    calib = get_calib(calib_file)

    # loading lidar points
    lidar_file = os.path.join(dataset_root, folder, 'velodyne', index + '.bin')
    points = get_lidar(lidar_file)

    return img, gt, calib, points


def get_pred(dataset_root, pred_folder_name, index):
    # loading pred files
    pred_file = os.path.join(dataset_root, 'pred', pred_folder_name, index + '.txt')
    pred = get_label(pred_file)
    return pred