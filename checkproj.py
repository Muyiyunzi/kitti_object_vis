"""
检查3D投影回来是否和2D重合
"""
import os, sys
import numpy as np
from utils.load_data import get_basic, get_pred


if __name__ == '__main__':
    pred_folder_name = 'monodle'  # monodle / caddn / monoflex
    dataset_root = '../dataset/kitti-object/'
    data_mode = 'val'  # train / test / val
    os.makedirs('save/logs', exist_ok=True)
    label_log = r'save/logs/pred_boxes.txt'
    pred_log = r'save/logs/pred_boxes.txt'
    f_pred = open(pred_log, 'w')

    # ======================GET INDICES==========================
    split_seq = []
    # with open('split/train.txt') as f:
    with open(f'split/{data_mode}.txt') as f:
        lines = f.readlines()
    for i in lines:
        split_seq.append(i.strip())

    # ======================TRAVERSING==========================
    print('printing calculations into files...')
    index = 0
    while index < len(split_seq):
        pic_num = split_seq[index]
        print("pic number:", pic_num)
        print("pic number:", pic_num, file=f_pred)

        img, gt, calib, lidar = get_basic(dataset_root, data_mode, pic_num)
        pred = get_pred(dataset_root, pred_folder_name, pic_num)

        # ========== check label boxes ==============

        # ========== check pred boxes ==============
        # generate gt 3d bbox in image plane
        corners3d_pred = np.zeros((len(pred), 8, 3), dtype=np.float32)  # N * 8 * 3
        for i in range(len(pred)):
            corners3d_pred[i] = pred[i].generate_corners3d()  # generate corners in 3D rect space
        _, box3ds_pred = calib.corners3d_to_img_boxes(corners3d_pred)  # project corners from 3D space to image plane

        for i, box3d in enumerate(box3ds_pred):
            bbox2d = [np.min(box3d[:, 0]), np.min(box3d[:, 1]), np.max(box3d[:, 0]), np.max(box3d[:, 1])]
            #bbox2d == gt[i].box2d
            print(bbox2d, file=f_pred)
            print(pred[i].box2d, file=f_pred)


        index += 1
    f_pred.close()