"""
将数据集GT的2D、3D bbox投影在2D图像中,同时将lidar可视化到mayavi的3D场景中
"""
import os
import cv2
from PIL import Image
import numpy as np
import psutil
from utils.load_data import get_basic, get_pred
import mayavi.mlab as mlab


def draw_projected_box3d(image, corners3d, color=(255, 255, 255), thickness=1):
    '''
    draw 3d bounding box in image plane
    input:
        image: RGB image
        corners3d: (8,3) array of vertices (in image plane) for the 3d box in following order:
            1 -------- 0
           /|         /|
          2 -------- 3 .
          | |        | |
          . 5 -------- 4
          |/         |/
          6 -------- 7
    '''

    corners3d = corners3d.astype(np.int32)
    # Ref: http://docs.enthought.com/mayavi/mayavi/auto/mlab_helper_functions.html
    for k in range(0, 4):
        i, j = k, (k + 1) % 4
        cv2.line(image, (corners3d[i, 0], corners3d[i, 1]), (corners3d[j, 0], corners3d[j, 1]), color, thickness)

        i, j = k + 4, (k + 1) % 4 + 4
        cv2.line(image, (corners3d[i, 0], corners3d[i, 1]), (corners3d[j, 0], corners3d[j, 1]), color, thickness)

        i, j = k, k + 4
        cv2.line(image, (corners3d[i, 0], corners3d[i, 1]), (corners3d[j, 0], corners3d[j, 1]), color, thickness)

    return image



def draw_lidar_points(pc, color=None, fig=None, bgcolor=(0,0,0), pts_scale=2, pts_mode='point'):
    if pc.shape[0]==0:
        return fig
    if fig is None:
        fig = mlab.figure(figure=None, bgcolor=bgcolor, fgcolor=None, engine=None, size=(1600, 1000))

    if color is None:
        x = pc[:, 0]  # x position of point
        y = pc[:, 1]  # y position of point
        col = np.sqrt(x ** 2 + y ** 2)  # map distance
    else:
        col = color

    if pts_mode=='sphere':
        mlab.points3d(pc[:,0], pc[:,1], pc[:,2], color=col, mode='sphere', scale_factor=0.1, figure=fig)
    else:
        mlab.points3d(pc[:,0], pc[:,1], pc[:,2], col, mode='point', colormap='spectral', scale_factor=pts_scale, figure=fig)

    # draw origin point
    mlab.points3d(0, 0, 0, color=(1,1,1), mode='sphere', scale_factor=0.2)

    # draw axis
    axes=np.array([
        [1.,0.,0.,0.],
        [0.,1.,0.,0.],
        [0.,0.,1.,0.],
    ],dtype=np.float64)
    mlab.plot3d([0, axes[0,0]], [0, axes[0,1]], [0, axes[0,2]], color=(1,0,0), tube_radius=None, figure=fig)
    mlab.plot3d([0, axes[1,0]], [0, axes[1,1]], [0, axes[1,2]], color=(0,1,0), tube_radius=None, figure=fig)
    mlab.plot3d([0, axes[2,0]], [0, axes[2,1]], [0, axes[2,2]], color=(0,0,1), tube_radius=None, figure=fig)

    return fig


def draw_lidar_3dbox(box3d, fig, color=(1, 1, 1), line_width=1, draw_text=True, text_scale=(1, 1, 1), color_list=None):
    num = len(box3d)
    for n in range(num):
        b = box3d[n]
        if color_list is not None:
            color = color_list[n]
        for k in range(0,4):
            #http://docs.enthought.com/mayavi/mayavi/auto/mlab_helper_functions.html
            i,j=k,(k+1)%4
            mlab.plot3d([b[i,0], b[j,0]], [b[i,1], b[j,1]], [b[i,2], b[j,2]], color=color, tube_radius=None, line_width=line_width, figure=fig)

            i,j=k+4,(k+1)%4 + 4
            mlab.plot3d([b[i,0], b[j,0]], [b[i,1], b[j,1]], [b[i,2], b[j,2]], color=color, tube_radius=None, line_width=line_width, figure=fig)

            i,j=k,k+4
            mlab.plot3d([b[i,0], b[j,0]], [b[i,1], b[j,1]], [b[i,2], b[j,2]], color=color, tube_radius=None, line_width=line_width, figure=fig)
    return fig


def kill_display():
    for proc in psutil.process_iter():  # 遍历当前process
        if proc.name() == "display":  # 如果process的name是display
            proc.kill()
    mlab.close()

def draw_box2d(image, box2d, class_type, gt=True, confidence=1, thickness = 1):
    # box2d label:
    if class_type=='DontCare':
        return image
    if gt is True:
        if class_type=='Car' or class_type=='Pedestrian' or class_type=='Cyclist':
            # ground truth uses color of green
            cv2.rectangle(image, (box2d[0], box2d[1]), (box2d[2], box2d[3]), color=(0, 255, 0), thickness=thickness)
        else:
            # color of yellow, doesn't count as false positive, meaning neglectable
            cv2.rectangle(image, (box2d[0], box2d[1]), (box2d[2], box2d[3]), color=(0, 255, 255), thickness=thickness)
    else:
        # pred label uses color of red, and also shows confidence for 3d bounding box (which is too crowded, so I'll put it in the 2d image)
        cv2.rectangle(image, (box2d[0], box2d[1]), (box2d[2], box2d[3]), color=(0, 0, 255), thickness=thickness)
        cv2.putText(image, "%02d" % (confidence*100), (box2d[0], int(box2d[1] - 2)), fontFace=2, fontScale=0.6, color=(0, 0, 255), thickness=1)
    return image


if __name__ == '__main__':

    # ======================BASIC SETTINGS==========================
    # put the output in the pred folder: see 'pred_file' in line 172
    pred_folder_name = 'monodle' # monodle / caddn / monoflex
    dataset_root = '../dataset/kitti-object/'
    data_mode = 'val' # train / test / val
    check_mode = 'save_only' # check_only / check_and_save / save_only
    getPred = False # True / False
    os.makedirs('save/image', exist_ok=True)
    os.makedirs('save/scene', exist_ok=True)

    # ======================GET INDICES==========================
    split_seq = []
    # with open('split/train.txt') as f:
    with open(f'split/{data_mode}.txt') as f:
        lines = f.readlines()
    for i in lines:
        split_seq.append(i.strip())

    # ======================TRAVERSING==========================
    index = 0
    while index < len(split_seq):
        pic_num = split_seq[index]
        print("pic number:", pic_num)

        img, gt, calib, lidar = get_basic(dataset_root, data_mode, pic_num)


        # ======================VISUALIZE BBOX IN 2D IMAGE==========================

        # PIL to cv2 for the requirements of drawing function
        img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
        img2d = np.copy(img)
        img3d = np.copy(img)

        #===================== draw 3d bbox in image plane =====================
        # generate gt 3d bbox in image plane
        corners3d_gt = np.zeros((len(gt), 8, 3), dtype=np.float32)  # N * 8 * 3

        for i in range(len(gt)):
            corners3d_gt[i] = gt[i].generate_corners3d()  # generate corners in 3D rect space
        _, box3ds_gt = calib.corners3d_to_img_boxes(corners3d_gt) # project corners from 3D space to image plane

        # draw gt 3d bbox
        for box3d in box3ds_gt:
            img3d = draw_projected_box3d(img3d, box3d, color=(0, 255, 0), thickness=1)

        # ===================== draw 2d bbox in another image =====================
        corners2d_gt = np.zeros((len(gt), 4), dtype=np.float32)  # N * 4
        for i in range(len(gt)):
            class_type, score, corners2d_gt[i] = gt[i].generate_corners2d()
            img2d = draw_box2d(img2d, corners2d_gt[i], class_type, gt=True)

        # ======================VISUALIZE BBOX IN 3D SPACE==========================

        fig = mlab.figure(size=(1200, 800), bgcolor=(0.9, 0.9, 0.9))
        fig = draw_lidar_points(lidar, fig=fig)

        # draw gt boxes
        corners3d_gt = calib.rect_to_lidar(corners3d_gt.reshape(-1, 3))  # from rect coordinate system to lidar coordinate system
        corners3d_gt = corners3d_gt.reshape(-1, 8, 3)
        fig = draw_lidar_3dbox(corners3d_gt, fig=fig, color=(0, 1, 0))

        # ===================== do it all again for pred boxes =====================
        if getPred:
            pred = get_pred(dataset_root, pred_folder_name, pic_num)

            # 3d
            corners3d_pred = np.zeros((len(pred), 8, 3), dtype=np.float32)  # N * 8 * 3
            for i in range(len(pred)):
                corners3d_pred[i] = pred[i].generate_corners3d()
            _, box3ds_pred = calib.corners3d_to_img_boxes(corners3d_pred)
            for box3d in box3ds_pred:
                img3d = draw_projected_box3d(img3d, box3d, color=(0, 0, 255), thickness=1)

            # 2d
            corners2d_pred = np.zeros((len(pred), 4), dtype=np.float32)  # N * 4
            for i in range(len(pred)):
                class_type, score, corners2d_pred[i] = pred[i].generate_corners2d()
                img2d = draw_box2d(img2d, corners2d_pred[i], class_type, gt=False, confidence=score)

            # lidar
            corners3d_pred = calib.rect_to_lidar(corners3d_pred.reshape(-1, 3))
            corners3d_pred = corners3d_pred.reshape(-1, 8, 3)
            fig = draw_lidar_3dbox(corners3d_pred, fig=fig, color=(1, 0, 0))

        # ======================SHOW OR SAVE==========================
        # cv2 to PIL and resize. Not all images are 1242, 375.
        img3d = Image.fromarray(cv2.cvtColor(img3d, cv2.COLOR_BGR2RGB)).resize((1242, 375))
        img2d = Image.fromarray(cv2.cvtColor(img2d, cv2.COLOR_BGR2RGB)).resize((1242, 375))

        new_img = Image.new('RGB', (1242, 750))
        new_img.paste(img2d, (0, 0))
        new_img.paste(img3d, (0, 375))
        mlab.view(azimuth=180, elevation=70, focalpoint=[12.0909996, -1.04700089, -2.03249991], distance=62.0, figure=fig)

        if check_mode == 'check_only' or check_mode == 'check_and_save':
            new_img.show()

        if check_mode == 'save_only' or check_mode == 'check_and_save':
            # img3d.save('save/image/img_3d_' + pic_num + '.png') # uncomment if you want to save separately
            # img2d.save('save/image/img_2d_' + pic_num + '.png') # uncomment if you want to save separately
            new_img.save('save/image/img_composed' + pic_num + '.png')
            mlab.savefig('save/scene/scene_' + pic_num +'.png', figure=fig)

        if check_mode == 'save_only':
            kill_display()
            print("image No.%s saved." % pic_num)
            index += 1
        else:
            key = input("proceed?(y for next;n for last;q for quit):")
            if key == 'y':
                index += 1
                kill_display()
            elif key == 'n':
                index -= 1
                kill_display()
            else:
                kill_display()
                exit()
