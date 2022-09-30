import os 
import glob 
import cv2
import numpy as np
import torch 
from torch.utils import data 
import math 

from utils import (
    pcl_to_voxels, 
    label_to_gt_box_3d, 
    camera_to_lidar_box, 
    center_to_corner_box3d,
    lidar_to_camera_point,
    angle_in_limit,
    lidar_to_camera_box,
    box3d_to_label,
    center_to_corner_box_2d,
)

from config import get_cfg_defaults 

cfg = get_cfg_defaults()


class KITTIDataset(data.Dataset):
    def __init__(self, data_dir: str, shuffle=True, augment=True, test=False):
        self.data_dir = data_dir
        self.shuffle = shuffle 
        self.test = test
        self.augment = augment
        
        self.images = glob.glob(os.path.join(self.data_dir, "image_2") + "/*.png")
        self.pcls = glob.glob(os.path.join(self.data_dir, "velodyne") + "/*.bin")
        self.labels = glob.glob(os.path.join(self.data_dir, "label_2") + "/*.txt")
        self.images = list(sorted(self.images)) 
        self.pcls = list(sorted(self.pcls)) 
        self.labels = list(sorted(self.labels)) 

        assert len(self.images) == len(self.pcls) == len(self.labels)

        self.indices = list(range(len(self.images)))
        if self.shuffle:
            np.random.shuffle(self.indices)
    
    def __getitem__(self, idx):
        index = self.indices[idx]
        tag = os.path.split(self.images[index])[1][:-4] 
        if not self.augment: 
            img = cv2.imread(self.images[index]) 
            pcl = np.fromfile(self.pcls[index], dtype=np.float32).reshape(-1, 4)
            
            if not self.test: 
                labels = [line for line in open(self.labels[index], 'r').readlines()]
            else: 
                labels = []

            voxels = pcl_to_voxels(pcl, 'Car', False) 
        
        else:
            tag, img, pcl, voxels, labels = pcl_augmentation(tag, self.data_dir) 

        return tag, img, pcl, labels, voxels 

    def __len__(self):
        return len(self.images) 


# custom batching function
def collate_fn(parts: tuple) -> tuple:
    """Batching.

    Args:
        parts (tuple): _description_

    Returns:
        tuple: Labels, Voxel-Features, Voxel-Numbers, 
        Voxel-Coordinates, RGB image, Raw Lidar 
    """
    tag = [p[0] for p in parts]
    rgb = [p[1] for p in parts] 
    raw_lidar = [p[2] for p in parts]
    label = [p[3] for p in parts] 
    voxel = [p[4] for p in parts] 

    voxel_features, voxel_numbers, voxel_coordinates = prepare_voxel(voxel) 
    
    outs = (
        tag,
        np.array(label, dtype=object),
        [torch.from_numpy(f) for f in voxel_features], 
        np.array(voxel_numbers, dtype=object),
        [torch.from_numpy(c) for c in voxel_coordinates],
        np.array(rgb, dtype=object),
        np.array(raw_lidar, dtype=object),
    )
    return outs 


# there is for sure a way to do this more efficient
def prepare_voxel(voxels: dict) -> tuple:
     
    features = []
    numbers = []
    coordinates = []

    for i, voxel in enumerate(voxels):
        features.append(voxel['feature_buffer']) # shape (K, T, 7), T=35
        numbers.append(voxel['number_buffer']) # (K,)
        coordinates.append(
            np.pad(
                voxel['coordinate_buffer'], 
                ((0, 0), (1, 0)), 
                mode='constant', 
                constant_values=i
            )
        ) # from (K, 3) to (K, 4)

    return features, numbers, coordinates


def pcl_augmentation(tag: str, data_dir: str):
    np.random.seed()

    rgb = cv2.imread(os.path.join(data_dir, 'image_2', tag + '.png'))
    
    lidar = np.fromfile(
            os.path.join(data_dir, 'velodyne', tag + '.bin'), 
            dtype=np.float32).reshape(-1, 4)
    label = np.array([line for line in open(os.path.join(
        data_dir, 'label_2', tag + '.txt'), 'r').readlines()])

    cls_name = np.array([line.split()[0] for line in label])
    gt_box_3d = label_to_gt_box_3d(
        np.array(label)[np.newaxis, :], 
        cls_name='', 
        coordinate='camera',
    )[0]

    choice = np.random.randint(0, 10)

    # paper section 3.2
    if choice >= 7:
        # bounding box augmentation 
        lidar_center_gt_box3d = camera_to_lidar_box(gt_box_3d) 
        lidar_corner_gt_box3d = center_to_corner_box3d(
                lidar_center_gt_box3d, coordinate='lidar')

        for idx in range(len(lidar_corner_gt_box3d)):
            is_collision = True 
            count = 0 
            while is_collision and count < 100:
                t_rz = np.random.uniform(-np.pi / 10, np.pi / 10)
                t_x = np.random.normal()
                t_y = np.random.normal()
                t_z = np.random.normal()

                tmp = box_transform(
                        lidar_center_gt_box3d[[idx]], 
                        t_x, t_y, t_z, t_rz, 'lidar')
                is_collision = False 

                for idy in range(idx):
                    x1, y1, w1, l1, r1 = tmp[0][[0, 1, 4, 5, 6]]
                    x2, y2, w2, l2, r2 = lidar_center_gt_box3d[idy][[0, 1, 4, 5, 6]]
                    iou = calc_iou2d(
                        np.array([x1, y1, w1, l1, r1], dtype=np.float32),
                        np.array([x2, y2, w2, l2, r2], dtype=np.float32),
                    )
                    if iou > 0:
                        is_collision = True 
                        count += 1 
                        break 
            if not is_collision:
                box_corner = lidar_corner_gt_box3d[idx]
                minx = np.min(box_corner[:, 0])
                miny = np.min(box_corner[:, 1])
                minz = np.min(box_corner[:, 2])
                maxx = np.max(box_corner[:, 0])
                maxy = np.max(box_corner[:, 1])
                maxz = np.max(box_corner[:, 2])
                bound_x = np.logical_and(lidar[:, 0] >= minx, lidar[:, 0] <= maxx)
                bound_y = np.logical_and(lidar[:, 1] >= miny, lidar[:, 1] <= maxy)
                bound_z = np.logical_and(lidar[:, 2] >= minz, lidar[:, 2] <= maxz)
                bound_box = np.logical_and(np.logical_and(bound_x, bound_y), bound_z)
                lidar[bound_box, 0:3] = point_transform(
                        lidar[bound_box, 0:3], t_x, t_y, t_z, t_rz)
                lidar_center_gt_box3d[idx] = box_transform(
                        lidar_center_gt_box3d[[idx]], t_x, t_y, t_z, t_rz, 'lidar')
        gt_box_3d = lidar_to_camera_box(lidar_center_gt_box3d)
        newtag = 'aug_{}_1_{}'.format(tag, np.random.randint(1, 1024))
    
    elif choice < 7 and choice >= 4:
        # global rotation 
        angle = np.random.uniform(-np.pi / 4, np.pi / 4)
        lidar[:, 0:3] = point_transform(lidar[:, 0:3], 0, 0, 0, rz=angle)
        lidar_center_gt_box3d = camera_to_lidar_box(gt_box_3d)
        lidar_center_gt_box3d = box_transform(
                lidar_center_gt_box3d, 0, 0, 0, angle, coordinate='lidar')
        gt_box_3d = lidar_to_camera_box(lidar_center_gt_box3d)
        newtag = 'aug_{}_2_{:.4f}'.format(tag, angle).replace('.', '_')
    
    else:
        # global scaling
        factor = np.random.uniform(0.95, 1.05)
        lidar[:, 0:3] = lidar[:, 0:3] * factor
        lidar_center_gt_box3d = camera_to_lidar_box(gt_box_3d)
        lidar_center_gt_box3d[:, 0:6] = lidar_center_gt_box3d[:, 0:6] * factor 
        gt_box_3d = lidar_to_camera_box(lidar_center_gt_box3d)
        newtag = 'aug_{}_3_{:.4f}'.format(tag, factor).replace('.', '_')

    label = box3d_to_label(
        gt_box_3d[np.newaxis, ...], 
        cls_name[np.newaxis, ...], 
        coordinate='camera',
    )[0]
    voxel_dict = pcl_to_voxels(lidar, cfg.OBJECT.NAME)
    
    return newtag, rgb, lidar, voxel_dict, label 


def calc_iou2d(box1, box2, T_VELO_2_CAM=None, R_RECT_0=None):
    buf1 = np.zeros((cfg.IMAGE.HEIGHT, cfg.IMAGE.WIDTH, 3))
    buf2 = np.zeros((cfg.IMAGE.HEIGHT, cfg.IMAGE.WIDTH, 3))
    tmp = center_to_corner_box_2d(
        np.array([box1, box2]), 
        coordinate = 'lidar', 
        T_VELO_2_CAM = T_VELO_2_CAM, 
        R_RECT_0 = R_RECT_0,
    )
    box1_corner = batch_lidar_to_birdview(tmp[0]).astype(np.int32)
    box2_corner = batch_lidar_to_birdview(tmp[1]).astype(np.int32)
    buf1 = cv2.fillConvexPoly(buf1, box1_corner, color = (1, 1, 1))[..., 0]
    buf2 = cv2.fillConvexPoly(buf2, box2_corner, color = (1, 1, 1))[..., 0]
    indiv = np.sum(np.absolute(buf1-buf2))
    share = np.sum((buf1 + buf2) == 2)
    if indiv == 0:
        return 0.0 # when target is out of bound

    return share / (indiv + share)


def batch_lidar_to_birdview(points, factor=1):
    a = (points[:, 0] - cfg.OBJECT.X_MIN) / cfg.OBJECT.X_VOXEL_SIZE * factor
    b = (points[:, 1] - cfg.OBJECT.Y_MIN) / cfg.OBJECT.Y_VOXEL_SIZE * factor
    a = np.clip(a, a_max = (cfg.OBJECT.X_MAX - cfg.OBJECT.X_MIN) /\
            cfg.OBJECT.X_VOXEL_SIZE * factor, a_min = 0)
    b = np.clip(b, a_max = (cfg.OBJECT.Y_MAX - cfg.OBJECT.Y_MIN) /\
            cfg.OBJECT.Y_VOXEL_SIZE * factor, a_min = 0)

    return np.concatenate([a[:, np.newaxis], b[:, np.newaxis]], axis = -1)


def box_transform(lidar_center_boxes, t_x, t_y, t_z, t_rz, coordinate='lidar'):
    boxes_corner = center_to_corner_box3d(
            lidar_center_boxes, coordinate=coordinate)
    for idx in range(len(boxes_corner)):
        boxes_corner[idx] = point_transform(
                boxes_corner[idx], t_x, t_y, t_z, t_rz) 

    return corner_to_center_box3d(boxes_corner, coordinate=coordinate) 


def point_transform(corner_box, tx, ty, tz, rz=0, ry=0, rx=0):
    N = corner_box.shape[0]
    points = np.hstack([corner_box, np.ones((N, 1))])
    
    # apply translation 
    mat1 = np.eye(4)
    mat1[3, 0:3] = tx, ty, tz
    points = np.matmul(points, mat1)
    
    if rx != 0:
        mat = np.zeros((4, 4)) 
        mat[0, 0] = 1
        mat[3, 3] = 1 
        mat[1, 1] = np.cos(rx) 
        mat[1, 2] = -np.sin(rx) 
        mat[2, 1] = np.sin(rx)
        mat[2, 2] = np.cos(rx)
        points = np.matmul(points, mat)
    if ry != 0:
        mat = np.zeros((4, 4)) 
        mat[1, 1] = 1
        mat[3, 3] = 1 
        mat[0, 0] = np.cos(ry) 
        mat[0, 2] = np.sin(ry) 
        mat[2, 0] = -np.sin(ry)
        mat[2, 2] = np.cos(ry)
        points = np.matmul(points, mat)

    if rz != 0:
        mat = np.zeros((4, 4)) 
        mat[2, 2] = 1
        mat[3, 3] = 1 
        mat[0, 0] = np.cos(rz) 
        mat[0, 1] = -np.sin(rz) 
        mat[1, 0] = np.sin(rz)
        mat[1, 1] = np.cos(rz)
        points = np.matmul(points, mat)

    return points[:, 0:3] 


def corner_to_center_box3d(
    boxes_corner, 
    coordinate='camera', 
    T_VELO_2_CAM = None, 
    R_RECT_0 = None,
    corner2centeravg = True,
):
    # (N, 8, 3) -> (N, 7); x,y,z,h,w,l,ry/z
    if coordinate == 'lidar':
        for idx in range(len(boxes_corner)):
            boxes_corner[idx] = lidar_to_camera_point(
                    boxes_corner[idx], T_VELO_2_CAM, R_RECT_0)
    ret = []
    for roi in boxes_corner:
        if corner2centeravg:  # average version
            roi = np.array(roi)
            h = abs(np.sum(roi[:4, 1] - roi[4:, 1]) / 4)
            w = np.sum(
                np.sqrt(np.sum((roi[0, [0, 2]] - roi[3, [0, 2]])**2)) +
                np.sqrt(np.sum((roi[1, [0, 2]] - roi[2, [0, 2]])**2)) +
                np.sqrt(np.sum((roi[4, [0, 2]] - roi[7, [0, 2]])**2)) +
                np.sqrt(np.sum((roi[5, [0, 2]] - roi[6, [0, 2]])**2))
            ) / 4
            l = np.sum(
                np.sqrt(np.sum((roi[0, [0, 2]] - roi[1, [0, 2]])**2)) +
                np.sqrt(np.sum((roi[2, [0, 2]] - roi[3, [0, 2]])**2)) +
                np.sqrt(np.sum((roi[4, [0, 2]] - roi[5, [0, 2]])**2)) +
                np.sqrt(np.sum((roi[6, [0, 2]] - roi[7, [0, 2]])**2))
            ) / 4
            x = np.sum(roi[:, 0], axis = 0) / 8
            y = np.sum(roi[0:4, 1], axis = 0) / 4
            z = np.sum(roi[:, 2], axis = 0) / 8
            ry = np.sum(
                math.atan2(roi[2, 0] - roi[1, 0], roi[2, 2] - roi[1, 2]) +
                math.atan2(roi[6, 0] - roi[5, 0], roi[6, 2] - roi[5, 2]) +
                math.atan2(roi[3, 0] - roi[0, 0], roi[3, 2] - roi[0, 2]) +
                math.atan2(roi[7, 0] - roi[4, 0], roi[7, 2] - roi[4, 2]) +
                math.atan2(roi[0, 2] - roi[1, 2], roi[1, 0] - roi[0, 0]) +
                math.atan2(roi[4, 2] - roi[5, 2], roi[5, 0] - roi[4, 0]) +
                math.atan2(roi[3, 2] - roi[2, 2], roi[2, 0] - roi[3, 0]) +
                math.atan2(roi[7, 2] - roi[6, 2], roi[6, 0] - roi[7, 0])
            ) / 8
            if w > l:
                w, l = l, w
                ry = angle_in_limit(ry + np.pi / 2)
        else:  # max version
            h = max(abs(roi[:4, 1] - roi[4:, 1]))
            w = np.max(
                np.sqrt(np.sum((roi[0, [0, 2]] - roi[3, [0, 2]])**2)) +
                np.sqrt(np.sum((roi[1, [0, 2]] - roi[2, [0, 2]])**2)) +
                np.sqrt(np.sum((roi[4, [0, 2]] - roi[7, [0, 2]])**2)) +
                np.sqrt(np.sum((roi[5, [0, 2]] - roi[6, [0, 2]])**2))
            )
            l = np.max(
                np.sqrt(np.sum((roi[0, [0, 2]] - roi[1, [0, 2]])**2)) +
                np.sqrt(np.sum((roi[2, [0, 2]] - roi[3, [0, 2]])**2)) +
                np.sqrt(np.sum((roi[4, [0, 2]] - roi[5, [0, 2]])**2)) +
                np.sqrt(np.sum((roi[6, [0, 2]] - roi[7, [0, 2]])**2))
            )
            x = np.sum(roi[:, 0], axis = 0) / 8
            y = np.sum(roi[0:4, 1], axis = 0) / 4
            z = np.sum(roi[:, 2], axis = 0) / 8
            ry = np.sum(
                math.atan2(roi[2, 0] - roi[1, 0], roi[2, 2] - roi[1, 2]) +
                math.atan2(roi[6, 0] - roi[5, 0], roi[6, 2] - roi[5, 2]) +
                math.atan2(roi[3, 0] - roi[0, 0], roi[3, 2] - roi[0, 2]) +
                math.atan2(roi[7, 0] - roi[4, 0], roi[7, 2] - roi[4, 2]) +
                math.atan2(roi[0, 2] - roi[1, 2], roi[1, 0] - roi[0, 0]) +
                math.atan2(roi[4, 2] - roi[5, 2], roi[5, 0] - roi[4, 0]) +
                math.atan2(roi[3, 2] - roi[2, 2], roi[2, 0] - roi[3, 0]) +
                math.atan2(roi[7, 2] - roi[6, 2], roi[6, 0] - roi[7, 0])
            ) / 8
            if w > l:
                w, l = l, w
                ry = angle_in_limit(ry + np.pi / 2)
        ret.append([x, y, z, h, w, l, ry])
    if coordinate == 'lidar':
        ret = camera_to_lidar_box(np.array(ret), T_VELO_2_CAM, R_RECT_0)

    return np.array(ret)


def test():
    data_dir = "/data/kitti/3d_vision/training" 
    dataset = KITTIDataset(data_dir)
    print(f"Length dataset: {len(dataset)}")

    for img, pcl, labels, voxels in dataset:
        print(img.shape)
        print(pcl.shape)
        print(len(labels))
        print(voxels.keys())
        break

    img, pcl, labels, voxels = dataset[12]
    print(img.shape)
    print(pcl.shape)
    print(len(labels))
    print(voxels.keys())


    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=4, shuffle=False, collate_fn=collate_fn)

    for data in dataloader:
        print(type(data)) 
        label, voxel_features, voxel_numbers, voxel_coordinates, rgb, raw_lidar = data 
        print(type(label)) 
        print(len(voxel_coordinates))
        print(len(voxel_features)) 
        break


if __name__ == "__main__":
    test()
