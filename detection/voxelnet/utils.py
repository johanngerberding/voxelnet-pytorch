import torch 
import numpy as np 
import cv2 
import math 


from config import get_cfg_defaults
cfg = get_cfg_defaults()


def pcl_to_voxels(pcl, target: str, verbose: bool = False) -> dict:
    """Preprocess point cloud. Exclude points out of defined range, 
    create an index for the voxels and the voxel features as described in the 
    paper section 2.1.1.  

    Args:
        pcl (np.ndarray): Velodyne point cloud 
        target (str): Class value ("Car", "Pedestrian", "Cyclist")
        verbose (bool, optional): Print info about voxel dict. Defaults to False.

    Returns:
        dict: Voxel dict containing features, unique voxels and number of points per voxel 
    """     

    if target == 'Car':
        voxel_size = np.array([0.4, 0.2, 0.2], dtype=np.float32)
        grid_size = np.array([10, 400, 352], dtype=np.float32)
        lidar_coord = np.array([0, 40, 3], dtype=np.float32)
        max_number_points = 35  
    else: # pedestrian & cyclist 
        voxel_size = np.array([0.4, 0.2, 0.2], dtype=np.float32)
        grid_size = np.array([10, 200, 240], dtype=np.float32)
        lidar_coord = np.array([0, 20, 3], dtype=np.float32)
        max_number_points = 45
    
    np.random.shuffle(pcl)

    shifted_coord = pcl[:, :3] + lidar_coord
    # change pcl coordinates order (X,Y,Z) -> (Z,Y,X) 
    # Voxel index for every point (point to voxel) 
    voxel_index = np.floor(shifted_coord[:, ::-1] / voxel_size).astype(int)
    
    bound_x = np.logical_and(
        voxel_index[:, 2] >= 0,
        voxel_index[:, 2] < grid_size[2]
    )

    bound_y = np.logical_and(
        voxel_index[:, 1] >= 0,
        voxel_index[:, 1] < grid_size[1]
    )

    bound_z = np.logical_and(
        voxel_index[:, 0] >= 0,
        voxel_index[:, 0] < grid_size[0]
    )

    bound_box = np.logical_and(np.logical_and(bound_x, bound_y), bound_z)
    # filter point cloud, only points in the box (also for voxels)  
    
    pcl = pcl[bound_box]
    voxel_index = voxel_index[bound_box]
    # [K, 3] coordinate buffer, number of unique voxels 
    coordinate_buffer = np.unique(voxel_index, axis=0)
    
    K = len(coordinate_buffer)
    T = max_number_points

    # [K, 1] store number of points in each voxel grid 
    number_buffer = np.zeros(shape=(K), dtype=np.int64) 
    # [K, T, 7] feature buffer
    feature_buffer = np.zeros(shape=(K, T, 7), dtype=np.float32)

    # index all the voxels 
    index_buffer = {}
    for i in range(K):
        index_buffer[tuple(coordinate_buffer[i])] = i

    for voxel, point in zip(voxel_index, pcl):
        idx = index_buffer[tuple(voxel)] 
        number = number_buffer[idx]

        if number < T: 
            feature_buffer[idx, number, :4] = point 
            number_buffer[idx] += 1 

    # add relative offsets as last 3 values (coordinate - centroids), see paper 2.1.1  
    feature_buffer[:, :, -3:] = feature_buffer[:, :, :3] - \
        feature_buffer[:, :, :3].sum(axis=1, keepdims=True) / number_buffer.reshape(K, 1, 1)

    voxel_dict = {
        'feature_buffer': feature_buffer, # voxel features 
        'coordinate_buffer': coordinate_buffer, # unique voxels
        'number_buffer': number_buffer, # number of points per voxel
    }
    if verbose:  
        print(f"Coordinate buffer shape: {voxel_dict['coordinate_buffer'].shape}")
        print(f"Feature buffer shape: {voxel_dict['feature_buffer'].shape}")
        print(f"Number buffer shape: {voxel_dict['number_buffer'].shape}")
        
    return voxel_dict



def generate_anchors():
    """Generate anchors.

    Args:
        cfg (_type_): YACS config object 

    Returns:
        np.ndarray: Anchors 
    """    
    x = np.linspace(cfg.OBJECT.X_MIN, cfg.OBJECT.X_MAX, cfg.OBJECT.FEATURE_WIDTH) 
    y = np.linspace(cfg.OBJECT.Y_MIN, cfg.OBJECT.Y_MAX, cfg.OBJECT.FEATURE_HEIGHT)
    cx, cy = np.meshgrid(x, y) 
    cx = np.tile(cx[..., np.newaxis], 2)
    cy = np.tile(cy[..., np.newaxis], 2)
    cz = np.ones_like(cx) * cfg.OBJECT.ANCHOR_Z
    w = np.ones_like(cx) * cfg.OBJECT.ANCHOR_W
    l = np.ones_like(cx) * cfg.OBJECT.ANCHOR_L 
    h = np.ones_like(cx) * cfg.OBJECT.ANCHOR_H
    r = np.ones_like(cx) 
    r[..., 0] = 0 
    r[..., 1] = 90 / 180 * np.pi  # 90 

    anchors = np.stack([
        cx, cy, cz, h, w, l, r
    ], axis=-1) 
    
    return anchors  


def angle_in_limit(angle):
    """Limit the angle between -pi/2 and pi/2"""     
    while angle >= np.pi / 2:
        angle -= np.pi 
    while angle < -np.pi / 2:
        angle += np.pi 

    if abs(angle + np.pi / 2) < 5 / 180 * np.pi:
        angle = np.pi / 2  
    
    return angle 


def camera_to_lidar(x, y, z, T_VELO_2_CAM=None, R_RECT_0=None):
    """Transform camera coordinates to lidar coordinates."""
    if not T_VELO_2_CAM:
        T_VELO_2_CAM = np.array(cfg.CALIB.T_VELO_2_CAM)

    if not R_RECT_0:
        R_RECT_0 = np.array(cfg.CALIB.R_RECT_0) 

    point = np.array([x, y, z, 1])
    point = np.matmul(np.linalg.inv(R_RECT_0), point)
    point = np.matmul(np.linalg.inv(T_VELO_2_CAM), point) 
    point = point[:3]
    
    return point 


def camera_to_lidar_box(boxes, T_VELO_2_CAM=None, R_RECT_0=None):
    """Transform boxes in camera coordinates to lidar coordinates."""  
    lidar_boxes = []
    for box in boxes: 
        x, y, z, h, w, l, ry = box 
        (x, y, z), h, w, l, rz = camera_to_lidar(x, y, z, T_VELO_2_CAM, R_RECT_0), \
            h, w, l, -ry - np.pi / 2 
        rz = angle_in_limit(rz)
        lidar_boxes.append([x, y, z, h, w, l, rz])
    
    return np.array(lidar_boxes).reshape(-1, 7)



def label_to_gt_box_3d(
    labels, 
    cls_name: str, 
    coordinate: str, 
    T_VELO_2_CAM = None, 
    R_RECT_0 = None,
):
    boxes3d = []

    if cls_name == 'Car':
        acc_cls = ['Car', 'Van']
    elif cls_name == 'Pedestrian':
        acc_cls = ['Pedestrian'] 
    elif cls_name == 'Cyclist':
        acc_cls = ['Cyclist'] 
    else: 
        acc_cls = []

    for label in labels:
        boxes3d_label = []
        for line in label:
            anno = line.split()
            if anno[0] in acc_cls or acc_cls == []:
                h, w, l, x, y, z, r = [float(i) for i in anno[-7:]]
                box3d = np.array([x, y, z, h, w, l, r])
                boxes3d_label.append(box3d)

        if coordinate == 'lidar':
            boxes3d_label = camera_to_lidar_box(
                np.array(boxes3d_label), T_VELO_2_CAM, R_RECT_0)
        
        boxes3d.append(np.array(boxes3d_label).reshape(-1, 7))

    return boxes3d


# TODO: integration in generate_anchors??
def anchor_to_standup_box2d(anchors):
    # x, y, w, l -> x1, y1, x2, y2
    anchor_standup = np.zeros_like(anchors)

    anchor_standup[::2, 0] = anchors[::2, 0] - anchors[::2, 3] / 2   
    anchor_standup[::2, 1] = anchors[::2, 1] - anchors[::2, 2] / 2   
    anchor_standup[::2, 2] = anchors[::2, 0] - anchors[::2, 3] / 2   
    anchor_standup[::2, 3] = anchors[::2, 1] - anchors[::2, 2] / 2   

    anchor_standup[1::2, 0] = anchors[1::2, 0] - anchors[1::2, 2] / 2   
    anchor_standup[1::2, 1] = anchors[1::2, 1] - anchors[1::2, 3] / 2   
    anchor_standup[1::2, 2] = anchors[1::2, 0] - anchors[1::2, 2] / 2   
    anchor_standup[1::2, 3] = anchors[1::2, 1] - anchors[1::2, 3] / 2   
    
    return anchor_standup


def corner_to_standup_box2d(boxes_corners):
    # (bs, 4, 2) -> (N, 4) = x1, y1, x2, y2
    N = boxes_corners.shape[0]
    standup_boxes_2d = np.zeros((N, 4))
    standup_boxes_2d[:, 0] = np.min(boxes_corners[:, :, 0], axis=1)
    standup_boxes_2d[:, 1] = np.min(boxes_corners[:, :, 1], axis=1)
    standup_boxes_2d[:, 2] = np.max(boxes_corners[:, :, 0], axis=1)
    standup_boxes_2d[:, 3] = np.max(boxes_corners[:, :, 1], axis=1)

    return standup_boxes_2d


def center_to_corner_box_2d(
    boxes_center,
    coordinate='lidar', 
    T_VELO_2_CAM=None, 
    R_RECT_0=None,
):  
    # (N, 5) -> (N, 4, 2)
    N = boxes_center.shape[0]
    boxes_3d_center = np.zeros((N, 7))
    boxes_3d_center[:, [0, 1, 4, 5, 6]] = boxes_center
    boxes_3d_corner = center_to_corner_box_3d(
        boxes_3d_center, coordinate, T_VELO_2_CAM, R_RECT_0
    )

    return boxes_3d_corner[:, 0:4, 0:2] 


def lidar_to_camera_point(points, T_VELO_2_CAM=None, R_RECT_0=None):
    # (N,3) -> (N,3) 
    N = points.shape[0]
    points = np.hstack([points, np.ones((N, 1))]).T
    
    if type(T_VELO_2_CAM) == type(None):
        T_VELO_2_CAM = np.array(cfg.CALIB.T_VELO_2_CAM)
    
    if type(R_RECT_0) == type(None):
        R_RECT_0 = np.array(cfg.CALIB.R_RECT_0)

    points = np.matmul(T_VELO_2_CAM, points)
    points = np.matmul(R_RECT_0, points).T
    points = points[:, 0:3]

    return points.reshape(-1, 3)


def lidar_to_camera(x, y, z, T_VELO_2_CAM=None, R_RECT_0=None):
    """Transform lidar coordinates to camera coordinates."""    
    if not T_VELO_2_CAM:
        T_VELO_2_CAM = np.array(cfg.CALIB.T_VELO_2_CAM)

    if not R_RECT_0:
        R_RECT_0 = np.array(cfg.CALIB.R_RECT_0) 

    # homogenuous point
    point = np.array([x, y, z, 1]) 
    point = np.matmul(T_VELO_2_CAM, point)
    point = np.matmul(R_RECT_0, point)
    point = point[0:3]

    return tuple(point) 


def center_to_corner_box_3d(
    boxes_center, 
    coordinate, 
    T_VELO_2_CAM=None, 
    R_RECT_0=None,
):
    # (N, 7) -> (N, 8, 3) = 8 corners in 3d
    N = boxes_center.shape[0]
    corner_box_3d = np.zeros((N, 8, 3), dtype=np.float32)

    if coordinate == 'camera':
        boxes_center = camera_to_lidar_box(boxes_center, T_VELO_2_CAM, R_RECT_0)
    
    for i in range(N):
        box = boxes_center[i]
        translation = box[0:3]
        size = box[3:6]
        rotation = [0, 0, box[-1]]

        h, w, l = size[0], size[1], size[2]

        # in velodyne coordinates around zero point and without orientation
        tracklet_box = np.array([
            [-l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2], \
            [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2], \
            [0, 0, 0, 0, h, h, h, h]
        ])

        # re-create 3D bounding box in velodyne coordinate system 
        yaw = rotation[2] 
        rot_mat = np.array([
            [np.cos(yaw), -np.sin(yaw), 0.0],
            [np.sin(yaw), np.cos(yaw), 0.0],
            [0.0, 0.0, 1.0]
        ])

        corner_pos_in_velo = np.dot(rot_mat, tracklet_box) +\
             np.tile(translation, (8, 1)).T

        box3d = corner_pos_in_velo.transpose()
        corner_box_3d[i] = box3d
    
    if coordinate == 'camera':
        for idx in range(len(corner_box_3d)):
            corner_box_3d[idx] = lidar_to_camera_point(
                corner_box_3d[idx], T_VELO_2_CAM, R_RECT_0)

    return corner_box_3d 


def bbox_iou(box1: np.ndarray, box2: np.ndarray):
    N = box1.shape[0]
    K = box2.shape[0]

    overlaps = np.zeros((N, K), dtype=np.float32)

    for k in range(K):
        box_area = (
            (box2[k, 2] - box2[k, 0] + 1) * (box2[k, 3] - box2[k, 1] + 1)
        ) 
        for n in range(N):
            iw = (
                min(box1[n, 2], box2[k, 2]) - 
                max(box1[n, 0], box2[k, 0]) + 1
            )
            if iw > 0: 
                ih = (
                    min(box1[n, 3], box2[k, 3]) - 
                    max(box1[n, 1], box2[k, 1]) + 1
                )

                if ih > 0:
                    ua = float(
                        (box1[n, 1] - box1[n, 0] + 1) *
                        (box1[n, 3] - box1[n, 1] + 1) + 
                        box_area - iw * ih
                    )
                    overlaps[n, k] = iw * ih / ua
    
    return overlaps 


def generate_targets(
    labels, 
    feature_map_shape, 
    anchors, 
    cls_name='Car', 
    coordinate='lidar', 
):
    batch_size = labels.shape[0]
    batch_gt_boxes_3d = label_to_gt_box_3d(labels, cls_name, coordinate)

    anchors_reshaped = anchors.reshape(-1, 7)
    # diagonal of the base of the anchor box (section 2.2)
    anchors_diag = np.sqrt(
        anchors_reshaped[:, 4] ** 2 + anchors_reshaped[:, 5] ** 2
    ) 

    pos_equal_one = np.zeros((batch_size, *feature_map_shape, 2))
    neg_equal_one = np.zeros((batch_size, *feature_map_shape, 2))
    targets = np.zeros((batch_size, *feature_map_shape, 14))

    for batch_id in range(batch_size):
        # transform anchors from (x, y, w, l) to (x1, y1, x2, y2)
        anchors_standup_2d = anchor_to_standup_box2d(anchors_reshaped[:, [0, 1, 4, 5]])
        gt_standup_2d = corner_to_standup_box2d(
            center_to_corner_box_2d(
                batch_gt_boxes_3d[batch_id][:, [0, 1, 4, 5, 6]], coordinate
            )
        )        
        
        # calculate iou between anchors and gt boxes
        iou = bbox_iou(
            np.ascontiguousarray(anchors_standup_2d).astype(np.float32),
            np.ascontiguousarray(gt_standup_2d).astype(np.float32),
        )
        # find anchor with highest iou 
        id_max = np.argmax(iou.T, axis=1)
        id_max_gt = np.arange(iou.T.shape[0])
        mask = iou.T[id_max_gt, id_max] > 0 
        id_max, id_max_gt = id_max[mask], id_max_gt[mask]

        # get anchour iou > cfg.OBJECT.POS_IOU
        id_pos, id_pos_gt = np.where(iou > cfg.OBJECT.RPN_POS_IOU)
        # get anchor iou < cfg.OBJECT.NEG_IOU
        id_neg = np.where(np.sum(iou < cfg.OBJECT.RPN_NEG_IOU, axis=1) == iou.shape[1])[0]

        id_pos = np.concatenate([id_pos, id_max])
        id_pos_gt = np.concatenate([id_pos_gt, id_max_gt])
        
        id_pos, idx = np.unique(id_pos, return_index=True)
        id_pos_gt = id_pos_gt[idx]
        id_neg.sort()

        index_x, index_y, index_z = np.unravel_index(id_pos, (*feature_map_shape, 2))
        pos_equal_one[batch_id, index_x, index_y, index_z] = 1

        # delta x from paper section 2.2.
        targets[batch_id, index_x, index_y, np.array(index_z) * 7] = (
            batch_gt_boxes_3d[batch_id][id_pos_gt, 0] - anchors_reshaped[id_pos, 0]
        ) / anchors_diag[id_pos] 
        #delta y from paper section 2.2.
        targets[batch_id, index_x, index_y, np.array(index_z) * 7 + 1] = (
            batch_gt_boxes_3d[batch_id][id_pos_gt, 1] - anchors_reshaped[id_pos, 1]
        ) / anchors_diag[id_pos]
        # delta z from paper section 2.2 
        targets[batch_id, index_x, index_y, np.array(index_z) * 7 + 2] = (
            batch_gt_boxes_3d[batch_id][id_pos_gt, 2] - anchors_reshaped[id_pos, 2]
        ) / cfg.OBJECT.ANCHOR_H 
        # delta l
        delta_l = np.log(
            batch_gt_boxes_3d[batch_id][id_pos_gt, 3] / anchors_reshaped[id_pos, 3]
        )
        targets[batch_id, index_x, index_y, np.array(index_z) * 7 + 3] = delta_l
        # delta w
        delta_w = np.log(
            batch_gt_boxes_3d[batch_id][id_pos_gt, 4] / anchors_reshaped[id_pos, 4]
        )
        targets[batch_id, index_x, index_y, index_z * 7 + 4] = delta_w 
        # delta h 
        targets[batch_id, index_x, index_y, np.array(index_z) * 7 + 5] = np.log(
            batch_gt_boxes_3d[batch_id][id_pos_gt, 5] / anchors_reshaped[id_pos, 5]
        )
        # delta theta
        targets[batch_id, index_x, index_y, np.array(index_z) * 7 + 6] = (
            batch_gt_boxes_3d[batch_id][id_pos_gt, 6] - anchors_reshaped[id_pos, 6]
        )

        index_x, index_y, index_z = np.unravel_index(id_neg, (*feature_map_shape, 2))
        neg_equal_one[batch_id, index_x, index_y, index_z] = 1
        # avoid a box to be positive and negative at the same time  
        # index_x, index_z, index_z = np.unravel_index(id_max, (*feature_map_shape, 2))
        # print(f"id max shape: {id_max.shape}")
        # print(f"id pos shape: {id_pos.shape}") 
        # print(f"index x : {index_x}")
        # print(f"index y : {index_y}")
        # print(f"index z : {index_z}")
        # neg_equal_one[batch_id, index_x, index_y, index_z] = 0 

    return pos_equal_one, neg_equal_one, targets 


def deltas_to_boxes_3d(deltas, anchors):
    anchors_reshaped = anchors.reshape(-1, 7)
    deltas = deltas.reshape(deltas.shape[0], -1, 7)
    anchors_d = np.sqrt(
        anchors_reshaped[:, 4] ** 2 + anchors_reshaped[:, 5] ** 2
    ) 
    boxes_3d = np.zeros_like(deltas)
    
    boxes_3d[..., [0, 1]] = deltas[..., [0, 1]] * anchors_d[:, np.newaxis] + anchors_reshaped[..., [0, 1]]
    boxes_3d[..., [2]] = deltas[..., [2]] * cfg.OBJECT.ANCHOR_H + anchors_reshaped[..., [2]]
    boxes_3d[..., [3, 4, 5]] = np.exp(deltas[..., [3, 4, 5]]) * anchors_reshaped[..., [3, 4, 5]]
    boxes_3d[..., 6] = deltas[..., 6] + anchors_reshaped[..., 6] 
    
    return boxes_3d


def nms(boxes, scores, overlap: float = 0.5, top_k: int = 200):
    # Original author: Francisco Massa:
    # https://github.com/fmassa/object-detection.torch
    # Ported to PyTorch by Max deGroot (02/01/2017)      
    
    keep = scores.new(scores.size(0)).zero_().long()
    count = 0 

    if boxes.numel() == 0:
        return keep, count
    
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    area = torch.mul(x2 - x1, y2 - y1)
    v, idx = scores.sort(0)  # sort in ascending order
    # I = I[v >= 0.01]
    idx = idx[-top_k:]  # indices of the top-k largest vals
    xx1 = boxes.new()
    yy1 = boxes.new()
    xx2 = boxes.new()
    yy2 = boxes.new()
    w = boxes.new()
    h = boxes.new()

    # keep = torch.Tensor()
    while idx.numel() > 0:
        i = idx[-1]  # index of current largest val
        # keep.append(i)
        keep[count] = i
        count += 1
        if idx.size(0) == 1:
            break
        idx = idx[:-1]  # remove kept element from view
        # load bboxes of next highest vals
        
        torch.index_select(x1, 0, idx, out = xx1)
        torch.index_select(y1, 0, idx, out = yy1)
        torch.index_select(x2, 0, idx, out = xx2)
        torch.index_select(y2, 0, idx, out = yy2)
        # store element-wise max with next highest score
        xx1 = torch.clamp(xx1, min = x1[i])
        yy1 = torch.clamp(yy1, min = y1[i])
        xx2 = torch.clamp(xx2, max = x2[i])
        yy2 = torch.clamp(yy2, max = y2[i])
        w.resize_as_(xx2)
        h.resize_as_(yy2)
        w = xx2 - xx1
        h = yy2 - yy1
        # check sizes of xx1 and xx2.. after each iteration
        w = torch.clamp(w, min = 0.0)
        h = torch.clamp(h, min = 0.0)
        inter = w * h
        # IoU = i / (area(a) + area(b) - i)
        rem_areas = torch.index_select(area, 0, idx)  # load remaining areas)
        union = (rem_areas - inter) + area[i]
        IoU = inter / union  # store result in iou
        # keep only elements with an IoU <= overlap
        idx = idx[IoU.le(overlap)]

    return keep, count 


def load_calib(calib_path: str):
    lines = open(calib_path).readlines() 
    lines = [line.split()[1:] for line in lines][:-1]

    P = np.array(lines[2]).reshape(3, 4)
    P = np.concatenate((P, np.array([[0, 0, 0, 0]])), 0)

    Tr_velo_to_cam = np.array(lines[5]).reshape(3, 4)
    Tr_velo_to_cam = np.concatenate([Tr_velo_to_cam, np.array([0, 0, 0, 1]).reshape(1, 4)], 0)

    R_cam_to_rect = np.eye(4)
    R_cam_to_rect[:3, :3] = np.array(lines[4][:9]).reshape(3, 3)

    P = P.astype('float32')
    Tr_velo_to_cam = Tr_velo_to_cam.astype('float32')
    R_cam_to_rect = R_cam_to_rect.astype('float32')

    return P, Tr_velo_to_cam, R_cam_to_rect


def center_to_corner_box3d(
    boxes_center, 
    coordinate='lidar', 
    T_VELO_2_CAM=None, 
    R_RECT_0=None,
):
    # (N, 7) -> (N, 8, 3)
    N = boxes_center.shape[0]
    ret = np.zeros((N, 8, 3), dtype = np.float32)

    if coordinate == 'camera':
        boxes_center = camera_to_lidar_box(boxes_center, T_VELO_2_CAM, R_RECT_0)

    for i in range(N):
        box = boxes_center[i]
        translation = box[0:3]
        size = box[3:6]
        rotation = [0, 0, box[-1]]

        h, w, l = size[0], size[1], size[2]
        trackletBox = np.array([  # in velodyne coordinates around zero point and without orientation yet
            [-l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2], \
            [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2], \
            [0, 0, 0, 0, h, h, h, h]])

        # re-create 3D bounding box in velodyne coordinate system
        yaw = rotation[2]
        rotMat = np.array([
            [np.cos(yaw), -np.sin(yaw), 0.0],
            [np.sin(yaw), np.cos(yaw), 0.0],
            [0.0, 0.0, 1.0]])
        cornerPosInVelo = np.dot(rotMat, trackletBox) + \
            np.tile(translation, (8, 1)).T
        box3d = cornerPosInVelo.transpose()
        ret[i] = box3d

    if coordinate == 'camera':
        for idx in range(len(ret)):
            ret[idx] = lidar_to_camera_point(ret[idx], T_VELO_2_CAM, R_RECT_0)

    return ret



def lidar_box3d_to_camera_box(
    boxes_3d, cal_projection=False, 
    P2=None, T_VELO_2_CAM=None, R_RECT_0=None):
    
    num = len(boxes_3d)
    boxes2d = np.zeros((num, 4), dtype = np.int32)
    projections = np.zeros((num, 8, 2), dtype = np.float32)

    lidar_boxes3d_corner = center_to_corner_box3d(
        boxes_3d, 
        coordinate = 'lidar', 
        T_VELO_2_CAM = T_VELO_2_CAM, 
        R_RECT_0 = R_RECT_0,
    )
    if type(P2) == type(None):
        P2 = np.array(cfg.CALIB.MATRIX_P2)

    for n in range(num):
        box3d = lidar_boxes3d_corner[n]
        box3d = lidar_to_camera_point(box3d, T_VELO_2_CAM, R_RECT_0)
        points = np.hstack((box3d, np.ones((8, 1)))).T  # (8, 4) -> (4, 8)
        points = np.matmul(P2, points).T

        points = np.nan_to_num(points)

        points[:, 0] /= points[:, 2]
        points[:, 1] /= points[:, 2]

        projections[n] = points[:, 0:2]
        minx = 0 if np.isnan(np.min(points[:, 0])) else int(np.min(points[:, 0]))
        maxx = 0 if np.isnan(np.max(points[:, 0])) else int(np.max(points[:, 0]))
        miny = 0 if np.isnan(np.min(points[:, 1])) else int(np.min(points[:, 1]))
        maxy = 0 if np.isnan(np.max(points[:, 1])) else int(np.max(points[:, 1]))

        boxes2d[n, :] = minx, miny, maxx, maxy

    return projections if cal_projection else boxes2d 


def draw_lidar_box_3d_on_image(
    img, 
    boxes_3d, 
    gt_boxes_3d=np.array([]), 
    color=(0, 255, 255), 
    gt_color=(255,0,255), 
    thickness=1, 
    P2=None, 
    T_VELO_2_CAM=None, 
    R_RECT_0=None,
):
    img = img.copy()
    projections = lidar_box3d_to_camera_box(
        boxes_3d, 
        cal_projection = True, 
        P2 = P2, 
        T_VELO_2_CAM = T_VELO_2_CAM, 
        R_RECT_0 = R_RECT_0,
    )
    # TODO here is the problem 
    gt_projections = lidar_box3d_to_camera_box(
        gt_boxes_3d, 
        cal_projection = True, 
        P2 = P2, 
        T_VELO_2_CAM = T_VELO_2_CAM, 
        R_RECT_0 = R_RECT_0,
    )
    img = np.array(img)
    img = img.astype(np.uint8)
    # Draw projections
    for qs in projections:
        for k in range(0, 4):
            i, j = k, (k + 1) % 4
            cv2.line(img, (int(qs[i, 0]), int(qs[i, 1])), (int(qs[j, 0]),
                                                 int(qs[j, 1])), color, thickness, cv2.LINE_AA)

            i, j = k + 4, (k + 1) % 4 + 4
            cv2.line(img, (int(qs[i, 0]), int(qs[i, 1])), (int(qs[j, 0]),
                                                 int(qs[j, 1])), color, thickness, cv2.LINE_AA)

            i, j = k, k + 4
            cv2.line(img, (int(qs[i, 0]), int(qs[i, 1])), (int(qs[j, 0]),
                                                 int(qs[j, 1])), color, thickness, cv2.LINE_AA)
    # Draw gt projections
    for qs in gt_projections:
        for k in range(0, 4):
            i, j = k, (k + 1) % 4
            cv2.line(img, (int(qs[i, 0]), int(qs[i, 1])), (int(qs[j, 0]),
                                                 int(qs[j, 1])), gt_color, thickness, cv2.LINE_AA)

            i, j = k + 4, (k + 1) % 4 + 4
            cv2.line(img, (int(qs[i, 0]), int(qs[i, 1])), (int(qs[j, 0]),
                                                 int(qs[j, 1])), gt_color, thickness, cv2.LINE_AA)

            i, j = k, k + 4
            cv2.line(img, (int(qs[i, 0]), int(qs[i, 1])), (int(qs[j, 0]),
                                                 int(qs[j, 1])), gt_color, thickness, cv2.LINE_AA)

    return cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2RGB)


def lidar_to_bird_view_image(lidar, factor=1):
    # Input:
    #   lidar: (N', 4)
    # Output:
    #   birdview: (w, l, 3)
    birdview = np.zeros(
        (cfg.OBJECT.HEIGHT * factor, cfg.OBJECT.WIDTH * factor, 1))
    for point in lidar:
        x, y = point[0:2]
        if cfg.OBJECT.X_MIN < x < cfg.OBJECT.X_MAX and cfg.OBJECT.Y_MIN < y < cfg.OBJECT.Y_MAX:
            x, y = int((x - cfg.OBJECT.X_MIN) / cfg.OBJECT.X_VOXEL_SIZE *
                       factor), int((y - cfg.OBJECT.Y_MIN) / cfg.OBJECT.Y_VOXEL_SIZE * factor)
            birdview[y, x] += 1
    birdview = birdview - np.min(birdview)
    divisor = np.max(birdview) - np.min(birdview)
    # TODO: adjust this factor
    birdview = np.clip((birdview / divisor * 255) *
                       5 * factor, a_min = 0, a_max = 255)
    birdview = np.tile(birdview, 3).astype(np.uint8)

    return birdview


def draw_lidar_box_3d_on_birdview(
    birdview, boxes3d, gt_boxes3d=np.array([]), 
    color=(0,255,255), gt_color=(255,0,255), thickness=1, 
    factor=1, P2=None, T_VELO_2_CAM=None, R_RECT_0=None):

    # Input:
    #   birdview: (h, w, 3)
    #   boxes3d (N, 7) [x, y, z, h, w, l, r]
    #   scores
    #   gt_boxes3d (N, 7) [x, y, z, h, w, l, r]
    img = birdview.copy()
    img = img.astype(np.uint8)
    corner_boxes3d = center_to_corner_box3d(boxes3d, coordinate = 'lidar', T_VELO_2_CAM = T_VELO_2_CAM, R_RECT_0 = R_RECT_0)
    corner_gt_boxes3d = center_to_corner_box3d(gt_boxes3d, coordinate = 'lidar', T_VELO_2_CAM = T_VELO_2_CAM, R_RECT_0 = R_RECT_0)
    # draw gt
    for box in corner_gt_boxes3d:
        x0, y0 = lidar_to_bird_view(*box[0, 0:2], factor = factor)
        x1, y1 = lidar_to_bird_view(*box[1, 0:2], factor = factor)
        x2, y2 = lidar_to_bird_view(*box[2, 0:2], factor = factor)
        x3, y3 = lidar_to_bird_view(*box[3, 0:2], factor = factor)

        cv2.line(img, (int(x0), int(y0)), (int(x1), int(y1)),
                 gt_color, thickness, cv2.LINE_AA)
        cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)),
                 gt_color, thickness, cv2.LINE_AA)
        cv2.line(img, (int(x2), int(y2)), (int(x3), int(y3)),
                 gt_color, thickness, cv2.LINE_AA)
        cv2.line(img, (int(x3), int(y3)), (int(x0), int(y0)),
                 gt_color, thickness, cv2.LINE_AA)

    # draw detections
    for box in corner_boxes3d:
        x0, y0 = lidar_to_bird_view(*box[0, 0:2], factor = factor)
        x1, y1 = lidar_to_bird_view(*box[1, 0:2], factor = factor)
        x2, y2 = lidar_to_bird_view(*box[2, 0:2], factor = factor)
        x3, y3 = lidar_to_bird_view(*box[3, 0:2], factor = factor)

        cv2.line(img, (int(x0), int(y0)), (int(x1), int(y1)),
                 color, thickness, cv2.LINE_AA)
        cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)),
                 color, thickness, cv2.LINE_AA)
        cv2.line(img, (int(x2), int(y2)), (int(x3), int(y3)),
                 color, thickness, cv2.LINE_AA)
        cv2.line(img, (int(x3), int(y3)), (int(x0), int(y0)),
                 color, thickness, cv2.LINE_AA)

    return cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2RGB)


def lidar_to_bird_view(x, y, factor=1):
    # using the cfg.INPUT_XXX
    a = (x - cfg.OBJECT.X_MIN) / cfg.OBJECT.X_VOXEL_SIZE * factor
    b = (y - cfg.OBJECT.Y_MIN) / cfg.OBJECT.Y_VOXEL_SIZE * factor
    a = np.clip(a, a_max = (cfg.OBJECT.X_MAX - cfg.OBJECT.X_MIN) / cfg.OBJECT.X_VOXEL_SIZE * factor, a_min = 0)
    b = np.clip(b, a_max = (cfg.OBJECT.Y_MAX - cfg.OBJECT.Y_MIN) / cfg.OBJECT.Y_VOXEL_SIZE * factor, a_min = 0)

    return a, b 

def colorize(value, factor=1, vmin=None, vmax=None):
    # normalize
    value = np.sum(value, axis = -1)
    vmin = np.min(value) if vmin is None else vmin
    vmax = np.max(value) if vmax is None else vmax
    value = (value - vmin) / (vmax - vmin)  # vmin..vmax

    value = (value * 255).astype(np.uint8)
    value = cv2.applyColorMap(value, cv2.COLORMAP_JET)
    value = cv2.cvtColor(value, cv2.COLOR_BGR2RGB)
    x, y, _ = value.shape
    value = cv2.resize(value, (y * factor, x * factor))

    return value



def box3d_to_label(batch_box3d, batch_cls, batch_score = [], coordinate='camera', P2 = None, T_VELO_2_CAM = None, R_RECT_0 = None):
    # Input:
    #   (N, N', 7) x y z h w l r
    #   (N, N')
    #   cls: (N, N') 'Car' or 'Pedestrain' or 'Cyclist'
    #   coordinate(input): 'camera' or 'lidar'
    # Output:
    #   label: (N, N') N batches and N lines
    batch_label = []
    if batch_score:
        template = '{} ' + ' '.join(['{:.4f}' for i in range(15)]) + '\n'
        for boxes, scores, clses in zip(batch_box3d, batch_score, batch_cls):
            label = []
            for box, score, cls in zip(boxes, scores, clses):
                if coordinate == 'camera':
                    box3d = box
                    box2d = lidar_box3d_to_camera_box(
                        camera_to_lidar_box(box[np.newaxis, :].astype(np.float32), T_VELO_2_CAM, R_RECT_0), cal_projection = False,
                        P2 = P2, T_VELO_2_CAM = T_VELO_2_CAM, R_RECT_0 = R_RECT_0)[0]
                else:
                    box3d = lidar_to_camera_box(
                        box[np.newaxis, :].astype(np.float32), T_VELO_2_CAM, R_RECT_0)[0]
                    box2d = lidar_box3d_to_camera_box(
                        box[np.newaxis, :].astype(np.float32), cal_projection = False, P2 = P2, T_VELO_2_CAM = T_VELO_2_CAM, R_RECT_0 = R_RECT_0)[0]
                x, y, z, h, w, l, r = box3d
                box3d = [h, w, l, x, y, z, r]
                label.append(template.format(
                    cls, 0, 0, 0, *box2d, *box3d, float(score)))
            batch_label.append(label)
    else:
        template = '{} ' + ' '.join(['{:.4f}' for i in range(14)]) + '\n'
        for boxes, clses in zip(batch_box3d, batch_cls):
            label = []
            for box, cls in zip(boxes, clses):
                if coordinate == 'camera':
                    box3d = box
                    box2d = lidar_box3d_to_camera_box(
                        camera_to_lidar_box(box[np.newaxis, :].astype(np.float32), T_VELO_2_CAM, R_RECT_0),
                        cal_projection = False,  P2 = P2, T_VELO_2_CAM = T_VELO_2_CAM, R_RECT_0 = R_RECT_0)[0]
                else:
                    box3d = lidar_to_camera_box(
                        box[np.newaxis, :].astype(np.float32), T_VELO_2_CAM, R_RECT_0)[0]
                    box2d = lidar_box3d_to_camera_box(
                        box[np.newaxis, :].astype(np.float32), cal_projection = False, P2 = P2, T_VELO_2_CAM = T_VELO_2_CAM, R_RECT_0 = R_RECT_0)[0]
                x, y, z, h, w, l, r = box3d
                box3d = [h, w, l, x, y, z, r]
                label.append(template.format(cls, 0, 0, 0, *box2d, *box3d))
            batch_label.append(label)

    return np.array(batch_label)


def lidar_to_camera_box(boxes, T_VELO_2_CAM = None, R_RECT_0 = None):
    # (N, 7) -> (N, 7) x,y,z,h,w,l,r
    ret = []
    for box in boxes:
        x, y, z, h, w, l, rz = box
        (x, y, z), h, w, l, ry = lidar_to_camera(
            x, y, z, T_VELO_2_CAM, R_RECT_0), h, w, l, -rz - np.pi / 2
        ry = angle_in_limit(ry)
        ret.append([x, y, z, h, w, l, ry])

    return np.array(ret).reshape(-1, 7)



def test():
    pcl_path = "/data/kitti/3d_vision/training/velodyne/000009.bin"
    pcl = np.fromfile(pcl_path, dtype=np.float32).reshape(-1, 4)
    pcl_preproc = pcl_to_voxels(pcl, "Car", True)
    from config import get_cfg_defaults
    cfg = get_cfg_defaults()
    anchors = generate_anchors(cfg)


if __name__ == "__main__":
    test()