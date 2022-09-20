from multiprocessing.sharedctypes import Value
from re import A, I
import numpy as np 


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



def generate_anchors(cfg):
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


def camera_to_lidar(x, y, z, cfg, T_VELO_2_CAM=None, R_RECT_0=None):
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


def camera_to_lidar_box(boxes, cfg, T_VELO_2_CAM=None, R_RECT_0=None):
    """Transform boxes in camera coordinates to lidar coordinates."""  
    lidar_boxes = []
    for box in boxes: 
        x, y, z, h, w, l, ry = box 
        (x, y, z), h, w, l, rz = camera_to_lidar(x, y, z, cfg, T_VELO_2_CAM, R_RECT_0), \
            h, w, l, -ry - np.pi / 2 
        rz = angle_in_limit(rz)
        lidar_boxes.append([x, y, z, h, w, l, rz])
    
    return np.array(lidar_boxes).reshape(-1, 7)



def label_to_gt_box_3d(
    labels, 
    cls_name: str, 
    coordinate: str, 
    cfg,
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
                np.array(boxes3d_label), cfg, T_VELO_2_CAM, R_RECT_0)
        
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
    cfg, 
    coordinate='lidar', 
    T_VELO_2_CAM=None, 
    R_RECT_0=None,
):  
    # (N, 5) -> (N, 4, 2)
    N = boxes_center.shape[0]
    boxes_3d_center = np.zeros((N, 7))
    boxes_3d_center[:, [0, 1, 4, 5, 6]] = boxes_center
    boxes_3d_corner = center_to_corner_box_3d(
        boxes_3d_center, coordinate, cfg, T_VELO_2_CAM, R_RECT_0
    )

    return boxes_3d_corner[:, 0:4, 0:2] 


def lidar_to_camera_point(boxes, cfg, T_VELO_2_CAM=None, R_RECT_0=None):
    # (N, 7) -> (N, 7) = x, y, z, h, w, l, r 
    camera_boxes = []

    for box in boxes:
        x, y, z, h, w, l, rz = box 
        (x, y, z), h, w, l, ry = lidar_to_camera(
            x, y, z, T_VELO_2_CAM, R_RECT_0, cfg
        ), h, w, l, -rz - np.pi / 2 
        ry = angle_in_limit(ry)
        camera_boxes.append([x, y, z, h, w, l, ry])
    
    return np.array(camera_boxes).reshape(-1, 7)


def lidar_to_camera(x, y, z, cfg, T_VELO_2_CAM=None, R_RECT_0=None):
    """Transform lidar coordinates to camera coordinates."""    
    if not T_VELO_2_CAM:
        T_VELO_2_CAM = np.array(cfg.CALIB.T_VELO_2_CAM)

    if not R_RECT_0:
        R_RECT_0 = np.array(cfg.CALIB.R_RECT_0) 

    # homogenuous point
    point = np.array([x, y, z, 1]) 
    point = np.matmul(T_VELO_2_CAM, point)
    point = np.matmul(R_RECT_0, point)
    point = point[:, :3]

    return tuple(point) 


def center_to_corner_box_3d(
    boxes_center, 
    coordinate, 
    cfg, 
    T_VELO_2_CAM=None, 
    R_RECT_0=None,
):
    # (N, 7) -> (N, 8, 3) = 8 corners in 3d
    N = boxes_center.shape[0]
    corner_box_3d = np.zeros((N, 8, 3), dtype=np.float32)

    if coordinate == 'camera':
        boxes_center = camera_to_lidar_box(boxes_center, cfg, T_VELO_2_CAM, R_RECT_0)
    
    for i in range(N):
        box = boxes_center[i]
        translation = box[:3]
        size = box[4:6]
        rotation = [0, 0, box[-1]]

        h, w, l = box[0], box[1], box[2]

        # in velodyne coordinates around zero point and without orientation
        tracklet_box = np.array([
            [-l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2], \
            [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2], \
            [0, 0, 0, 0, h, h, h, h]
        ])

        # re-create 3D bounding box in velodyne coordinate system 
        yaw = box[-1] 
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
                corner_box_3d[idx], cfg, T_VELO_2_CAM, R_RECT_0)

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
    cfg,
    cls_name='Car', 
    coordinate='lidar', 
):
    batch_size = labels.shape[0]
    batch_gt_boxes_3d = label_to_gt_box_3d(labels, cls_name, coordinate, cfg)

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
        print(f"Anchors 2d shape: {anchors_standup_2d.shape}") 
        print(anchors_standup_2d[0, :]) 
        gt_standup_2d = corner_to_standup_box2d(
            center_to_corner_box_2d(
                batch_gt_boxes_3d[batch_id][:, [0, 1, 4, 5, 6]], cfg, coordinate
            )
        )        
        print(f"Groundtruth 2d boxes: {gt_standup_2d.shape}")
        if gt_standup_2d.shape[0] > 0: 
            print(gt_standup_2d[0, :]) 
        # calculate iou between anchors and gt boxes
        iou = bbox_iou(
            np.ascontiguousarray(anchors_standup_2d).astype(np.float32),
            np.ascontiguousarray(gt_standup_2d).astype(np.float32),
        )
        print(f"IOU shape: {iou.shape}")
        print(iou[0,:])
        # find anchor with highest iou 


    return None  




def test():
    pcl_path = "/data/kitti/3d_vision/training/velodyne/000009.bin"
    pcl = np.fromfile(pcl_path, dtype=np.float32).reshape(-1, 4)
    pcl_preproc = pcl_to_voxels(pcl, "Car", True)
    from config import get_cfg_defaults
    cfg = get_cfg_defaults()
    anchors = generate_anchors(cfg)


if __name__ == "__main__":
    test()