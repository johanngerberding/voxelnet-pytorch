import numpy as np 

def pcl_to_voxels(pcl, target: str) -> dict:
    # input = (n, 4)
    # output = voxel_dict

    if target == 'Car':
        scene_size = np.array([4, 80, 70.4], dtype=np.float32)
        voxel_size = np.array([0.4, 0.2, 0.2], dtype=np.float32)
        grid_size = np.array([10, 400, 352], dtype=np.float32)
        lidar_coord = np.array([0, 40, 3], dtype=np.float32)
        max_number_points = 35  
    else: # pedestrian & cyclist 
        scene_size = np.array([4, 40, 48], dtype=np.float32)
        voxel_size = np.array([0.4, 0.2, 0.2], dtype=np.float32)
        grid_size = np.array([10, 200, 240], dtype=np.float32)
        lidar_coord = np.array([0, 20, 3], dtype=np.float32)
        max_number_points = 45
        np.random.shuffle(pcl)

    shifted_coord = pcl[:, :3] + lidar_coord
    # change pcl coordinates order (X,Y,Z) -> (Z,Y,X) 
    print(shifted_coord.shape) 
    voxel_index = np.floor(shifted_coord[:, ::-1] / voxel_size).astype(np.int)
    print(voxel_index.shape)

    bound_x = np.logical_and(
        voxel_index[:, 2] >= 0,
        voxel_index[:, 2] < grid_size[2]
    )
    print(bound_x) 
    bound_x_ = voxel_index[:, 2] >= 0 & voxel_index[:, 2] < grid_size[2]
    assert bound_x == bound_x_

    bound_y = np.logical_and(
        voxel_index[:, 1] >= 0,
        voxel_index[:, 1] < grid_size[1]
    )

    bound_z = np.logical_and(
        voxel_index[:, 0] >= 0,
        voxel_index[:, 0] < grid_size[0]
    )

    bound_box = np.logical_and(np.logical_and(bound_x, bound_y), bound_z)

    pcl = pcl[bound_box]
    voxel_index = voxel_index[bound_box]

    # [K, 3] coordinate buffer, see paper 
    coordinate_buffer = np.unique(voxel_index, axis=0)

    K = len(coordinate_buffer)
    T = max_number_points

    
