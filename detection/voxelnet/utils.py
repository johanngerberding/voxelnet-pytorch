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


"""
def test():
    pcl_path = "/data/kitti/3d_vision/training/velodyne/000009.bin"
    pcl = np.fromfile(pcl_path, dtype=np.float32).reshape(-1, 4)
    pcl_preproc = pcl_to_voxels(pcl, "Car", True)


if __name__ == "__main__":
    test()
"""