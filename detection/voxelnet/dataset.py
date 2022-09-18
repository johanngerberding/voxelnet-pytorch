import os 
import glob 
import cv2
import numpy as np
import torch 
from torch.utils import data 

from utils import pcl_to_voxels 


class KITTIDataset(data.Dataset):
    def __init__(self, data_dir: str, shuffle=True, test=False):
        self.data_dir = data_dir
        self.shuffle = shuffle 
        self.test = test 
        
        self.images = glob.glob(os.path.join(self.data_dir, "image_2")+ "/*.png")
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
        img = cv2.imread(self.images[index]) 
        pcl = np.fromfile(self.pcls[index], dtype=np.float32).reshape(-1, 4)
        
        if not self.test: 
            labels = [line for line in open(self.labels[index], 'r').readlines()]
        else: 
            labels = []

        voxels = pcl_to_voxels(pcl, 'Car', False) 

        return img, pcl, labels, voxels 


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
    rgb = [p[0] for p in parts] 
    raw_lidar = [p[1] for p in parts]
    label = [p[2] for p in parts] 
    voxel = [p[3] for p in parts] 

    voxel_features, voxel_numbers, voxel_coordinates = prepare_voxel(voxel) 
    
    outs = (
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