from concurrent.futures import process
import os 
import glob 
import cv2
import numpy as np
import torch 
from torch.utils import data 



class KITTIDataset(data.Dataset):
    def __init__(self, data_dir: str, test=False):
        self.data_dir = data_dir
        self.test = test 
        
        self.images = glob.glob(os.path.join(self.data_dir, "image_2")+ "/*.png")
        self.pcls = glob.glob(os.path.join(self.data_dir, "velodyne") + "/*.bin")
        self.labels = glob.glob(os.path.join(self.data_dir, "label_2") + "/*.txt")
        self.images = list(sorted(self.images)) 
        self.pcls = list(sorted(self.pcls)) 
        self.labels = list(sorted(self.labels)) 

        assert len(self.images) == len(self.pcls) == len(self.labels)


    
    def __getitem__(self, idx):
        img = cv2.imread(self.images[idx]) 
        pcl = np.fromfile(self.pcls[idx], dtype=np.float32)
        
        if not self.test: 
            labels = [line for line in open(self.labels[idx], 'r').readlines()]
        else: 
            labels = []

        voxels = None 

        return img, pcl, labels, voxels 

    def __len__(self):
        return len(self.images) 



def main():
    data_dir = "/data/kitti/3d_vision/training" 
    dataset = KITTIDataset(data_dir)
    print(f"Length dataset: {len(dataset)}")

    for img, pcl, labels, voxels in dataset:
        print(img.shape)
        print(pcl.shape)
        print(len(labels))
        print(voxels)
        break


if __name__ == "__main__":
    main()