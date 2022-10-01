import os 
import argparse
import torch 
import glob 
import random
import numpy as np

from config import get_cfg_defaults 
from utils import pcl_to_voxels, deltas_to_boxes_3d
from dataset import prepare_voxel
from model import filter_boxes


def main():
    parser = argparse.ArgumentParser() 
    parser.add_argument(
        "--model", 
        type=str, 
        default="exps/2022-09-30-000/checkpoints/best.pth",
    )
    parser.add_argument(
        "--pcl", 
        type=str, 
        default="/data/kitti/3d_vision/data/MD_KITTI/validation/velodyne",
    )
    parser.add_argument(
        "--out", 
        type=str, 
        default="out",
    )
    args = parser.parse_args()

    cfg = get_cfg_defaults() 
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    os.makedirs(args.out, exist_ok=True)
    # load the model 
    model = torch.load(args.model).to(device)
    model.train(False) 
    print(model)

    # load a pointcloud 
    if os.path.isdir(args.pcl):
        pcls = glob.glob(args.pcl + "/*.bin")
        pcl = np.fromfile(random.choice(pcls), dtype=np.float32).reshape(-1, 4)
    elif os.path.isfile(args.pcl):
        pcl = np.fromfile(args.pcl, dtype=np.float32).reshape(-1, 4) 
    else: 
        raise ValueError("Please provide a path to an existing pointcloud file.")

    voxels = pcl_to_voxels(pcl, 'Car', False)
    features, numbers, coordinates = prepare_voxel([voxels]) 
    
    voxel_features = [torch.from_numpy(f).to(device) for f in features]
    voxel_coordinates = [torch.from_numpy(c).to(device) for c in coordinates]

    with torch.no_grad():
        features = model.feature_net(voxel_features, voxel_coordinates)
        probs, deltas = model.middle_rpn(features)
        bs = probs.shape[0] 
        probs = probs.cpu().detach().numpy()
        deltas = deltas.cpu().detach().numpy()
        
        batch_boxes_3d = deltas_to_boxes_3d(deltas, model.anchors)
        batch_boxes_2d = batch_boxes_3d[:, :, [0, 1, 4, 5, 6]] 
        batch_probs = probs.reshape((bs, -1))
        ret_box_3d, ret_score = filter_boxes(
            bs, batch_probs, batch_boxes_3d, batch_boxes_2d, device)
        
        print(ret_box_3d)
        print(ret_score)


if __name__ == "__main__":
    main()
