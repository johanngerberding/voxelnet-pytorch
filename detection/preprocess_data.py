# Crop the data for training 

import numpy as np 
from scipy.misc import imread 

# we use cam number 2 => RGB  
CAM = 2 


def load_velo_points(filename: str):
    points = np.fromfile(filename, dtype=np.float32).reshape(-1, 4)
    return points


def load_calib(calib_dir: str):
    # P2 * R0_rect * Tr_velo_to_cam * y
    lines = open(calib_dir).readlines()
    lines = [
        line.split()[1:] for line in lines 
    ][:-1]
    
    P = np.array(lines[CAM]).reshape(3, 4)

    Tr_velo_to_cam = np.array(lines[5]).reshape(3, 4)
    Tr_velo_to_cam = np.concatenate([
        Tr_velo_to_cam, np.array([0, 0, 0, 1]).reshape(1, 4)
    ], 0)

    R_cam_to_rect = np.eye(4)
    R_cam_to_rect[:3, :3] = np.array(lines[4][:9]).reshape(3, 3)

    P = P.astype('float32')





def main():
    pass 


if __name__ == "__main__":
    main()