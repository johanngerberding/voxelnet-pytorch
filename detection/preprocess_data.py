# Crop the data for training 
import os 
import argparse
import numpy as np 
import glob 
import shutil 
import cv2 
from PIL import Image 
# we use cam number 2 => RGB  
CAM = 2 


def load_velo_points(filename: str) -> np.ndarray:
    points = np.fromfile(filename, dtype=np.float32).reshape(-1, 4)
    return points


def load_calib(calib_dir: str) -> tuple:
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
    Tr_velo_to_cam = Tr_velo_to_cam.astype('float32')
    R_cam_to_rect = R_cam_to_rect.astype('float32')

    return P, Tr_velo_to_cam, R_cam_to_rect


def prepare_velo_points(points) -> tuple:
    pts3d = points 
    # reflectance > 0
    idxs = pts3d[:, 3] > 0
    pts3d = pts3d[idxs, :]
    pts3d[:, 3] = 1

    return pts3d.transpose(), idxs


def project_velo_to_img(points, T_cam_velo, R_rect, P_rect) -> tuple:
    # 3d points in camera reference frame  
    points_3d_cam = R_rect.dot(T_cam_velo.dot(points)) 
    # keep only points with z > 0 
    idxs = (points_3d_cam[2, :] >= 0)
    points_2d_cam = P_rect.dot(points_3d_cam[:, idxs])

    return points[:, idxs], points_2d_cam / points_2d_cam[2, :], idxs


def align_img_and_velo(img_path: str, pc_path: str, calib_path: str) -> np.ndarray:
    #img = cv2.imread(img_path)
    img = Image.open(img_path) 
    img = np.asarray(img)
    print(img.shape) 
    points = load_velo_points(pc_path)
    P, Tr_velo_to_cam, R_cam_to_rect = load_calib(calib_path)
    print(P)
    print(Tr_velo_to_cam)
    print(R_cam_to_rect)
    print("-"*25)
    points_3d, idxs = prepare_velo_points(points)
    points_3d_org = points_3d.copy()
    reflectances = points[idxs, 3]
    print(reflectances.shape)
    points_3d, points_2d_normed, idx = project_velo_to_img(points_3d, Tr_velo_to_cam, R_cam_to_rect, P)
    reflectances = reflectances[idx]
    print(points_3d.shape)
    print(points_2d_normed.shape)
    print(points_2d_normed[:, 1]) 
    print("-"*25)
    assert reflectances.shape[0] == points_3d.shape[1] == points_2d_normed.shape[1]

    rows, cols = img.shape[:2]
    print(rows)
    print(cols) 
    pts = []
    for i in range(points_2d_normed.shape[1]):
        col = int(np.round(points_2d_normed[0, i]))
        row = int(np.round(points_2d_normed[1, i]))
          
        if col < cols and row < rows and row > 0 and col > 0:
            color = img[row, col, :]
            point = [
                points_3d[0, i],
                points_3d[1, i],
                points_3d[2, i],
                reflectances[i],
                color[0],
                color[1],
                color[2],
                points_2d_normed[0, i],
                points_2d_normed[1, i],
            ]
            pts.append(point)

    pts = np.array(pts)

    return pts 
    

def main():
    parser = argparse.ArgumentParser() 
    parser.add_argument(
        '--imgs_root', 
        help="Root directory of the images", 
        type=str, 
        default="/data/kitti/3d_vision/training/image_2",
    )
    parser.add_argument(
        '--pc_root', 
        help="Root directory of the velodyne point cloud files", 
        type=str, 
        default="/data/kitti/3d_vision/training/velodyne",
    )
    parser.add_argument(
        '--calib_root', 
        help="Root directory of the calibration files", 
        type=str, 
        default="/data/kitti/3d_vision/training/calib",
    )

    args = parser.parse_args()

    # copy old velodyne points to archive folder 
    archive = os.path.join(args.pc_root, "archive") 
    if os.path.isdir(archive):
        shutil.rmtree(archive)    
    os.makedirs(archive)

    pcls = glob.glob(args.pc_root + "/*.bin")
    print(f"Found {len(pcls)} point cloud files.")
    for pcl in pcls:
        filename = os.path.split(pcl)[1]
        nfilename = os.path.join(archive, filename)
        shutil.copy(pcl, nfilename)

    print("Overwrite Point Cloud Files:")
    print("------------------------------")
    for frame in range(len(os.listdir(args.imgs_root))):
        img_path = args.imgs_root + f"/{str(frame).zfill(6)}.png"
        assert os.path.isfile(img_path)
        pcl_path = args.pc_root + f"/{str(frame).zfill(6)}.bin"
        assert os.path.isfile(pcl_path) 
        calib_path = args.calib_root + f"/{str(frame).zfill(6)}.txt"
        assert os.path.isfile(calib_path)

        points = align_img_and_velo(img_path, pcl_path, calib_path)
        print(points.shape)
        if points.shape[0] > 0:
            print(f"Good file: {frame}.bin") 
            break
        outname = args.pc_root + f"/{str(frame).zfill(6)}.bin"

        #points[:, :4].astype('float32').tofile(outname)
        print(f"Saved: {outname}")


if __name__ == "__main__":
    main()