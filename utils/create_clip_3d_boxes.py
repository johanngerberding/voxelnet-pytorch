import os 
import argparse
import numpy as np 
import parse_tracklet_xml
import cv2
from moviepy.editor import ImageSequenceClip

from visualize import load_dataset

colors = {
    'Car': 'b',
    'Tram': 'r',
    'Cyclist': 'g',
    'Van': 'c',
    'Truck': 'm',
    'Pedestrian': 'y',
    'Sitter': 'k',
}


def cart2homogen(points3D):
    n = points3D.shape[0]
    homogen_points3D = np.hstack((points3D, np.ones((n, 1))))
    return homogen_points3D


def load_Rt_velo_to_cam(path: str):
    """Load rotation and translation parameters from Lidar to Camera.

    Args:
        path (str): Path to calibration file. 

    Returns:
        np.ndarray: Translation and Rotation Matrix 
    """
    lines = []
    with open(path, 'r') as fp: 
        for line in fp.readlines():
            lines.append(line)
    lines = lines[1:3]
    # rotation matrix
    velo_to_cam_rotations = lines[0].split(' ')[1:]
    velo_to_cam_rotations = np.array([float(el.strip()) 
                                      for el in velo_to_cam_rotations]).reshape((3,3))
    # translation 
    velo_to_cam_translation = lines[1].split(' ')[1:]
    velo_to_cam_translation = np.array([float(el.strip()) 
                                        for el in velo_to_cam_translation]).reshape((3,1))
    Rt_velo_to_cam = np.hstack((velo_to_cam_rotations, velo_to_cam_translation))
    Rt_velo_to_cam = np.vstack((Rt_velo_to_cam, np.array([0, 0, 0, 1])))
    
    return Rt_velo_to_cam


def tracklets_velo2cam_2d(
    tracklets, 
    dataset, 
    velo_to_cam_path: str, 
    n_frames: int, 
    camera_id=2,
):
    tracklets_2d_boxes = {}
    tracklets_2d_boxes_types = {}
    
    for i in range(n_frames):
        tracklets_2d_boxes[i] = []
        tracklets_2d_boxes_types[i] = []
        
    R_rect_00 = dataset.calib.R_rect_00
    
    if camera_id == 0:
        P_rect = dataset.calib.P_rect_00
    elif camera_id == 1:
        P_rect = dataset.calib.P_rect_10
    elif camera_id == 2:
        P_rect = dataset.calib.P_rect_20
    elif camera_id == 3:
        P_rect = dataset.calib.P_rect_30
        
    Rt_velo_to_cam = load_Rt_velo_to_cam(velo_to_cam_path)
    
    for i, tracklet in enumerate(tracklets):
        
        h, w, l = tracklet.size
        # in velodyne coordinates around zero point and without orientation yet
        trackletBox = np.array([
            [-l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2],
            [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2],
            [0.0, 0.0, 0.0, 0.0, h, h, h, h]
        ])
        
        for (translation, rotation, state, occlusion, 
             truncation, amtOcclusion, amtBorders, 
             absoluteFrameNumber) in tracklet:
            
            if truncation not in [0, 1]: # 0 -> in image, 1 -> truncated
                continue 
                
            yaw = rotation[2]
            assert np.abs(rotation[:2]).sum() == 0, 'object rotations other than yaw given!'
            # rotation around z-axis 
            rotMat = np.array([
                [np.cos(yaw), -np.sin(yaw), 0.0],
                [np.sin(yaw), np.cos(yaw), 0.0],
                [0.0, 0.0, 1.0]
            ])
            
            cornerPosInVelo = np.dot(rotMat, trackletBox) + np.tile(translation, (8, 1)).T
            # transform to homogenuous coordinates
            cornerPosInVeloHomo = cart2homogen(cornerPosInVelo.T)
            # apply rotation and translation 
            cornerPosInCameraHomo = Rt_velo_to_cam @ cornerPosInVeloHomo.T
            cornerPosInCameraHomo = R_rect_00 @ cornerPosInCameraHomo
            cornerPosInCameraHomo = P_rect @ cornerPosInCameraHomo
            
            cornerPosInCameraHomo = cornerPosInCameraHomo.T
            # transform to pixel coordinates
            cornerPosInCameraHomo[:,0] /= cornerPosInCameraHomo[:,2]
            cornerPosInCameraHomo[:,1] /= cornerPosInCameraHomo[:,2]
            cornerPosInCamera = cornerPosInCameraHomo[:,:2]
            
            tracklets_2d_boxes[absoluteFrameNumber] = tracklets_2d_boxes[absoluteFrameNumber] + [cornerPosInCamera]
            tracklets_2d_boxes_types[absoluteFrameNumber] = tracklets_2d_boxes_types[absoluteFrameNumber] + [tracklet.objectType]
    
    return tracklets_2d_boxes, tracklets_2d_boxes_types


def draw_3d_boxes_img(
    dataset, 
    frame_id, 
    tracklets_2d_boxes, 
    color=(128, 255, 128), 
    thickness=1,
):
    
    connections = [[0, 1], [1, 2], [2, 3], [3, 0], 
                [4, 5], [5, 6], [6, 7], [7, 4], 
                [0, 4], [1, 5], [2, 6], [3, 7], 
                [0, 5], [1, 4]] 
    # Point connections: 
    # 0 -> right, back, down
    # 1 -> left, back, down
    # 2 -> left, front, down
    # 3 -> right, front, down 
    # 4 -> right, back, up
    # 5 -> left, back, up
    # 6 -> left, front, up
    # 7 -> right, front, up
    
    cam2_gen = np.asarray(dataset.get_cam2(frame_id))
    boxes = tracklets_2d_boxes.get(frame_id)
    
    for box in boxes:
        for connection in connections:
            start = box[connection[0]]
            end = box[connection[1]]
            start = (int(start[0]), int(start[1]))
            end = (int(end[0]), int(end[1]))
            cam2_gen = cv2.line(cam2_gen, start, end, color, thickness)
    
    return cam2_gen


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir', type=str, default='/home/johann/dev/kitti-data-exploration/data')
    parser.add_argument('--date', type=str, default='2011_09_26')
    parser.add_argument('--drive', type=str, default='0001')
    parser.add_argument('--format', type=str, default='gif') 
    parser.add_argument('--out', type=str, default='3d_bboxes.mp4') 
    parser.add_argument('--fps', type=int, default=10)    
    args = parser.parse_args()

    if not os.path.isdir(args.base_dir):
        raise ValueError("Please provide a valid data base directory.") 
    
    dataset = load_dataset(args.base_dir, args.date, args.drive)
    labels = os.path.join(
        args.base_dir, 
        args.date, 
        f"{args.date}_drive_{args.drive}_sync/tracklet_labels.xml",
    )
    if not os.path.isfile(labels):
        raise FileNotFoundError(f"Could not find the labels: {labels}") 

    velo_to_cam_calib = os.path.join(
        args.base_dir, 
        args.date, 
        "calib_velo_to_cam.txt",
    )
    if not os.path.isfile(velo_to_cam_calib):
        raise FileNotFoundError(f"Could not find the calibration file: {velo_to_cam_calib}")
    
    tracklets = parse_tracklet_xml.parseXML(labels)

    tracklets_2d_boxes, tracklets_2d_boxes_types = tracklets_velo2cam_2d(
        tracklets, 
        dataset, 
        velo_to_cam_calib, 
        n_frames=len(list(dataset.cam2)), 
        camera_id=2,
    )

    imgs = [
        draw_3d_boxes_img(dataset, i, tracklets_2d_boxes) 
        for i in range(len(list(dataset.cam2)))
    ]

    if args.format == 'mp4':
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        size = (imgs[0].shape[1], imgs[0].shape[0]) 
        print(size) 
        out = cv2.VideoWriter(
            args.out, 
            fourcc, 
            args.fps, 
            size,
        )

        for img in imgs:
            out.write(img)
            cv2.imshow("Frame", img) 
            key = cv2.waitKey(1) & 0xFF 
            if key == ord("q"):
                break 
        
        out.release()
        cv2.destroyAllWindows()

    elif args.format == 'gif':
        if not args.out.endswith('gif'):
            outpath = args.out[:-4] + ".gif"
        else: 
            outpath = args.out
        clip = ImageSequenceClip(imgs, fps=args.fps)
        clip.write_gif(outpath, fps=args.fps)
    else: 
        raise ValueError("Choose between mp4 and gif as format.")


if __name__ == "__main__":
    main()