import argparse
import os 
import shutil 
from moviepy.editor import ImageSequenceClip 
from visualize import (
    draw_box, 
    load_dataset, 
    load_tracklets_for_frames, 
    axes_limits, 
    colors,
)
import matplotlib.pyplot as plt 
import numpy as np


def draw_3d_plot(frame, dataset, tracklet_rects, tracklet_types, temp, points=0.2):

    dataset_velo = list(dataset.velo)
    f = plt.figure(figsize=(12, 8))
    axis = f.add_subplot(111, projection='3d', xticks=[], yticks=[], zticks=[])

    points_step = int(1. / points)
    point_size = 0.01 * (1. / points)
    velo_range = range(0, dataset_velo[frame].shape[0], points_step)
    velo_frame = dataset_velo[frame][velo_range, :]
    axis.scatter(*np.transpose(velo_frame[:, [0, 1, 2]]), s=point_size, c=velo_frame[:, 3], cmap='gray')
    axis.set_xlim3d(*axes_limits[0])
    axis.set_ylim3d(*axes_limits[1])
    axis.set_zlim3d(*axes_limits[2])
    
    for t_rects, t_type in zip(tracklet_rects[frame], tracklet_types[frame]):
        draw_box(axis, t_rects, axes=[0, 1, 2], color=colors[t_type])
    
    
    filename = '{0}/frame_{1:0>4}.png'.format(temp, frame)
    plt.savefig(filename)
    plt.close(f)
    
    return filename


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "data", 
        type=str, 
        help="Path to data directory",
    )
    parser.add_argument(
        "out", 
        type=str, 
        help="Output directory for the gif-file.", 
    )
    parser.add_argument(
        "--date", 
        type=str, 
        help="Date of the drive.", 
        default='2011_09_26',
    )
    parser.add_argument(
        "--drive", 
        type=str, 
        help="ID of the drive.", 
        default='0001',
    )
    parser.add_argument(
        "--fps", 
        type=int, 
        help="Gif FPS",
        default=5, 
    )
    args = parser.parse_args()

    dataset = load_dataset(args.data, args.date, args.drive)

    tracklet_rects, tracklet_types = load_tracklets_for_frames(
        len(list(dataset.velo)), 
        '{}/{}/{}_drive_{}_sync/tracklet_labels.xml'.format(
            args.data, args.date, args.date, args.drive), 
    )

    frames = []
    n_frames = len(list(dataset.velo))

    temp = os.path.join(os.path.split(os.path.abspath(__file__))[0], 'temp')

    if os.path.isdir(temp):
        shutil.rmtree(temp)
    os.makedirs(temp)

    print('Preparing animation frames...')
    for i in range(n_frames):
        filename = draw_3d_plot(i, dataset, tracklet_rects, tracklet_types, temp)
        frames += [filename]
    print('...Animation frames ready.')

    clip = ImageSequenceClip(frames, fps=args.fps)
    clip.write_gif('{}/pcl_data.gif'.format(args.out), fps=args.fps)

    shutil.rmtree(temp)


if __name__ == "__main__":
    main()
