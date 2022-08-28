import numpy as np 
import pykitti 
import matplotlib.pyplot as plt 

import parse_tracklet_xml  

colors = {
        'Car': 'b',
        'Tram': 'r',
        'Cyclist': 'g',
        'Van': 'c',
        'Truck': 'm',
        'Pedestrian': 'y',
        'Sitter': 'k',
    }

axes_limits = [
        [-20, 80], # X
        [-20, 20], # Y
        [-3, 10], # Z
    ]

axes_str = ['X', 'Y', 'Z']


def load_dataset(datadir, date, drive, calibrated=False, frame_range=None):
    dataset = pykitti.raw(datadir, date, drive)

    if calibrated:
        dataset.load_calib()

    np.set_printoptions(precision=4, suppress=True)
    
    print("\nDrive: {}".format(str(dataset.drive)))
    print("Frame range: {}".format(str(dataset.frames)))

    if calibrated:
        print("IMU-to-Velodyne transformation:\n{}".format(str(dataset.calib.T_velo_imu)))
        print("Gray stereo pair baseline [m]: {}".format(str(dataset.calib.b_gray)))
        print("RGB stereo pair baseline [m]: {}".format(str(dataset.calib.b_rgb)))

    return dataset


def load_tracklets_for_frames(n_frames: int, xml_path: str):
    tracklets = parse_tracklet_xml.parseXML(xml_path)

    frame_tracklets = {}
    frame_tracklet_types = {}

    for i in range(n_frames):
        frame_tracklets[i] = []
        frame_tracklet_types[i] = []

    for i, tracklet in enumerate(tracklets):
        h, w, l = tracklet.size
        # in velodyne coordinates around zero point and without orientation yet
        trackletBox = np.array([
            [-l/2, -l/2, l/2, l/2, -l/2, -l/2, l/2, l/2],
            [w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2],
            [0.0, 0.0, 0.0, 0.0, h, h, h, h],
        ])

        # loop over all the data in the tracklet 
        for translation, rotation, state, occlusion, truncation, amtOcclusion, amtBorders, absoluteFrameNumber in tracklet:
            # check if the object is in the image 
            if truncation not in (parse_tracklet_xml.TRUNC_IN_IMAGE, parse_tracklet_xml.TRUNC_TRUNCATED):
                continue 
            # re-create 3D bounding box in velodyne coordinate system
            yaw = rotation[2]
            assert np.abs(rotation[:2]).sum() == 0, "Object rotations other than yaw given"

            rotMat = np.array([
                [np.cos(yaw), -np.sin(yaw), 0.0],
                [np.sin(yaw), np.cos(yaw), 0.0],
                [0.0, 0.0, 1.0],
            ])
            

            # 3x8 matrix, corner points translated to velodyne coordinate system
            cornerPosInVelo = np.dot(rotMat, trackletBox) + np.tile(translation, (8,1)).T
            frame_tracklets[absoluteFrameNumber] = frame_tracklets[absoluteFrameNumber] + [cornerPosInVelo]
            frame_tracklet_types[absoluteFrameNumber] = frame_tracklet_types[absoluteFrameNumber] + [tracklet.objectType]

    
    return (frame_tracklets, frame_tracklet_types)




def draw_box(pyplot_axis, vertices, axes=[0, 1, 2], color='black'):
    # Draw a 3d bounding box 
    vertices = vertices[axes, :]
    connections = [
        [0, 1], [1, 2], [2, 3], [3, 0],  # Lower plane parallel to Z=0 plane
        [4, 5], [5, 6], [6, 7], [7, 4],  # Upper plane parallel to Z=0 plane
        [0, 4], [1, 5], [2, 6], [3, 7]  # Connections between upper and lower planes
    ]
    for connection in connections:
        pyplot_axis.plot(*vertices[:, connection], c=color, lw=0.5)


def display_frame_statistics(dataset, tracklet_rects, tracklet_types, frame, points=0.2):
    # Draw camera data, 3D plot of the lidar point cloud data and point cloud projections to
    # various planes 
    dataset_gray = list(dataset.gray)
    dataset_rgb = list(dataset.rgb)
    dataset_velo = list(dataset.velo)

    print("Frame timestamp: {}".format(str(dataset.timestamps[frame])))
    # Draw camera data 
    f, ax = plt.subplots(2, 2, figsize=(15, 5))
    ax[0, 0].imshow(dataset_gray[frame][0], cmap='gray')
    ax[0, 0].set_title('Left Gray Image (cam0)')
    ax[0, 1].imshow(dataset_gray[frame][1], cmap='gray')
    ax[0, 1].set_title('Right Gray Image (cam1)')
    ax[1, 0].imshow(dataset_rgb[frame][0])
    ax[1, 0].set_title('Left RGB Image (cam2)')
    ax[1, 1].imshow(dataset_rgb[frame][1])
    ax[1, 1].set_title('Right RGB Image (cam3)')
    plt.show()
    f.savefig("images.png")

    points_step = int(1. / points)
    point_size = 0.01 * (1. / points)
    velo_range = range(0, dataset_velo[frame].shape[0], points_step)
    velo_frames = dataset_velo[frame][velo_range, :]

    def draw_point_cloud(ax, title, axes=[0, 1, 2], xlim3d=None, ylim3d=None, zlim3d=None):
        # Draw point cloud projections 
        ax.scatter(*np.transpose(velo_frames[:, axes]), s=point_size, c=velo_frames[:, 3], cmap='gray')
        ax.set_title(title)
        ax.set_xlabel('{} axis'.format(axes_str[axes[0]]))
        ax.set_ylabel('{} axis'.format(axes_str[axes[1]]))
        if len(axes) > 2:
            ax.set_xlim3d(*axes_limits[axes[0]])
            ax.set_ylim3d(*axes_limits[axes[1]])
            ax.set_zlim3d(*axes_limits[axes[2]])
            ax.set_zlabel('{} axis'.format(axes_str[axes[2]]))
        else:
            ax.set_xlim(*axes_limits[axes[0]])
            ax.set_ylim(*axes_limits[axes[1]])
        # User specified limits
        if xlim3d!=None:
            ax.set_xlim3d(xlim3d)
        if ylim3d!=None:
            ax.set_ylim3d(ylim3d)
        if zlim3d!=None:
            ax.set_zlim3d(zlim3d)
            
        for t_rects, t_type in zip(tracklet_rects[frame], tracklet_types[frame]):
            draw_box(ax, t_rects, axes=axes, color=colors[t_type])

    # Draw point cloud data as 3D plot
    f2 = plt.figure(figsize=(15, 8))
    ax2 = f2.add_subplot(111, projection='3d')                    
    draw_point_cloud(ax2, 'Velodyne scan', xlim3d=(-10,30))
    plt.show()
    f2.savefig('velodyne-frame-{}.png'.format(frame))
    
    # Draw point cloud data as plane projections
    f, ax3 = plt.subplots(3, 1, figsize=(15, 25))
    draw_point_cloud(
        ax3[0], 
        'Velodyne scan, XZ projection (Y = 0), the car is moving in direction left to right', 
        axes=[0, 2] # X and Z axes
    )
    draw_point_cloud(
        ax3[1], 
        'Velodyne scan, XY projection (Z = 0), the car is moving in direction left to right', 
        axes=[0, 1] # X and Y axes
    )
    draw_point_cloud(
        ax3[2], 
        'Velodyne scan, YZ projection (X = 0), the car is moving towards the graph plane', 
        axes=[1, 2] # Y and Z axes
    )
    plt.show()
    f.savefig('velodyne-projections-frame-{}'.format(frame))



def main():
    frame = 10
    date = '2011_09_26'
    drive = '0001'
    dataset = load_dataset(date, drive)
    tracklet_rects, tracklet_types = load_tracklets_for_frames(
        len(list(dataset.velo)), 
        'data/{}/{}_drive_{}_sync/tracklet_labels.xml'.format(date, date, drive), 
    )
    display_frame_statistics(dataset, tracklet_rects, tracklet_types, frame)


if __name__ == "__main__":
    main()




























