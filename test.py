import numpy as np 
import math 
from mpl_toolkits import mplot3d 
import matplotlib.pyplot as plt 



def main():
    point1 = np.array([2, 4, 2])
    point2 = np.array([3, 0, 1.5])
    point3 = np.array([4, 3, 2])

    rot_mat = np.array([
        [math.cos(45), -math.sin(45), 0],
        [math.sin(45), math.cos(45), 0],
        [0, 0, 1],
        ])

    
    tx = np.random.normal()
    xs = np.array([np.random.normal() for _ in range(5000)])

    fig = plt.figure()
    plt.hist(xs, bins='auto')
    plt.show()

    print(tx)
    t_rz = np.random.uniform(-np.pi / 4, np.pi / 4)
    print(t_rz)

    
    npoint1 = np.dot(point1, rot_mat)
    print(point1)
    print(npoint1)
    npoint2 = np.dot(point2, rot_mat)
    npoint3 = np.dot(point3, rot_mat)
    
    pcl = np.stack([point1, point2, point3], axis=0)
    print(pcl)
    xdata = pcl[:, 0]
    ydata = pcl[:, 1]
    zdata = pcl[:, 2]
    print(f"xdata: {xdata}")
    print(f"ydata: {ydata}")
    print(f"zdata: {zdata}")

    npcl = np.stack([npoint1, npoint2, npoint3], axis=0)
    print(npcl)
    nxdata = npcl[:, 0]
    nydata = npcl[:, 1]
    nzdata = npcl[:, 2]

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.scatter3D(xdata, ydata, zdata, c=zdata, marker='x')
    ax.scatter3D(nxdata, nydata, nzdata, c=nzdata, marker='o')
    ax.set_xlim([0.0, 6.0])
    ax.set_ylim([-3.0, 5.0])
    ax.set_zlim([0.0, 3.0])

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.show() 


if __name__ == "__main__":
    main()
