import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import time
import glob
import numpy as np
import open3d as o3d
import matplotlib as mpl
import matplotlib.pyplot as plt
from utils.plane_utils import generate_plane_points, rotate_view
from utils.plane import PlaneCandidate


if __name__ == '__main__':
    root_path = os.path.dirname(os.path.abspath(__file__))
    
    # color mapping
    colormap_name = ['Greens', 'Purples']
    cmap_norm = mpl.colors.Normalize(vmin=0.0, vmax=1.0)
    
    data_path_list = glob.glob(os.path.join(root_path, 'site_cloud_data', '*.ply'))
    data_path = data_path_list[0]
    pcd_plane = o3d.io.read_point_cloud(data_path)
    noise_plane = np.asarray(pcd_plane.points)
    
    weights = np.ones((noise_plane.shape[0], 1))
    plane_obj = PlaneCandidate(1, noise_plane, weights)
    
    point_colors = plt.get_cmap('RdBu')(cmap_norm(np.squeeze(plane_obj.weights)))[:, 0:3]
    pcd_noise = o3d.geometry.PointCloud()
    pcd_noise.points = o3d.utility.Vector3dVector(noise_plane)
    pcd_noise.colors = o3d.utility.Vector3dVector(point_colors)
    
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=3.0)
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pcd_noise)
    vis.add_geometry(coordinate_frame)
    for _ in range(10):
        plane_obj.update()
        point_colors = plt.get_cmap('RdBu')(cmap_norm(np.squeeze(plane_obj.weights)))[:, 0:3]
        pcd_noise.colors = o3d.utility.Vector3dVector(point_colors)
        vis.update_geometry(pcd_noise)
        vis.poll_events()
        vis.update_renderer()
        time.sleep(0.001)
    vis.run()
    vis.destroy_window()
    
