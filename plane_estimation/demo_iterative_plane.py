import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import time
import numpy as np
import open3d as o3d
import matplotlib as mpl
import matplotlib.pyplot as plt
from utils.plane_utils import generate_plane_points, rotate_view
from utils.plane import PlaneCandidate


if __name__ == '__main__':
    root_path = os.path.dirname(os.path.abspath(__file__))
    
    # color mapping
    colormap_name = ['winter', 'Wistia', 'cool']
    cmap_norm = mpl.colors.Normalize(vmin=0.0, vmax=1.0)

    # generate original data
    plane1 = generate_plane_points([1, 2, 3, 3], 300) 
    plane2 = generate_plane_points([1, 1, 1, 1], 1000) 
    plane3 = generate_plane_points([1, 3, 2, 0], 200) 
    points = np.vstack((plane2, plane1, plane3))
    
    plane_dict = dict()
    outlier_points = points
    rest_point_num = outlier_points.shape[0]
    iter_num = 1
    
    while rest_point_num > 150:
        weights = np.ones((outlier_points.shape[0], 1))
        plane_obj = PlaneCandidate(iter_num, outlier_points, weights)
        for _ in range(15):
            plane_obj.update()
        if np.sum(plane_obj.inliers) > 150:
            plane_dict[plane_obj.id] = plane_obj
        outlier_points = outlier_points[np.squeeze(plane_obj.inliers == 0), :]
        rest_point_num = outlier_points.shape[0]
        iter_num += 1
    
    plane_points_list = list()
    for idx, (plane_id, plane_obj) in enumerate(plane_dict.items()):
        inlier_points = plane_obj.get_inlier_points()
        point_colors = plt.get_cmap(colormap_name[idx])(cmap_norm(np.squeeze(plane_obj.inlier_weights)))[:, 0:3]
        pcd_points = o3d.geometry.PointCloud()
        pcd_points.points = o3d.utility.Vector3dVector(inlier_points)
        pcd_points.colors = o3d.utility.Vector3dVector(point_colors)
        plane_points_list.append(pcd_points)
    
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    for plane_points in plane_points_list:
        vis.add_geometry(plane_points)
    vis.register_animation_callback(rotate_view)
    vis.run()
    vis.destroy_window()
