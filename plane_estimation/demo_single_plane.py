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
    colormap_name = ['Greens', 'Purples']
    cmap_norm = mpl.colors.Normalize(vmin=0.0, vmax=1.0)

    plane1 = generate_plane_points([1, 2, 3, 3], 300) 
    plane2 = generate_plane_points([1, 1, 1, 1], 1000) 
    plane3 = generate_plane_points([1, 3, 2, 0], 200) 
    noise_plane = np.vstack((plane2, plane1, plane3))
    
    weights = np.ones((noise_plane.shape[0], 1))
    plane_obj = PlaneCandidate(1, noise_plane, weights)
    
    point_colors = plt.get_cmap('RdBu')(cmap_norm(np.squeeze(plane_obj.weights)))[:, 0:3]
    pcd_noise = o3d.geometry.PointCloud()
    pcd_noise.points = o3d.utility.Vector3dVector(noise_plane)
    pcd_noise.colors = o3d.utility.Vector3dVector(point_colors)
    
    camera_config = 'camera_config.json'
    camera_parameters = o3d.io.read_pinhole_camera_parameters(camera_config)
    # coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame()
    vis = o3d.visualization.Visualizer()
    vis.register_animation_callback(rotate_view)
    vis.create_window()
    vis.add_geometry(pcd_noise)
    ctr = vis.get_view_control()
    ctr.convert_from_pinhole_camera_parameters(camera_parameters)
    # vis.add_geometry(coordinate_frame)
    for _ in range(15):
        plane_obj.update()
        point_colors = plt.get_cmap('RdBu')(cmap_norm(np.squeeze(plane_obj.weights)))[:, 0:3]
        pcd_noise.colors = o3d.utility.Vector3dVector(point_colors)
        vis.update_geometry(pcd_noise)
        vis.poll_events()
        vis.update_renderer()
        time.sleep(0.5)
    vis.run()
    vis.destroy_window()
    # print(plane_obj.plane_params)
    
    # # other planes test
    # new_weights = weights[np.squeeze(plane_obj.inliers == 0)]
    # new_noise_plane = noise_plane[np.squeeze(plane_obj.inliers == 0), :]
    # plane_obj2 = PlaneCandidate(2, new_noise_plane, new_weights)
    # for _ in range(10):
    #     plane_obj2.update()
    # print(plane_obj2.plane_params)
    
    # # plane 1 
    # inlier_points1 = plane_obj.get_inlier_points()
    # point_colors1 = plt.get_cmap(colormap_name[0])(cmap_norm(np.squeeze(plane_obj.inlier_weights)))[:, 0:3]
    # pcd_noise = o3d.geometry.PointCloud()
    # pcd_noise.points = o3d.utility.Vector3dVector(inlier_points1)
    # pcd_noise.colors = o3d.utility.Vector3dVector(point_colors1)
    
    # # plane 2 
    # inlier_points2 = plane_obj2.get_inlier_points()
    # point_colors2 = plt.get_cmap(colormap_name[1])(cmap_norm(np.squeeze(plane_obj2.inlier_weights)))[:, 0:3]
    # pcd_noise2 = o3d.geometry.PointCloud()
    # pcd_noise2.points = o3d.utility.Vector3dVector(inlier_points2)
    # pcd_noise2.colors = o3d.utility.Vector3dVector(point_colors2)
    
    # vis = o3d.visualization.Visualizer()
    # vis.create_window()
    # vis.add_geometry(pcd_noise)
    # vis.register_animation_callback(rotate_view)
    # vis.run()
    # vis.destroy_window()
    
