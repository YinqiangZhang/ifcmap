import os 
import pickle
import glob
import open3d as o3d
import numpy as np 
import matplotlib as mpl
import matplotlib.pyplot as plt
from utils.plane import PlaneCandidate

root_path = os.path.dirname(os.path.abspath(__file__))
data_folder = os.path.join(root_path, 'plane_candidates')
data_date = '20230208'
plane_folder = os.path.join(data_folder, data_date)

plane_list = list()
plane_paths = glob.glob(os.path.join(plane_folder, 'segmented_frames', '*.ply'))
cmap_norm = mpl.colors.Normalize(vmin=0.0, vmax=len(plane_paths))

plane_map = o3d.geometry.PointCloud()
for idx, plane_path in enumerate(plane_paths):
    plane_points = o3d.io.read_point_cloud(plane_path, format='ply')
    points = np.asarray(plane_points.points)
    plane_obj = PlaneCandidate(idx, points, np.ones((points.shape[0], 1)))
    plane_obj.update()
    valid_points = points[np.where(plane_obj.inliers == 1)[0], :]
    valid_plane_points = o3d.geometry.PointCloud()
    valid_plane_points.points = o3d.utility.Vector3dVector(valid_points)
    color = plt.get_cmap('nipy_spectral')(cmap_norm(idx))[0:3]
    valid_plane_points.paint_uniform_color(color)
    plane_list.append(plane_obj)
    plane_map += valid_plane_points
    print('Plane ID: {}'.format(idx))
    print('Outlier Number: {}'.format(np.sum((plane_obj.inliers == 0))))
    
with open(os.path.join(plane_folder, 'plane_objects.pkl'), 'wb') as f:
    pickle.dump(plane_list, f)

vis = o3d.visualization.Visualizer()
vis.create_window()
vis.add_geometry(plane_map)
vis.run()
vis.destroy_window()