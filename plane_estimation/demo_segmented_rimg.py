import os 
import pickle
import glob
import open3d as o3d
import numpy as np 
import matplotlib as mpl
import matplotlib.pyplot as plt
from utils.plane_utils import rotate_view
from utils.plane import PlaneCandidate

root_path = os.path.dirname(os.path.abspath(__file__))    
static_map_path = glob.glob(os.path.join(root_path, 'segmented_frames', '*.ply'))[0]
static_map = o3d.io.read_point_cloud(static_map_path, format='ply')
static_map.paint_uniform_color(np.array([220, 220, 220])/255)

plane_list = list()
plane_paths = glob.glob(os.path.join(root_path, 'site_planes', '*.ply'))
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
    
with open(os.path.join(root_path, 'plane_data', 'plane_objects.pkl'), 'wb') as f:
    pickle.dump(plane_list, f)

vis = o3d.visualization.Visualizer()
vis.create_window()
# vis.register_animation_callback(rotate_view)
# vis.add_geometry(static_map.voxel_down_sample(0.1))
# ctr = vis.get_view_control()
# ctr.convert_from_pinhole_camera_parameters(camera_parameters)
vis.add_geometry(plane_map)
vis.run()
vis.destroy_window()