import os 
import glob
import pickle
import copy
import numpy as np 
import open3d as o3d 
import matplotlib.pyplot as plt 
import matplotlib as mpl
import trimesh

# read segmented plane
root_path = os.path.dirname(os.path.abspath(__file__))    
plane_data_path = os.path.join(root_path, 'plane_data')
mesh_data_path = os.path.join(root_path, 'mesh_data')
with open(os.path.join(plane_data_path, 'plane_objects.pkl'), 'rb') as f:
    plane_data = pickle.load(f)

cmap_norm = mpl.colors.Normalize(vmin=0.0, vmax=len(plane_data))
sorted_plane_data = sorted(plane_data, key=lambda x:np.sum(x.inliers==1), reverse=True)

# read mesh data
mesh_path = glob.glob(os.path.join(root_path, 'plane_data', 'filtered_structure.ply'))[0]
mesh_path_list = glob.glob(os.path.join(mesh_data_path, '*.ply'))

mesh_list = list(trimesh.load(elem_path) for elem_path in mesh_path_list)
sorted_mesh_list = sorted(mesh_list, key=lambda x:x.area, reverse=True)
demo_mesh = trimesh.util.concatenate(sorted_mesh_list[:2])

o3d_mesh = o3d.io.read_triangle_mesh(mesh_path)
o3d_mesh.compute_vertex_normals()

planes = o3d.geometry.PointCloud()
ground_planes = o3d.geometry.PointCloud()
for idx, plane in enumerate(sorted_plane_data):
    plane_points = o3d.geometry.PointCloud()
    inlier_points = plane.points[np.squeeze(plane.inliers==1), :]
    plane_points.points = o3d.utility.Vector3dVector(inlier_points)
    color = plt.get_cmap('nipy_spectral')(cmap_norm(plane.id))[0:3]
    plane_points.paint_uniform_color(color)
    dws_points = plane_points.voxel_down_sample(voxel_size=0.5)
    print('Plane parameters: {}'.format(plane.plane_params))
    print('Point Number: {}'.format(np.asarray(dws_points.points).shape[0]))
    planes += dws_points
    if idx < 2:
        ground_planes += dws_points
    
transf_mat, cost = trimesh.registration.mesh_other(
    demo_mesh, np.asarray(ground_planes.points), samples=50, scale=False, icp_first=2, icp_final=5
)
print(transf_mat, cost)
transformed_planes = copy.deepcopy(planes).transform(np.linalg.inv(transf_mat))

vis = o3d.visualization.Visualizer()
vis.create_window()
# vis.add_geometry(demo_mesh.as_open3d)
vis.add_geometry(o3d_mesh)
vis.add_geometry(transformed_planes)
# vis.add_geometry(planes)
vis.run()
vis.destroy_window()