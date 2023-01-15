import os 
import glob
import pickle
import numpy as np 
import open3d as o3d 
import matplotlib.pyplot as plt 
import matplotlib as mpl
import numpy as np

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

mesh_list = list(o3d.io.read_triangle_mesh(elem_path) for elem_path in mesh_path_list)
sorted_mesh_list = sorted(mesh_list, key=lambda x:x.get_surface_area(), reverse=True)

point_list = list()
strucutre_points = o3d.geometry.PointCloud()
for mesh in sorted_mesh_list:
    area = mesh.get_surface_area()
    points = mesh.sample_points_uniformly(number_of_points=int(area), use_triangle_normal=True)
    point_list.append(points)
    strucutre_points += points
    # o3d.visualization.draw_geometries([points])

# plane processing
planes = o3d.geometry.PointCloud()
ground_planes = o3d.geometry.PointCloud()
for idx, plane in enumerate(sorted_plane_data):
    plane_points = o3d.geometry.PointCloud()
    inlier_points = plane.points[np.squeeze(plane.inliers==1), :]
    plane_points.points = o3d.utility.Vector3dVector(inlier_points)
    plane_points.normals = o3d.utility.Vector3dVector(
        np.repeat(plane.plane_params[:, :-1], inlier_points.shape[0], axis=0)
        )
    color = plt.get_cmap('nipy_spectral')(cmap_norm(plane.id))[0:3]
    plane_points.paint_uniform_color(color)
    dws_points = plane_points.voxel_down_sample(voxel_size=1.0)
    print('Plane parameters: {}'.format(plane.plane_params))
    print('Point Number: {}'.format(np.asarray(dws_points.points).shape[0]))
    planes += dws_points
    if idx < 2:
        ground_planes += dws_points

result = o3d.pipelines.registration.registration_icp(
    planes, strucutre_points, 2.0, np.identity(4),
    o3d.pipelines.registration.TransformationEstimationPointToPlane())

print('Transformation result: {}'.format(result.transformation))

planes.transform(result.transformation)

vis = o3d.visualization.Visualizer()
vis.create_window()
vis.add_geometry(strucutre_points)
vis.add_geometry(planes)
vis.run()
vis.destroy_window()

