import os 
import glob
import numpy as np 
import open3d as o3d
import matplotlib.pyplot as plt
from utils.plane_utils import project_to_range_image
import matplotlib as mpl

w = 1024
h = 128
dpi = 500
cmap_norm = mpl.colors.Normalize(vmin=0.0, vmax=75.0)


root_path = os.path.dirname(os.path.abspath(__file__))
data_path_list = glob.glob(os.path.join(root_path, 'site_cloud_data', '*.ply'))

# test one of them 
data_path = data_path_list[0]

cloud = o3d.io.read_point_cloud(data_path)
range_image, vertex_map = project_to_range_image(cloud, w, h, max_range=75)

filtered_cloud_points = vertex_map.reshape([-1, 3])
pcd_points = o3d.geometry.PointCloud()
point_colors = plt.get_cmap('magma')(cmap_norm(np.squeeze(range_image.reshape([-1, 1]))))[:, 0:3]
pcd_points.points = o3d.utility.Vector3dVector(filtered_cloud_points)
pcd_points.colors = o3d.utility.Vector3dVector(point_colors)

vis = o3d.visualization.Visualizer()
vis.create_window()
vis.add_geometry(pcd_points)
vis.run()
vis.destroy_window()

fig = plt.figure(frameon=False, figsize=(w / dpi, h / dpi), dpi=dpi)
ax = plt.Axes(fig, [0.0, 0.0, 1.0, 1.0])
ax.set_axis_off()
fig.add_axes(ax)

# Then draw your image on it :
ax.imshow(range_image, aspect="auto", cmap="magma")
plt.show()
fig.savefig(os.path.join(root_path, "range_image.png"), dpi=dpi)