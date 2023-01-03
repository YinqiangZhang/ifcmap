import os
import numpy as np 
import open3d as o3d

# read mesh
mesh = o3d.io.read_triangle_mesh(os.path.join('result_map', 'structure_small.ply'))
# mesh = o3d.io.read_triangle_mesh(os.path.join('result_map', 'layers', 'Storey_2_F.ply'))
mesh.compute_vertex_normals()
mesh.paint_uniform_color(np.array([65,105,225])/255.0)
# pcd = mesh.sample_points_uniformly(number_of_points=5000)

vis = o3d.visualization.Visualizer()
vis.create_window()
vis.add_geometry(mesh)
vis.run()
vis.destroy_window()
