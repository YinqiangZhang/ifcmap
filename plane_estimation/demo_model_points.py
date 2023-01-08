import os 
import glob
import numpy as np
import open3d as o3d

root_path = os.path.dirname(os.path.abspath(__file__))    
data_path_list = glob.glob(os.path.join(root_path, 'layers', '*.ply'))
mesh = o3d.io.read_triangle_mesh(data_path_list[5])
mesh.compute_vertex_normals()
mesh.paint_uniform_color(np.array([65, 105, 225])/255.0)

pcd = mesh.sample_points_uniformly(number_of_points=2000000)

vis = o3d.visualization.Visualizer()
vis.create_window()
# vis.add_geometry(mesh)
vis.add_geometry(pcd)
vis.run()
vis.destroy_window()