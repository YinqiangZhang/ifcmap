import os
import glob
import numpy as np 
import open3d as o3d
import pyvista as pv


root_path = os.path.dirname(os.path.abspath(__file__))
mesh_path = glob.glob(os.path.join(root_path, 'result_map', 'filtered_structure.ply'))[0]
mesh = o3d.io.read_triangle_mesh(mesh_path)
mesh.compute_vertex_normals()
vis = o3d.visualization.Visualizer()
vis.create_window()
vis.add_geometry(mesh)
vis.run()
vis.destroy_window()