import os
import numpy as np 
import open3d as o3d

# read mesh
# mesh = o3d.io.read_triangle_mesh(os.path.join('result_map', 'structure_small.ply'))
# mesh = o3d.io.read_triangle_mesh(os.path.join('result_map', 'layers_all', 'Storey_2_F.ply'))
# mesh.compute_vertex_normals()
# mesh.paint_uniform_color(np.array([65, 105, 225])/255.0)
target_storey_list = {
                    # 'Storey_0.000.ply': [107, 142, 35],
                    # 'Storey_B1_F.ply': [255, 130, 71],
                    # 'Storey_B1M_F.ply': [83, 134, 139],
                    # 'Storey_G_F.ply': [72, 61, 139],
                    # 'Storey_1_F.ply': [47, 79, 79],
                    'Storey_2_F.ply': [65, 105, 225], 
                    'Storey_3_F.ply': [255, 215, 0], 
                    'Storey_3M_F.ply': [205, 92, 92],
                    # 'Storey_4_F.ply': [139, 101, 8],
                    # 'Storey_5_F.ply': [28, 134, 238],
                    # 'Storey_6_F.ply': [154, 205, 50], 
                    # 'Storey_7_F.ply': [132, 112, 255],
                    }

mesh_list = list()
for target_storey, color in target_storey_list.items():
    mesh = o3d.io.read_triangle_mesh(os.path.join('result_map', 'layers_all', target_storey))
    mesh.compute_vertex_normals()
    mesh.paint_uniform_color(np.array(color)/255.0)
    mesh_list.append(mesh)
	
pcd = mesh.sample_points_uniformly(number_of_points=5000)

vis = o3d.visualization.Visualizer()
vis.create_window()
for mesh in mesh_list:
    vis.add_geometry(mesh)
vis.run()
vis.destroy_window()
