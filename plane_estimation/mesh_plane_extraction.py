import os 
import glob
import numpy as np
import open3d as o3d
import trimesh
import networkx as nx

root_path = os.path.dirname(os.path.abspath(__file__))    

# read mesh data
mesh_path = glob.glob(os.path.join(root_path, 'plane_data', 'filtered_structure.ply'))[0]
# mesh = o3d.io.read_triangle_mesh(mesh_path)
# mesh.compute_vertex_normals()

mesh = trimesh.load(mesh_path)
# mesh.show()
print('Number of Bodies: {}'.format(mesh.body_count))

facets_num = len(mesh.facets)
facet_areas = mesh.facets_area
sorted_id = np.argsort(mesh.facets_area)[::-1]

ground_plates = list()
# for facet_id in sorted_id[:2]:
#     faces_id = mesh.facets[facet_id]
#     facet_mesh = mesh.submesh(mesh.faces[faces_id, :]) 
#     facet_mesh = trimesh.util.concatenate(facet_mesh)
#     facet_mesh.show()

graph = nx.from_edgelist(mesh.face_adjacency)
groups = nx.connected_components(graph)
for group in groups:
    facet_mesh = mesh.submesh(mesh.faces[list(group), :]) 
    facet_mesh = trimesh.util.concatenate(facet_mesh)
    facet_mesh.show()
    
o3d_mesh = mesh.as_open3d

# # visualization
# vis = o3d.visualization.Visualizer()
# vis.create_window()
# vis.add_geometry(o3d_mesh)
# vis.run()
# vis.destroy_window()