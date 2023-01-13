import os 
import glob
import numpy as np 
import open3d as o3d
import trimesh 
import pyvista as pv


root_path = os.path.dirname(os.path.abspath(__file__))    
static_map_path = glob.glob(os.path.join(root_path, 'segmented_frames', '*.ply'))[3]

# open3d 
mesh = o3d.io.read_triangle_mesh(static_map_path)
mesh.compute_vertex_normals()
mesh.paint_uniform_color(np.array([119, 136, 153])/255)

triangles = np.asarray(mesh.triangles)
vertices = np.asarray(mesh.vertices)

vertex_points = o3d.geometry.PointCloud()
vertex_points.points = o3d.utility.Vector3dVector(vertices)

vis = o3d.visualization.Visualizer()
vis.create_window()
vis.add_geometry(mesh)
vis.run()
vis.destroy_window()

# trimesh
# mesh = trimesh.load(static_map_path)
# connected_components = trimesh.graph.connected_components(mesh.face_adjacency, min_len=3)
# for component in connected_components:
#     submesh = mesh.copy()
#     mask = np.zeros(len(mesh.faces), dtype=np.bool)
#     mask[component] = True
#     submesh.update_faces(mask)
#     submesh.show()

# pyvista
# mesh = pv.read(static_map_path)
# roi = pv.Cube(center=(0, 0, 0), x_length=80, y_length=80, z_length=200)
# # roi.rotate_z(25, point=(200,-20,5), inplace=True)
# extracted = mesh.clip_box(roi, invert=True)
# extracted = extracted.extract_surface()
# extracted.save(os.path.join(root_path, 'segmented_frames', 'architecture_cropped.ply'))

# pl = pv.Plotter()
# pl.set_background('white')
# pl.add_mesh(extracted, color='blue', specular=1.0, specular_power=15, lighting=True)
# pl.link_views()
# pl.view_isometric()
# pl.show()



