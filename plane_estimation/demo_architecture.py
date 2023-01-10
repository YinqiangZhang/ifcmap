import os 
import glob
import numpy as np 
import open3d as o3d
import trimesh 
import pyvista as pv


root_path = os.path.dirname(os.path.abspath(__file__))    
static_map_path = glob.glob(os.path.join(root_path, 'segmented_frames', '*.ply'))[2]

# open3d 
# mesh = o3d.io.read_triangle_mesh(static_map_path)
# mesh.compute_vertex_normals()
# mesh.paint_uniform_color(np.array([119, 136, 153])/255)

# triangles = np.asarray(mesh.triangles)
# vertices = np.asarray(mesh.vertices)

# vertex_points = o3d.geometry.PointCloud()
# vertex_points.points = o3d.utility.Vector3dVector(vertices)

# vis = o3d.visualization.Visualizer()
# vis.create_window()
# vis.add_geometry(mesh)
# vis.run()
# vis.destroy_window()

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
mesh = pv.read(static_map_path)
pl = pv.Plotter()
pl.set_background('white')
pl.add_mesh(mesh, color='blue', specular=1.0, specular_power=15, lighting=True)
# pl.add_points(mesh.points, color='red', point_size=20)
# pl.remove_scalar_bar()
pl.link_views()
pl.view_isometric()
pl.show()



