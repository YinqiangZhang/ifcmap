import os 
import glob
import copy 
import ifcopenshell
import numpy as np 
from ifcfunctions import meshfromshape, getunitfactor
from ifcopenshell.util.element import get_decomposition, get_type
from ifcopenshell.util.placement import get_storey_elevation # get the elevation of each storey 
from ifcopenshell.util.placement import get_local_placement
import ifcopenshell.geom as ifc_geom
from ifcopenshell.geom import tree
import pandas as pd
import open3d as o3d 
import trimesh
from sklearn.cluster import DBSCAN


root_path = os.path.dirname(os.path.abspath(__file__))    
ifc_filepath = glob.glob(os.path.join(root_path, 'real_models', '*.ifc'))[2]

ifc_model = ifcopenshell.open(ifc_filepath)
storeys = ifc_model.by_type('IfcBuildingStorey')
# walls = ifc_model.by_type("IfcWallType")
settings = ifc_geom.settings()
settings.set(settings.USE_WORLD_COORDS, True)
unitfactor = getunitfactor(ifc_model) # millimeter

color_dict = {
    # 'IfcPlate': [107, 142, 35],
    # 'IfcBeam': [255, 130, 71],
    # 'IfcMember': [47, 79, 79],
    # 'IfcWallStandardCase': [83, 134, 139],
    'IfcColumn': [72, 61, 139], # steel structure
    'IfcSlab': [65, 105, 225], # ground and ceiling
}

target_elev = 58780.0 / unitfactor
storey_mesh = o3d.geometry.TriangleMesh()
for idx, storey in enumerate(storeys[0:12], 0):
    elev = get_storey_elevation(storey)
    elements = get_decomposition(storey)
    print('Storey {}, Elevation: {:.4}'.format(storey.Name, elev/unitfactor))
    types = list(elem.get_info()['type'] for elem in elements)
    print(set(types))
    
    for element in elements:
        elem_type = element.get_info()['type']
        color = color_dict.get(elem_type, None)
        if color is None:
            continue
        try:
            shape = ifc_geom.create_shape(settings, element)
        except RuntimeError:
            print('error')
            continue
        mesh = meshfromshape(shape, [0,255,0,100])
        o3d_mesh = mesh.as_open3d
        
        # check elevation
        z_coords = np.asarray(o3d_mesh.vertices)[:, 2]
        min_elev, max_elev = np.min(z_coords), np.max(z_coords)
        if max_elev < target_elev:
            continue
        o3d_mesh.compute_vertex_normals()
        o3d_mesh.paint_uniform_color(np.array(color)/255.0)
        storey_mesh += o3d_mesh
        
o3d.visualization.draw_geometries([storey_mesh])
o3d.io.write_triangle_mesh(os.path.join(root_path, 'result_map', 'filtered_structure.ply'), storey_mesh)
            # face_normals = mesh.face_normals
            # normal_cluster = DBSCAN(eps=0.1, min_samples=3).fit(face_normals)
            # color_list = list()
            # labels = normal_cluster.labels_
            # mesh.visual.face_colors = [255,255,255,255]
            # for label in np.unique(labels):
            #     face_indices = np.where(labels==label)
            #     vertex_indices = np.unique(mesh.faces[face_indices].reshape(-1))
            #     mesh.visual.face_colors[vertex_indices] = [200, 200, 250, 255]
            
            # facet visualization with open3d
            # vertex_pcd = o3d.geometry.PointCloud()
            # facet_mesh = copy.deepcopy(o3d_mesh)
            # facet_mesh.paint_uniform_color(np.array([255, 185, 15])/255.0)
            # facet_mesh.triangles = o3d.utility.Vector3iVector([])
            # for idx, facet in enumerate(mesh.facets):
            #     # print('Facet area: {}'.format(mesh.facets_area[idx]))
            #     face_indices = mesh.faces[facet]
            #     vertex_indices = np.unique(face_indices.reshape(-1))
            #     vertex_points = np.asarray(o3d_mesh.vertices)[vertex_indices, :]
                
            #     facet_pcd = o3d.geometry.PointCloud()
            #     facet_pcd.points = o3d.utility.Vector3dVector(vertex_points)
            #     vertex_pcd += facet_pcd
            #     vertex_pcd.paint_uniform_color(np.array([205, 85, 85])/255.0)
                
            #     triangles = np.vstack((np.asarray(facet_mesh.triangles), face_indices))
            #     facet_mesh.triangles = o3d.utility.Vector3iVector(triangles)
            #     facet_mesh.compute_vertex_normals()
                
                # o3d.visualization.draw_geometries([vertex_pcd, facet_mesh], mesh_show_back_face=True)
            
    # matrix = get_local_placement(storey.ObjectPlacement)
    # psets = ifcopenshell.util.element.get_psets(storey)
    # print(psets)

# Extract Planes from BIM models

# 1. for every beam we try to choose three planes to fitting the model
# H-shape with three planes 
    