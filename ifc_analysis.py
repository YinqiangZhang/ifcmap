import os 
import glob
import copy 
import pickle
import ifcopenshell
import numpy as np 
from tqdm import tqdm 
from ifcfunctions import meshfromshape, getunitfactor
from ifcopenshell.util.element import get_decomposition, get_type
from ifcopenshell.util.placement import get_storey_elevation # get the elevation of each storey 
from ifcopenshell.util.placement import get_local_placement
import ifcopenshell.geom as ifc_geom
from ifcopenshell.geom import tree
import pandas as pd
import open3d as o3d 
import trimesh

def analyze_wall_shapes(mesh):
    model_plane_list = list()
    for facet, area in zip(mesh.facets, mesh.facets_area):
        vertex_indices = np.unique(mesh.faces[facet].reshape(-1)) 
        face_normals = np.mean(np.asarray(mesh.face_normals[facet]), axis=0)
        model_plane_list.append((area, vertex_indices, face_normals))
    sorted_model_planes = sorted(model_plane_list, key= lambda x: x[0], reverse=True)
    # if wall-like objects, we only extract two largest surfaces
    target_wall_planes = sorted_model_planes[:2]
    plane_params = list()
    for area, indices, normals in target_wall_planes:
        plane_points = np.asarray(mesh.vertices[indices])
        d = -np.dot(np.mean(plane_points, axis=0), normals)
        plane_params.append(np.append(normals, d))
    # extract vertex coordinates
    return plane_params

def model_plane_extraction(mesh_list, centroid):
    mesh_plane_list = list()
    new_mesh_list = list()
    for mesh in mesh_list:
        verts = np.asarray(mesh.vertices) - centroid
        new_mesh = trimesh.Trimesh(vertices=verts,faces=mesh.faces, edges=mesh.edges, process=True)
        obj_plane_params = analyze_wall_shapes(new_mesh)
        mesh_plane_list.append(obj_plane_params)
        new_mesh_list.append(new_mesh)
    return mesh_plane_list, new_mesh_list

root_path = os.path.dirname(os.path.abspath(__file__)) 
ifc_files = glob.glob(os.path.join(root_path, 'real_models', '*.ifc'))

ifc_file_env = ifc_files[0]
ifc_file_structure = ifc_files[2]

ifc_env_model = ifcopenshell.open(ifc_file_env)
env_storeys = ifc_env_model.by_type('IfcBuildingStorey')

ifc_structure_model = ifcopenshell.open(ifc_file_structure)
structure_storeys = ifc_structure_model.by_type('IfcBuildingStorey')

settings = ifc_geom.settings()
settings.set(settings.USE_WORLD_COORDS, True)
unitfactor = getunitfactor(ifc_env_model) # millimeter

# color_dict = {
#     # 'IfcPlate': [107, 142, 35],
#     # 'IfcBeam': [255, 130, 71],
#     # 'IfcMember': [47, 79, 79],
#     # 'IfcWallStandardCase': [83, 134, 139],
#     'IfcSite':  [47, 79, 79],
#     'IfcRamp': [83, 134, 139],
#     'IfcColumn': [72, 61, 139], # steel structure
#     'IfcSlab': [65, 105, 225], # ground and ceiling
# }

color_dict = {
    'IfcColumn': [106, 90, 205],
    # 'IfcSite': 	[65, 105, 225],
    # 'IfcRamp': [60, 179, 113], 
    # 'IfcFlowTerminal': [255, 215, 0],
    # 'IfcGrid': [205, 92, 92],
    'IfcWallStandardCase': [205, 133, 63],
    # 'IfcWall': [176, 48, 96],
    # 'IfcDoor': [255, 218, 185],
    'IfcSlab': [65, 105, 225],
    # 'IfcStair': [187, 255, 255],
    # 'IfcCovering': [192, 255, 62],
    # 'IfcBuildingElementProxy': [255, 246, 143],
    # 'IfcFurnishingElement': [205, 190, 112],
    # 'IfcWindow': [139, 101, 8],
    # 'IfcBeam': [255, 130, 71],
    # 'IfcRailing': [255, 140, 105],
    # 'IfcStairFlight': [238, 106, 80], 
    # 'IfcMember': [255, 181, 197], 
    # 'IfcPlate': [255, 225, 255], 
    # 'IfcFastener': [144, 238, 144], 
    # 'IfcCurtainWall': [224, 238, 224],
    # 'IfcRampFlight': [205, 183, 181], 
}

# element_colors = {
#                 'Rectangular steel tube column': [255, 215, 0], 
#                 'Column-UC': [255, 69, 0], 
#                 '楼板': [99, 184, 255],
#                 'Column-RC-Rectangular': [192, 255, 62], 
#                 }

# only use 180 to be the plate
slab_elements = {#'楼板:JCT-A-FLR-RC-50mm': [106, 90, 205], # env
                '楼板:JCT-C99-SLA-RC-180mm': [205, 92, 92], # structure
                }

wall_elements = {
    # '基本墙:JCT-A-WAL-RC-300mm': [112, 128, 144],
    # '基本墙:JCT-A-WAL-RC-225mm': [135, 206, 235], # blue 
    '基本墙:JCT-A-WAL-RC-150mm': [85, 107, 47],
    # '基本墙:JCT-A-WAL-RC-200mm': [205, 133, 63],
    '基本墙:JCT-A-WAL-RC-100mm': [255, 99, 71],
    '基本墙:JCT-A-WAL-Block Wall_100mm': [186, 85, 211], 
    '基本墙:JCT-A-WAL-partition-100mm':[255, 222, 173], 
    # '基本墙:JCT-A-WAL-partition-25mm': [171, 130, 255], 
    # '基本墙:JCT-A-WAL-partition-80mm': [238, 122, 233], 
    # '基本墙:JCT-C-SWL-RC-150mm': [255, 69, 0], orange rectangular 
}

target_elev = 58780.0 / unitfactor
building_mesh = o3d.geometry.TriangleMesh()
storey_mesh_list = list()
o3d_mesh_list = list()
tri_mesh_list = list()

for storey_idx, (env_storey, structure_storey) in enumerate(zip(env_storeys[:], structure_storeys[:])):
    storey_mesh = o3d.geometry.TriangleMesh()
    env_elev = get_storey_elevation(env_storey)
    structure_elev = get_storey_elevation(structure_storey)
    
    env_elements = get_decomposition(env_storey)
    structure_elements = get_decomposition(structure_storey)
    print('Storey {}, Elevation: {:.4}'.format(env_storey.Name, env_elev/unitfactor))
    # types = list(elem.get_info()['type'] for elem in env_elements)
    # print(set(types))
    
    for element_idx, element_pair in enumerate(zip(env_elements, structure_elements)):
        for pair_idx, element in enumerate(element_pair[:]):
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
            if mesh.vertices.shape[0] == 0:
                continue 
            o3d_mesh = mesh.as_open3d
            type_name = element.ObjectType.split(':')[0]
            
            # color = element_colors.get(type_name, None)
            # if color is None:
            #     continue
            
            if elem_type == 'IfcSlab':
                color = slab_elements.get(element.ObjectType, None)
            elif elem_type == 'IfcWallStandardCase':
                color = wall_elements.get(element.ObjectType, None)
            if color is None:
                continue
            
            # print(element.ObjectType, elem_type, pair_idx)
            # check elevation
            # z_coords = np.asarray(o3d_mesh.vertices)[:, 2]
            # min_elev, max_elev = np.min(z_coords), np.max(z_coords)
            # if max_elev < target_elev:
            #     continue
            o3d_mesh.compute_vertex_normals()
            o3d_mesh.paint_uniform_color(np.array(color)/255.0)
            building_mesh += o3d_mesh
            storey_mesh += o3d_mesh
            # o3d_mesh_list.append(('S{}_E{}.ply'.format(storey_idx, element_idx), o3d_mesh))
            
            # TODO: extract planes from primitives
            tri_mesh_list.append(mesh)
            # sample_pcd = o3d_mesh.sample_points_uniformly(number_of_points=2000, use_triangle_normal=True)
            # o3d.visualization.draw_geometries([o3d_mesh])

structure_vertices = np.asarray(building_mesh.vertices)
vertices_centroid = np.mean(structure_vertices, axis = 0)
structure_vertices -= vertices_centroid
mesh_plane_list, new_mesh_list = model_plane_extraction(tri_mesh_list, vertices_centroid)

# with open(os.path.join(root_path, 'BIM_plane_objects', 'initial_translation.pkl'), 'wb') as f:
#     pickle.dump(vertices_centroid, f)

# with open(os.path.join(root_path, 'BIM_plane_objects', 'model_plane_objects.pkl'), 'wb') as f:
#     pickle.dump(mesh_plane_list, f)

# for idx, mesh in enumerate(new_mesh_list):
#     with open(os.path.join(root_path, 'BIM_plane_objects', 'mesh_models', '{}.ply'.format(str(idx).zfill(4))), 'wb') as f:
#         f.write(trimesh.exchange.ply.export_ply(mesh))
building_mesh.vertices = o3d.utility.Vector3dVector(structure_vertices)
o3d.visualization.draw_geometries([building_mesh])

# for mesh_info in o3d_mesh_list:
#     vertices = np.asarray(mesh_info[1].vertices)
#     vertices -= vertices_centroid
#     mesh_info[1].vertices = o3d.utility.Vector3dVector(vertices)
    # o3d.io.write_triangle_mesh(
    #             os.path.join(root_path, 'plane_estimation', 'mesh_data', mesh_info[0]),
    #             mesh_info[1]
    #         )

# o3d.io.write_triangle_mesh(os.path.join(root_path, 'plane_estimation', 'plane_data', 'filtered_structure.ply'), 
#                            building_mesh,
#                            write_vertex_colors=True)

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
    