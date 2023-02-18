import os 
import copy
import glob
import pickle
import random
import numpy as np 
import open3d as o3d 
import matplotlib.pyplot as plt 
import matplotlib as mpl
import numpy as np
from utils.primitive_registor import PrimitiveRegistor
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
import trimesh
from multiprocessing import Pool


def online_clustering(association_dict, cos_thres=0.99, angle_thres=0.05):
    cluster_list = list()
    association_clusters = list()
    cluster_vectors = np.array([])
    for index_pair, vector in association_dict.items():
        is_new_cluster = False
        if cluster_vectors.shape[0] == 0:
            is_new_cluster = True
        else:
            cos_sim = np.matmul(cluster_vectors[:, :-1], np.atleast_2d(vector[:-1]).T)
            delta_angle = np.abs(cluster_vectors[:, -1] - vector[-1])
            score = np.arccos(np.squeeze(cos_sim)) + 2*delta_angle
            if np.max(cos_sim) < cos_thres or np.min(delta_angle) > angle_thres:
                is_new_cluster = True
            else:
                min_idx = -1
                min_score = np.inf
                for idx in range(cluster_vectors.shape[0]):
                    if cos_sim[idx] > cos_thres and delta_angle[idx] < angle_thres:
                        if min_score > score[idx]:
                            min_idx = idx
                            min_score = score[idx]
                if min_idx != -1:
                    cluster_list[min_idx] = np.vstack((cluster_list[min_idx], vector))
                    association_clusters[min_idx].append(index_pair)
                            
        if is_new_cluster:
            if cluster_vectors.shape[0] == 0:
                cluster_vectors = np.atleast_2d(vector)
            else:
                cluster_vectors = np.vstack((cluster_vectors, np.atleast_2d(vector)))
            cluster_list.append(np.atleast_2d(vector))
            association_clusters.append([index_pair])
    
    cluster_info = sorted(zip(cluster_list, cluster_vectors, association_clusters), key= lambda x: x[0].shape[0], reverse=True)
    return cluster_info

# read segmented planes
root_path = os.path.dirname(os.path.abspath(__file__))    
data_folder = os.path.join(root_path, 'plane_candidates')
data_date = '20230208'
plane_folder = os.path.join(data_folder, data_date)
model_folder = os.path.join(root_path, '..', 'BIM_plane_objects')
model_mesh_folder = os.path.join(model_folder, 'mesh_models')
    
with open(os.path.join(plane_folder, 'selected_plane_objects.pkl'), 'rb') as f:
    target_planes = pickle.load(f)

with open(os.path.join(model_folder, 'model_plane_objects.pkl'), 'rb') as f:
    model_plane_list = pickle.load(f)

model_mesh_filepaths = glob.glob(os.path.join(model_mesh_folder, '*.ply'))

# load selected BIM model
o3d_model_mesh = o3d.geometry.TriangleMesh()
model_mesh_list = list()
model_trimesh_list = list()
model_plane_params = list()
for mesh_path in model_mesh_filepaths:
    model_mesh = o3d.io.read_triangle_mesh(mesh_path)
    model_trimesh = trimesh.load(mesh_path)
    model_mesh.compute_vertex_normals()
    model_mesh_list.append(model_mesh)
    model_trimesh_list.append(model_trimesh)
    o3d_model_mesh += model_mesh

'''
Here,
1. source point clouds are extracted from BIM CAD model.
2. target point clouds are extracted from LiDAR measurements.
'''
source_params_list = list()
source_points_list = list()
source_points = o3d.geometry.PointCloud()
for o3d_mesh, plane_params in zip(model_mesh_list, model_plane_list):
    o3d_mesh.paint_uniform_color(np.array([65, 105, 225])/255)
    pcd = o3d_mesh.sample_points_uniformly(number_of_points=int(o3d_mesh.get_surface_area()*3))
    pcd.paint_uniform_color(np.array([205, 92, 92])/255)
    source_params_list.append(np.atleast_2d(plane_params[0]))
    source_points_list.append(pcd)
    source_points += pcd

target_params_list = list()
target_points_list = list()
target_points = o3d.geometry.PointCloud()
for idx, plane in enumerate(target_planes):
    target_params_list.append(plane.plane_params)
    o3d_points = o3d.geometry.PointCloud()
    o3d_points.points = o3d.utility.Vector3dVector(plane.points)
    o3d_points.normals = o3d.utility.Vector3dVector(
        np.repeat(plane.plane_params[:, :-1], plane.points.shape[0], axis=0)
        )
    tri_points = trimesh.PointCloud(vertices=plane.points)
    target_points_list.append(tri_points)
    target_points += o3d_points

source_points.paint_uniform_color(np.array([65, 105, 225])/255)
target_points.paint_uniform_color(np.array([218, 165, 32])/255)
o3d.visualization.draw_geometries([source_points, target_points])

'''
Here,
Association dict will be extracted. 
'''
association_dict = dict()
norm_list = np.array([])
for target_idx, target_param in enumerate(target_params_list):
    for source_idx, source_param in enumerate(source_params_list):
        # TODO: should be considered later 
        # each normal vector as two possible directions 
        # therefore, there are two possible angles
        # we can directly use sin values to represent this thing.
        target_norm = target_param[0, :-1] if target_param[0, 0] > 0 else -target_param[0, :-1]
        source_norm = source_param[0, :-1] if source_param[0, 0] > 0 else -source_param[0, :-1]
        # cross product: target to source vectors
        axis_vec = np.cross(target_norm, source_norm)
        axis_norm = np.linalg.norm(axis_vec)
        axis_vec /= axis_norm
        association_dict[target_idx, source_idx] = np.append(axis_vec, axis_norm)
        if norm_list.shape[0] == 0:
            norm_list = np.atleast_2d(axis_vec)
        else:
            norm_list = np.vstack((norm_list, np.atleast_2d(axis_vec)))

cluster_info = online_clustering(association_dict, 0.99, 0.05)

rot_mat_list = list()
association_list = list()
for vectors, center, associations in cluster_info[:5]:
    print(center)
    new_center = np.median(vectors, axis=0)
    new_center[:-1] /= np.linalg.norm(new_center[:-1])
    rot_mat_list.append(R.from_rotvec(new_center[:-1] * new_center[-1]))
    association_list.append(associations)
    
# TODO: analyze association pairs
optimization_pairs = dict()
for association_pair in association_list[0]:
    target_idx, source_idx = association_pair
    if optimization_pairs.get(target_idx, None) is None:
        optimization_pairs[target_idx] = [source_idx]
    else:
        optimization_pairs[target_idx].append(source_idx)
        
# merged_optimization_pairs = dict()
# for target_idx, pairs in optimization_pairs.items():
#     d_list = list()
#     for plane_pair in pairs:
#         plane_params = np.vstack(model_plane_list[plane_pair])
#         d_list.append(np.abs(plane_params[:, -1]).mean())
#     clustering = DBSCAN(eps=0.5, min_samples=1).fit(np.atleast_2d(np.array(d_list)).T)
#     merged_pair_list = list()
#     for label in np.unique(clustering.labels_):
#         merged_pair_list.append(np.array(pairs)[clustering.labels_ == label])
#     merged_optimization_pairs[target_idx] = merged_pair_list

# association matrix
# target_relation_graph = dict()
# target_indices = list(merged_optimization_pairs.keys())
# for j, idx_j in enumerate(target_indices):
#     for k, idx_k in enumerate(target_indices[j+1:], j+1):
#         plane_param_j = target_params_list[idx_j]
#         plane_param_k = target_params_list[idx_k]
#         similarity = np.abs(np.dot(plane_param_j[:, :-1], plane_param_k[:, :-1].T)).item()
#         target_relation_graph[idx_j, idx_k] = np.arccos(similarity) * 180.0 / np.pi 
#         target_relation_graph[idx_k, idx_j] = np.arccos(similarity) * 180.0 / np.pi 
        
# for target_idx, merged_pairs in merged_optimization_pairs.items():
#     points = np.asarray(target_points_list[target_idx].points)
#     homo_points = np.column_stack((points, np.ones((points.shape[0], 1))))

# source_relation_graph = dict()
# for idx_j in range(len(model_plane_list)):
#     for idx_k in range(j+1, len(model_plane_list)):
#         plane_param_j = model_plane_list[idx_j]
#         plane_param_k = model_plane_list[idx_k]
#         similarity = min(np.abs(np.dot(plane_param_j[0][:-1], plane_param_k[0][:-1])), 1.0)
#         source_relation_graph[idx_j, idx_k] = np.arccos(similarity) * 180.0 / np.pi 
#         source_relation_graph[idx_k, idx_j] = np.arccos(similarity) * 180.0 / np.pi 

# curr_association = association_list[0]
# for pair in curr_association:
#     target_idx, source_idx = association_pair
#     model_trimesh = model_trimesh_list[source_idx]
#     obj_plane_params = model_plane_params[source_idx]
#     points = np.asarray(target_points_list[target_idx].points)
#     pass 

# consistency_matrix = dict()
# for idx_j, pair_j in enumerate(curr_association):
#     for idx_k, pair_k in enumerate(curr_association[idx_j+1:], idx_j+1):
#         target_idx_j, source_idx_j = pair_j
#         target_idx_k, source_idx_k = pair_k
        
#         if target_idx_j != target_idx_k:
#             target_relation = target_relation_graph[target_idx_j, target_idx_k]
#             print('Target ({}, {}): {}'.format(target_idx_j, target_idx_k, target_relation))
#         if source_idx_j != source_idx_k:
#             source_relation = source_relation_graph[source_idx_j, source_idx_k]
#             print('Source ({}, {}): {}'.format(source_idx_j, source_idx_k, source_relation))
            
#         if target_idx_j == target_idx_k and source_relation < 10.0:
#             consistency_matrix[idx_j, idx_k] = 1
#         elif source_idx_j == source_idx_k and target_relation < 10.0:
#             consistency_matrix[idx_j, idx_k] = 1
#         elif np.abs(target_relation - source_relation) < 10.0:
#             consistency_matrix[idx_j, idx_k] = 1
#         else:
#             consistency_matrix[idx_j, idx_k] = 0
#         pass

# for target_idx, pairs in optimization_pairs.items():
#     all_points = o3d.geometry.PointCloud()
#     all_meshs = o3d.geometry.TriangleMesh()
#     for idx, points in enumerate(copy.deepcopy(target_points_list)):
#         if idx != target_idx:
#             points.paint_uniform_color(np.array([105, 105, 105])/255)
#         else:
#             points.paint_uniform_color(np.array([205, 92, 92])/255)
#         all_points += points
#     for idx, mesh in enumerate(copy.deepcopy(model_mesh_list)):
#         if idx in pairs[:1]:
#             mesh.paint_uniform_color(np.array([205, 92, 92])/255)
#         else:
#             mesh.paint_uniform_color(np.array([105, 105, 105])/255)
#         all_meshs += mesh
#     o3d.visualization.draw_geometries([all_points, all_meshs])

# TODO: RANSAC-based correspondence selection
used_source_indices = set()
registor = PrimitiveRegistor(model_trimesh_list, target_points_list, [])
for target_idx, pair_list in optimization_pairs.items():
    average_V_list = list()
    result_trans_list = list()
    state_list = list()
    valid_pair_list = list(idx for idx in pair_list if idx not in used_source_indices)
    for idx, source_idx in enumerate(valid_pair_list):
        correspondence = (target_idx, source_idx)
        registor.add_correspondence(correspondence)
        registor.set_damping()
        result_trans, _ = registor.optimize()
        average_V_list.append(registor.get_average_potential())
        state_list.append(copy.deepcopy(registor.state))
        result_trans_list.append(result_trans)
        registor.remove_correspondence()
        print('Index: {}, Total V: {}'.format(correspondence, average_V_list[-1]))
    
    best_idx = np.argmin(average_V_list)
    if (average_V_list[best_idx] < 0.03):
        registor.add_correspondence((target_idx, valid_pair_list[best_idx]))
        used_source_indices.add(valid_pair_list[best_idx])
        registor.state = state_list[best_idx]
        print('Current correspondences: {}'.format(registor.correspondence_list))
        best_trans = result_trans_list[best_idx]
        aligned_points = copy.deepcopy(target_points)
        aligned_points.transform(result_trans_list[best_idx])
        o3d.visualization.draw_geometries([aligned_points, o3d_model_mesh])
        
# for idx in range(len(association_list[0])):
#     registor.add_correspondence(association_list[0][idx])
#     result_trans, _ = registor.optimize()
#     average_V = registor.get_average_potential()
#     print('Index: {}, Total V: {}'.format(idx, average_V))
#     all_points = o3d.geometry.PointCloud()
#     all_meshs = o3d.geometry.TriangleMesh()
#     for j, points in enumerate(copy.deepcopy(target_points_list)):
#         if j != association_list[0][idx][0]:
#             points.paint_uniform_color(np.array([105, 105, 105])/255)
#         else:
#             points.paint_uniform_color(np.array([205, 92, 92])/255)
#         all_points += points
#     for k, mesh in enumerate(copy.deepcopy(model_mesh_list)):
#         if k == association_list[0][idx][1]:
#             mesh.paint_uniform_color(np.array([205, 92, 92])/255)
#         else:
#             mesh.paint_uniform_color(np.array([218, 165, 32])/255)
#         all_meshs += mesh
#     if average_V > 0.03:
#         registor.remove_correspondence()
#         print('Current correspondences: {}'.format(registor.correspondence_list))
#     else:
#         print('Current correspondences: {}'.format(registor.correspondence_list))
#         registor.visualize()
#         all_points.transform(result_trans)
#         o3d.visualization.draw_geometries([all_points, all_meshs])

# point-to-plane distances (normalized)
# initial results by rotation voting
initial_alignment_points = copy.deepcopy(target_points)
initial_alignment_points.transform(best_trans)
initial_alignment_points.paint_uniform_color(np.array([218, 165, 32])/255)

vis = o3d.visualization.Visualizer()
vis.create_window()
vis.add_geometry(source_points)
vis.add_geometry(o3d_model_mesh)
vis.add_geometry(initial_alignment_points)
vis.run()
vis.destroy_window()