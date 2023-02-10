import os 
import random
import pickle
import numpy as np 
import open3d as o3d 
import matplotlib.pyplot as plt 
import matplotlib as mpl
import numpy as np
from utils.plane import PlaneCandidate
import itertools

def find_connected_compoments(relation_dict):
    # actually this is a region growing methods
    graph_nodes = set(relation_dict.keys())
    group_list = []
    while len(graph_nodes) != 0:
        seed_idx = graph_nodes.pop()
        open_set = set([seed_idx])
        closed_set = set([seed_idx])
        while len(open_set) != 0:
            open_idx = open_set.pop()
            if relation_dict.get(open_idx, None) is not None:
                open_set.update(relation_dict[open_idx])
                closed_set.update(relation_dict[open_idx])
        group_list.append(closed_set)
        graph_nodes.difference_update(closed_set)
    return group_list

if __name__ == '__main__':
    # read segmented planes
    root_path = os.path.dirname(os.path.abspath(__file__))    
    data_folder = os.path.join(root_path, 'plane_candidates')
    data_date = '20230208'
    plane_folder = os.path.join(data_folder, data_date)
    
    with open(os.path.join(plane_folder, 'plane_objects.pkl'), 'rb') as f:
        plane_data = pickle.load(f)

    sorted_plane_data = sorted(plane_data, key=lambda x:np.sum(x.inliers==1), reverse=True)
    plane_info_list = list()
    for plane in sorted_plane_data:
        plane_params = plane.plane_params
        o3d_points = o3d.geometry.PointCloud()
        o3d_points.points = o3d.utility.Vector3dVector(plane.points)
        pcd_tree = o3d.geometry.KDTreeFlann(o3d_points)
        plane_info_list.append((plane_params, plane.points, pcd_tree))
    
    # check duplicated map representation
#     duplicated_counts = 0
#     duplicated_pairs = dict()
    raw_info_list = plane_info_list
#     for idx_1, plane_info_1 in enumerate(raw_info_list):
#         for idx_2, plane_info_2 in enumerate(raw_info_list[idx_1+1:], idx_1+1):
#             print('Index Pair: {}'.format((idx_1, idx_2)))
#             param_1, points_1, tree_1 = plane_info_1
#             param_2, points_2, tree_2 = plane_info_2
#             normal_diff = np.abs(np.dot(param_1[:, :-1], param_2[:, :-1].T)).item()
#             if (normal_diff > 0.97):
#                 homo_points_1 = np.column_stack((points_1, np.ones((points_1.shape[0], 1))))
#                 dist_list_1 = np.abs(np.matmul(homo_points_1, param_2.T))
#                 min_dist_1 = np.percentile(dist_list_1, 10)
#                 if (min_dist_1 < 0.04):
#                     valid_points_1 = points_1[np.squeeze(dist_list_1 <= min_dist_1)]
#                     is_duplicated = False
#                     for valid_point in valid_points_1:
#                         k, _, _ = tree_2.search_radius_vector_3d(valid_point, 0.4)
#                         if k != 0:
#                             is_duplicated = True
#                             break
#                     if is_duplicated:
#                         if(duplicated_pairs.get(idx_1, None) is None):
#                             duplicated_pairs[idx_1] = [idx_2]
#                         else:
#                             duplicated_pairs[idx_1].append(idx_2)
#                         duplicated_counts += 1

# connected_components = find_connected_compoments(duplicated_pairs)
# with open(os.path.join(plane_folder, 'component_list.pkl'), 'wb') as f:
#     pickle.dump(connected_components, f)
    
# print('Number of duplicated pairs: {}'.format(duplicated_counts))
# print('Number of components: {}'.format(len(connected_components)))

with open(os.path.join(plane_folder, 'component_list.pkl'), 'rb') as f:
    connected_components = pickle.load(f)

removed_plane_indices = set()
merged_planes = list()
for component in connected_components:
    merged_points = list(raw_info_list[idx][1] for idx in component)
    merged_points = np.vstack(tuple(merged_points))
    plane_obj = PlaneCandidate(min(component), merged_points, np.ones((merged_points.shape[0], 1)))
    plane_obj.update()
    sorted_plane_data[min(component)] = plane_obj
    component.remove(min(component))
    removed_plane_indices.update(component)

filtered_plane_data = list()
for idx, plane_data in enumerate(sorted_plane_data):
    if idx not in removed_plane_indices:
        filtered_plane_data.append(plane_data)
sorted_filter_plane_data = sorted(filtered_plane_data, key=lambda x:np.sum(x.inliers==1), reverse=True)
print('Final plane number: {}'.format(len(sorted_filter_plane_data)))


filter_plane_info_list = list()
for plane in sorted_filter_plane_data:
    plane_params = plane.plane_params
    o3d_points = o3d.geometry.PointCloud()
    o3d_points.points = o3d.utility.Vector3dVector(plane.points)
    pcd_tree = o3d.geometry.KDTreeFlann(o3d_points)
    filter_plane_info_list.append((plane_params, plane.points, pcd_tree))


# relation_dict = dict()
# relation_count = 0
# for idx_1, plane_info_1 in enumerate(filter_plane_info_list[:2]):
#     for idx_2, plane_info_2 in enumerate(filter_plane_info_list[idx_1+1:], idx_1+1):    
#         param_1, points_1, tree_1 = plane_info_1
#         param_2, points_2, tree_2 = plane_info_2
#         normal_score = np.abs(np.dot(param_1[:, :-1], param_2[:, :-1].T)).item()
#         print("Index Pair: {}".format((idx_1, idx_2)))
#         is_connected = False
#         for valid_point in points_1:
#             k, _, dist = tree_2.search_knn_vector_3d(valid_point, 1)
#             if k != 0:
#                 is_connected = True
#                 break
#         if is_connected:
#             if(relation_dict.get(idx_1, None) is None):
#                 relation_dict[idx_1] = [[idx_2], [normal_score]]
#             else:
#                 relation_dict[idx_1][0].append(idx_2)
#                 relation_dict[idx_1][1].append(normal_score)
#             relation_count += 1
                
# ground_connected_indices = set()
# for ground_idx, (connected_indices, relations) in relation_dict.items():
#     ground_connected_indices.add(ground_idx)
#     ground_connected_indices.update(connected_indices)
# target_planes = list(plane for idx, plane in enumerate(filtered_plane_data) if idx in ground_connected_indices)

target_indices = [0, 1, 2, 4, 5, 6, 8, 9, 10, 11, 13, 14, 15, 16, 20, 22, 25, 26, 27, 31, 32, 33, 35, 36, 39, 40, 42, 43]
target_planes = list(plane_obj for idx, plane_obj in enumerate(filtered_plane_data) if idx in target_indices)
with open(os.path.join(plane_folder, 'selected_plane_objects.pkl'), 'wb') as f:
    pickle.dump(target_planes, f)
# visualization
cmap_norm = mpl.colors.Normalize(vmin=0.0, vmax=len(target_planes))
visalized_points = o3d.geometry.PointCloud()
random.shuffle(target_planes)
for idx, plane in enumerate(target_planes):
    plane_points = o3d.geometry.PointCloud()
    plane_points.points = o3d.utility.Vector3dVector(plane.points)
    plane_points.normals = o3d.utility.Vector3dVector(
        np.repeat(plane.plane_params[:, :-1], plane.points.shape[0], axis=0)
        )
    color = plt.get_cmap('rainbow')(cmap_norm(idx))[0:3]
    plane_points.paint_uniform_color(color)
    visalized_points += plane_points
    
vis = o3d.visualization.Visualizer()
vis.create_window()
vis.add_geometry(visalized_points)
vis.run()
vis.destroy_window()