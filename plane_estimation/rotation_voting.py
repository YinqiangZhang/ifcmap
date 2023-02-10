import os 
import copy
import pickle
import random
import numpy as np 
import open3d as o3d 
import matplotlib.pyplot as plt 
import matplotlib as mpl
import numpy as np
from utils.plane import PlaneCandidate
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt


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
            score = np.arccos(np.squeeze(cos_sim)) + 10*delta_angle
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
    
with open(os.path.join(plane_folder, 'selected_plane_objects.pkl'), 'rb') as f:
    target_planes = pickle.load(f)
    
base_angles = np.array([np.pi/24, np.pi/6, np.pi/30])
r = R.from_rotvec(base_angles) #+ 0.001 * np.random.randn(3)
rot_mat = r.as_matrix()
source_params_list = list()
target_params_list = list()
target_points = o3d.geometry.PointCloud()
source_points = o3d.geometry.PointCloud()
for idx, plane in enumerate(target_planes):
    source_params_list.append(plane.plane_params)
    o3d_points = o3d.geometry.PointCloud()
    o3d_points.points = o3d.utility.Vector3dVector(plane.points)
    o3d_points.normals = o3d.utility.Vector3dVector(
        np.repeat(plane.plane_params[:, :-1], plane.points.shape[0], axis=0)
        )
    source_points += o3d_points
    if idx == 0 or 1 < idx < 14:
    # if  idx < 40:
        result_points = copy.deepcopy(o3d_points).rotate(rot_mat)
        plane_obj = PlaneCandidate(idx, np.asarray(result_points.points), np.ones((len(result_points.points), 1)))
        plane_obj.update()
        target_params_list.append(plane_obj.plane_params)
        target_points += o3d_points

target_points.rotate(R.from_rotvec(base_angles).as_matrix())
# target_points.translate(np.array([1.0, 1.0, 1.0]))
source_points.paint_uniform_color(np.array([65, 105, 225])/255)
target_points.paint_uniform_color(np.array([218, 165, 32])/255)

association_dict = dict()
norm_list = np.array([])
random.shuffle(target_params_list)
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

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(projection='3d')
u, v = np.mgrid[0:2 * np.pi:30j, 0:np.pi:20j]
x = np.cos(u) * np.sin(v)
y = np.sin(u) * np.sin(v)
z = np.cos(v)
ax.plot_wireframe(x, y, z, edgecolor="k", linewidth=0.5)
for idx, (vectors, _, _) in enumerate(cluster_info[:]):
    ax.scatter(vectors[:, 0], vectors[:, 1], vectors[:, 2], s=20, alpha=0.5)
    print('Label: {}, size: {}'.format(idx, vectors.shape[0]))
ax.set_box_aspect((1, 1, 0.9))
# ax.set_xticks([])
# ax.set_yticks([])
# ax.set_zticks([])
# plt.axis('off')
plt.show()

rot_mat_list = list()
association_list = list()
for vectors, center, associations in cluster_info[:1]:
    print(center)
    new_center = np.median(vectors, axis=0)
    new_center[:-1] /= np.linalg.norm(new_center[:-1])
    rot_mat_list.append(R.from_rotvec(new_center[:-1] * new_center[-1]))
    association_list.append(associations) 

# initial results by rotation voting
initial_alignment_points = copy.deepcopy(target_points)
initial_alignment_points.rotate(rot_mat_list[0].as_matrix())
initial_alignment_points.paint_uniform_color(np.array([218, 165, 32])/255)

reg_p2l = o3d.pipelines.registration.registration_icp(
    initial_alignment_points, source_points, 0.02, np.identity(4),
    o3d.pipelines.registration.TransformationEstimationPointToPlane())
initial_alignment_points.transform(reg_p2l.transformation)

vis = o3d.visualization.Visualizer()
vis.create_window()
vis.add_geometry(source_points)
vis.add_geometry(initial_alignment_points)
vis.run()
vis.destroy_window()