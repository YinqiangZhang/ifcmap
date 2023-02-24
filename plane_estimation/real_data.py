import os
import time
import copy
import glob
import pickle
import trimesh
import numpy as np 
import open3d as o3d 
from tqdm import tqdm
from multiprocessing import Pool
import matplotlib as mpl
import matplotlib.pyplot as plt
from utils.registration_utils import (rough_correspondence_generating, opt_agent)


if __name__ == '__main__':
    # read segmented planes
    root_path = os.path.dirname(os.path.abspath(__file__))
    data_folder = os.path.join(root_path, 'RealData')
    model_folder = os.path.join(data_folder, 'mesh_models')
    segment_folder = data_folder
    
    model_paths = glob.glob(os.path.join(model_folder, '*.ply'))
    with open(os.path.join(segment_folder, 'selected_plane_objects.pkl'), 'rb') as f:
        target_planes = pickle.load(f)
    
    model_meshes = list()
    o3d_model_mesh = o3d.geometry.TriangleMesh()
    for model_path in model_paths:
        model_mesh = trimesh.load_mesh(model_path)
        o3d_mesh = o3d.io.read_triangle_mesh(model_path)
        o3d_mesh.compute_vertex_normals()
        model_meshes.append(model_mesh)
        o3d_model_mesh += o3d_mesh
    
    real_points = list()
    o3d_real_points = list()
    scene_pcd = o3d.geometry.PointCloud()
    cmap_norm = mpl.colors.Normalize(vmin=0.0, vmax=len(target_planes))
    for idx, plane in enumerate(target_planes):
        o3d_points = o3d.geometry.PointCloud()
        o3d_points.points = o3d.utility.Vector3dVector(plane.points)
        o3d_points.normals = o3d.utility.Vector3dVector(
            np.repeat(plane.plane_params[:, :-1], plane.points.shape[0], axis=0)
            )
        color = plt.get_cmap('nipy_spectral')(cmap_norm(idx))[0:3]
        o3d_points.paint_uniform_color(color)
        tri_points = trimesh.PointCloud(vertices=plane.points)
        real_points.append(tri_points)
        o3d_real_points.append(o3d_points)
        scene_pcd += o3d_points
    
    target_meshes = model_meshes
    target_points = real_points[:50]
    o3d.visualization.draw_geometries([scene_pcd, o3d_model_mesh])
    optimization_pairs = rough_correspondence_generating(target_meshes, target_points)
    
    start_time = time.time()
    inliers = list()
    curr_state = None
    historical_best_V = np.inf
    used_source_indices = set()
    for target_idx, pair_list in tqdm(optimization_pairs.items(), total=len(optimization_pairs), leave=False):
        data_list = list()
        valid_pair_list = list(idx for idx in pair_list if idx not in used_source_indices)
        for idx, source_idx in enumerate(valid_pair_list):
            correspondence = (target_idx, source_idx)
            curr_inliers = copy.deepcopy(inliers)
            curr_inliers.append(correspondence)
            data_list.append((target_meshes, target_points, curr_inliers, curr_state))
        pool_num = len(data_list) if len(data_list) < int(os.cpu_count()) else int(os.cpu_count())
        with Pool(pool_num) as p:
            result_list = p.map(opt_agent, data_list)
            
        best_result = min(result_list, key=lambda x:x[1])
        if (best_result[1] < min(historical_best_V * 1.5, 0.02)):
            for idx, result in enumerate(result_list):
                if result[1] == best_result[1]:
                    best_idx = idx
                    break
            best_trans, best_V, curr_state = best_result
            historical_best_V = best_V
            inliers.append((target_idx, valid_pair_list[best_idx]))
            # used_source_indices.add(valid_pair_list[best_idx])
            print('\n Current Average V: {}'.format(best_V))
            print('\n Current correspondences: {}'.format(inliers))
            # aligned_pcd = copy.deepcopy(scene_pcd)
            # aligned_pcd.transform(best_trans)
            # o3d.visualization.draw_geometries([aligned_pcd, o3d_model_mesh])
    
    print('Total computation time: {} s'.format(time.time() - start_time))
    
    with open(os.path.join(data_folder, 'inliers.pkl'), 'wb') as f:
        pickle.dump(inliers, f)
        pickle.dump(best_trans, f)
    
    aligned_pcd = copy.deepcopy(scene_pcd)
    aligned_pcd.transform(best_trans)
    o3d.visualization.draw_geometries([aligned_pcd, o3d_model_mesh])