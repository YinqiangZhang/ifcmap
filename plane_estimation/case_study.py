import os
import time
import copy
import glob
import pickle
import trimesh
import argparse
import numpy as np 
import open3d as o3d 
from tqdm import tqdm
from multiprocessing import Pool
from scipy.spatial.transform import Rotation as R
from utils.registration_utils import (get_model_meshes, 
                                      rough_correspondence_generating, 
                                      get_real_points, 
                                      opt_agent)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Select target model:')
    parser.add_argument('--index', type=int, default=0, help='Model Index')
    args = parser.parse_args()
    # read segmented planes
    root_path = os.path.dirname(os.path.abspath(__file__))
    data_folder = os.path.join(root_path, 'CaseStudy')
    model_folder = os.path.join(data_folder, 'CaseStudyModels')
    segment_folder = os.path.join(data_folder, 'CaseStudySegments')
    
    model_paths = glob.glob(os.path.join(model_folder, '*.ply'))
    segments_paths = [glob.glob(os.path.join(segment_folder, folder, '*.ply')) \
                      for folder in os.listdir(segment_folder)]
    
    # set a init random transformation matrix
    case_index = args.index
    axis_angle_rep = np.array([np.pi/24, np.pi/6, np.pi/30])
    r = R.from_rotvec(axis_angle_rep)
    init_trans = np.identity(4)
    init_trans[:-1, :-1] = r.as_matrix()
    init_trans[:-1, -1] = np.array([-10.0, -10.0, -2.0])
    model_meshes, o3d_mesh = get_model_meshes(model_paths[case_index:case_index+1])
    real_points, o3d_real_points, scene_pcd = get_real_points(segments_paths[case_index:case_index+1], init_trans)
    
    target_meshes = model_meshes[0]
    target_points = real_points[0][:50]
    o3d.visualization.draw_geometries([o3d_mesh, scene_pcd])
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
        pool_num = len(data_list) if len(data_list) < int(os.cpu_count()/2) else int(os.cpu_count()/2)
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
            # o3d.visualization.draw_geometries([aligned_pcd, o3d_mesh])
    
    print('Total computation time: {} s'.format(time.time() - start_time))
    
    with open(os.path.join(data_folder, f'CaseStudy{case_index}_inliers.pkl'), 'wb') as f:
        pickle.dump(inliers, f)
        pickle.dump(best_trans, f)
    
    aligned_pcd = copy.deepcopy(scene_pcd)
    aligned_pcd.transform(best_trans)
    o3d.visualization.draw_geometries([aligned_pcd, o3d_mesh])