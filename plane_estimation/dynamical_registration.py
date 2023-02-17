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
from utils.primitive_registor import PrimitiveRegistor

# TODO: exchange the name of source and target
def opt_agent(data):
    mesh_list, points_list, correspondences, state = data
    registor = PrimitiveRegistor(mesh_list, points_list, correspondences)
    if state is not None:
        registor.state = state
    registor.set_damping()
    result_trans, _ = registor.optimize()
    average_V = registor.get_average_potential()
    # print('Index: {}, Average V: {}'.format(correspondences[-1], average_V))
    return (result_trans, average_V, registor.state)

def rough_correspondence_generating(mesh_list, point_list):
    correspondence_dict = dict()
    correspondence_num = 0
    for points_idx, points in enumerate(point_list):
        for mesh_idx, mesh in enumerate(mesh_list):
            mesh_area = mesh.area
            points_approx_area = points.convex_hull.area
            if points_approx_area > 5 * mesh_area:
                continue
            correspondence_num += 1
            if correspondence_dict.get(points_idx, None) is None:
                correspondence_dict[points_idx] = [mesh_idx]
            else:
                correspondence_dict[points_idx].append(mesh_idx)
    print('Reduction rate: {:.2f} %'.format(correspondence_num / (len(point_list)*len(mesh_list)) * 100))
    return correspondence_dict

if __name__ == '__main__':
    
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
    # o3d.visualization.draw_geometries([source_points, target_points])
    
    # exhausted searching the point clouds from large to small
    optimization_pairs = rough_correspondence_generating(model_trimesh_list, target_points_list)
    
    # record computation time
    start_time = time.time()
    inliers = list()
    curr_state = None
    used_source_indices = set()
    for target_idx, pair_list in tqdm(optimization_pairs.items(), total=len(optimization_pairs), leave=False):
        data_list = list()
        valid_pair_list = list(idx for idx in pair_list if idx not in used_source_indices)
        for idx, source_idx in enumerate(valid_pair_list):
            correspondence = (target_idx, source_idx)
            curr_inliers = copy.deepcopy(inliers)
            curr_inliers.append(correspondence)
            data_list.append((model_trimesh_list, target_points_list, curr_inliers, curr_state))
        pool_num = len(data_list) if len(data_list) < os.cpu_count() else os.cpu_count()
        with Pool(pool_num) as p:
            result_list = p.map(opt_agent, data_list)
            
        best_result = min(result_list, key=lambda x:x[1])
        if (best_result[1] < 0.025):
            for idx, result in enumerate(result_list):
                if result[1] == best_result[1]:
                    best_idx = idx
                    break
            best_trans, best_V, curr_state = best_result
            inliers.append((target_idx, valid_pair_list[best_idx]))
            used_source_indices.add(valid_pair_list[best_idx])
            print('\n Current correspondences: {}'.format(inliers))
            # aligned_points = copy.deepcopy(target_points)
            # aligned_points.transform(best_trans)
            # o3d.visualization.draw_geometries([aligned_points, o3d_model_mesh])
    
    print('Total computation time: {} s'.format(time.time() - start_time))
    initial_alignment_points = copy.deepcopy(target_points)
    initial_alignment_points.transform(best_trans)
    initial_alignment_points.paint_uniform_color(np.array([218, 165, 32])/255)
    
    with open(os.path.join(model_folder, 'inliers.pkl'), 'wb') as f:
        pickle.dump(inliers, f)
        
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(o3d_model_mesh)
    vis.add_geometry(initial_alignment_points)
    vis.run()
    vis.destroy_window()