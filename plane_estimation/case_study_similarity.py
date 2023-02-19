import os
import time
import copy
import glob
import pickle
import trimesh
import numpy as np 
import open3d as o3d 
from tqdm import tqdm
import matplotlib as mpl
import matplotlib.pyplot as plt
from multiprocessing import Pool
from utils.plane import PlaneCandidate
from scipy.spatial.transform import Rotation as R
from utils.primitive_registor import PrimitiveRegistor
from itertools import combinations


def opt_agent(data):
    mesh_list, points_list, correspondences, state = data
    registor = PrimitiveRegistor(mesh_list, points_list, correspondences)
    if state is not None:
        registor.state = state
    registor.set_damping()
    result_trans, _ = registor.optimize()
    average_V = registor.get_average_potential()
    print('Index: {}, Average V: {}'.format(correspondences[-1], average_V))
    return (result_trans, average_V, registor.state)

def rough_correspondence_generating(mesh_list, point_list):
    correspondence_dict = dict()
    correspondence_pairs = list()
    correspondence_num = 0
    for points_idx, points in enumerate(point_list):
        for mesh_idx, mesh in enumerate(mesh_list):
            mesh_area = mesh.area
            points_approx_area = points.convex_hull.area
            if points_approx_area > 2 * mesh_area:
                continue
            correspondence_num += 1
            if correspondence_dict.get(points_idx, None) is None:
                correspondence_dict[points_idx] = [mesh_idx]
            else:
                correspondence_dict[points_idx].append(mesh_idx)
            correspondence_pairs.append((points_idx, mesh_idx))
    print('Reduction rate: {:.2f} %'.format(correspondence_num / (len(point_list)*len(mesh_list)) * 100))
    return correspondence_dict, correspondence_pairs

def get_model_meshes(model_paths):
    model_meshes = list()
    o3d_model_mesh = o3d.geometry.TriangleMesh()
    for model_path in model_paths:
        model_mesh = trimesh.load_mesh(model_path)
        o3d_mesh = o3d.io.read_triangle_mesh(model_path)
        o3d_mesh.compute_vertex_normals()
        model_meshes.append(model_mesh.split())
        o3d_model_mesh += o3d_mesh
    return model_meshes, o3d_mesh

def get_real_points(segments_paths, init_trans=np.identity(4)):
    real_points = list()
    for segment_list in segments_paths:
        cmap_norm = mpl.colors.Normalize(vmin=0.0, vmax=len(segment_list))
        scene_points_list = list()
        scene_pcd = o3d.geometry.PointCloud()
        for idx, segment in enumerate(segment_list):
            segment_pcd = o3d.io.read_point_cloud(segment)
            segment_pcd.transform(init_trans)
            points = np.asarray(segment_pcd.points)
            plane_obj = PlaneCandidate(idx, points, np.ones((points.shape[0], 1)))
            plane_obj.update()
            valid_points = points[np.where(plane_obj.inliers == 1)[0], :]
            trimesh_points = trimesh.PointCloud(vertices=valid_points)
            valid_pcd = o3d.geometry.PointCloud()
            valid_pcd.points = o3d.utility.Vector3dVector(valid_points)
            color = plt.get_cmap('nipy_spectral')(cmap_norm(idx))[0:3]
            valid_pcd.paint_uniform_color(color)
            scene_pcd += valid_pcd
            scene_points_list.append(trimesh_points)
        scene_points_list = sorted(scene_points_list, key=lambda x: len(x.vertices), reverse=True)
        real_points.append(scene_points_list)
    return real_points, scene_pcd

if __name__ == '__main__':
    
    # read segmented planes
    root_path = os.path.dirname(os.path.abspath(__file__))
    data_folder = os.path.join(root_path, 'CaseStudy')
    model_folder = os.path.join(data_folder, 'CaseStudyModels')
    segment_folder = os.path.join(data_folder, 'CaseStudySegments')
    
    model_paths = glob.glob(os.path.join(model_folder, '*.ply'))
    segments_paths = [glob.glob(os.path.join(segment_folder, folder, '*.ply')) \
                      for folder in os.listdir(segment_folder)]
    
    # set a init random transformation matrix
    axis_angle_rep = np.array([np.pi/24, np.pi/6, np.pi/30])
    r = R.from_rotvec(axis_angle_rep)
    init_trans = np.identity(4)
    init_trans[:-1, :-1] = r.as_matrix()
    init_trans[:-1, -1] = np.array([-10.0, -10.0, -2.0])
    model_meshes, o3d_mesh = get_model_meshes(model_paths[0:1])
    real_points, scene_pcd = get_real_points(segments_paths[0:1], init_trans)
    
    target_meshes = model_meshes[0]
    target_points = real_points[0][:50]
    o3d.visualization.draw_geometries([o3d_mesh, scene_pcd])
    
    optimization_dict, correspondence_pairs = rough_correspondence_generating(target_meshes, target_points)
    
    start_time = time.time()
    computation_queue = list()
    for corr_0 in correspondence_pairs:
        # points = target_points[corr_0[0]]
        # meshes = target_meshes[corr_0[1]]
        # mesh_centroid = np.asarray(meshes.vertices).mean(axis=0)
        # point_centroid = points.centroid
        # init_trans = np.identity(4)
        # init_trans[:-1, -1] = mesh_centroid - point_centroid
        # points.apply_transform(init_trans)
        computation_queue.append((target_meshes, target_points, [corr_0], None)) 
    
    with Pool(os.cpu_count()) as p:
        result_list = p.map(opt_agent, computation_queue)
    
    valid_correspondences = list()
    valid_result_list = list()
    for result, pair in zip(result_list, correspondence_pairs):
        if result[1] <= 0.01:
            valid_correspondences.append(pair)
            valid_result_list.append(result)
    
    print('Number of valid correspondences: {}'.format(len(valid_correspondences)))
    print('Total computation time: {} s'.format(time.time() - start_time))
    
    # computation_queue = list()
    # for corr_0, corr_1 in combinations(valid_correspondences, 2):
    #     points = [target_points[corr_0[0]], target_points[corr_1[0]]]
    #     meshes = [target_meshes[corr_0[1]], target_meshes[corr_1[1]]]
    #     # mesh_centroid = np.asarray(meshes.vertices).mean(axis=0)
    #     # point_centroid = points.centroid
    #     # init_trans = np.identity(4)
    #     # init_trans[:-1, -1] = mesh_centroid - point_centroid
    #     # points.apply_transform(init_trans)
    #     computation_queue.append((meshes, points, [(0, 0), (1, 1)], None)) 
    
    # with Pool(os.cpu_count()) as p:
    #     valid_result_list = p.map(opt_agent, computation_queue)
        
    with open(os.path.join(data_folder, 'valid_correspondences.pkl'), 'wb') as f:
        pickle.dump(valid_correspondences, f)
    
    with open(os.path.join(data_folder, 'valid_correspondences.pkl'), 'rb') as f:
        valid_correspondences = pickle.load(f)
    
    # aligned_pcd = copy.deepcopy(scene_pcd)
    # aligned_pcd.transform(best_trans)
    # aligned_pcd.paint_uniform_color(np.array([218, 165, 32])/255)
    # o3d.visualization.draw_geometries([aligned_pcd, o3d_mesh])