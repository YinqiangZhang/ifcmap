import os 
import glob
import copy
import time
import random
import pickle
import trimesh
import argparse
import numpy as np
import open3d as o3d
from tqdm import tqdm
import matplotlib as mpl
import matplotlib.pyplot as plt
import open3d.visualization as vis
from utils.primitive_registor import PrimitiveRegistor
    

def main():
    # read segmented planes
    root_path = os.path.dirname(os.path.abspath(__file__))
    data_folder = os.path.join(root_path, 'RealData')
    model_folder = os.path.join(data_folder, 'simplified_mesh_models')
    segment_folder = data_folder
    with open(os.path.join(segment_folder, 'simplified_inliers.pkl'), 'rb') as f:
        inliers = pickle.load(f)
        best_trans = pickle.load(f)
    with open(os.path.join(segment_folder, 'initial_translation.pkl'), 'rb') as f:
        initial_translation = pickle.load(f)
    model_paths = sorted(glob.glob(os.path.join(model_folder, '*.ply')))
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
    plane_indices = list(range(len(target_planes)))
    random.shuffle(plane_indices)
    for idx, plane in zip(plane_indices, target_planes):
        o3d_points = o3d.geometry.PointCloud()
        o3d_points.points = o3d.utility.Vector3dVector(plane.points)
        o3d_points.normals = o3d.utility.Vector3dVector(
            np.repeat(plane.plane_params[:, :-1], plane.points.shape[0], axis=0)
            )
        color = plt.get_cmap('rainbow')(cmap_norm(idx))[0:3]
        o3d_points.paint_uniform_color(color)
        tri_points = trimesh.PointCloud(vertices=plane.points)
        real_points.append(tri_points)
        o3d_real_points.append(o3d_points)
        scene_pcd += o3d_points
        
    registor = PrimitiveRegistor(model_meshes, real_points, [], 0.005)
    for idx, inlier in enumerate(inliers):
        registor.add_correspondence(inlier)
        registor.set_damping()
        start_time = time.time() 
        trans, total_V, _ = registor.optimize()
        average_V = registor.get_average_potential()
        print('Optimization time: {}'.format(time.time()-start_time))
        print('Add Inlier: {}, Total V: {}, Average V: {}'.format(inlier, total_V, average_V))
        # aligned_pcd = copy.deepcopy(scene_pcd)
        # aligned_pcd.transform(trans)
        # o3d.visualization.draw_geometries([aligned_pcd, o3d_model_mesh])
        
        
if __name__ == '__main__':
    main()