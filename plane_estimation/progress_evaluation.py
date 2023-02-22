import os 
import glob
import copy
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


def save_view_point(pcd, filename):
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pcd)
    vis.run()  # user changes the view and press "q" to terminate
    param = vis.get_view_control().convert_to_pinhole_camera_parameters()
    o3d.io.write_pinhole_camera_parameters(filename, param)
    vis.destroy_window()
    
def set_material(color):
    mat_bim = vis.rendering.MaterialRecord()
    mat_bim.shader = 'defaultLitSSR'
    mat_bim.base_color = [color[0], color[1], color[2], 1.0]
    mat_bim.base_roughness = 0.0
    mat_bim.base_reflectance = 0.0
    mat_bim.base_clearcoat = 0.0
    mat_bim.thickness = 1.0
    mat_bim.transmission = 0.4
    mat_bim.absorption_distance = 10
    mat_bim.absorption_color = [color[0], color[1], color[2]]
    return mat_bim

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Select target model:')
    parser.add_argument('--index', type=int, default=0, help='Model Index')
    parser.add_argument('--aligned', type=bool, default=True, help='If use aligned points')
    parser.add_argument('--vis', type=bool, default=False, help='If use visualization')
    args = parser.parse_args()
    
    use_aligned = args.aligned
    has_visualization = args.vis 
    # read segmented planes
    root_path = os.path.dirname(os.path.abspath(__file__))
    data_folder = os.path.join(root_path, 'RealData')
    model_folder = os.path.join(data_folder, 'mesh_models')
    segment_folder = data_folder
    
    raw_pcd = o3d.io.read_point_cloud(os.path.join(data_folder, 'raw_map.ply'))
    
    with open(os.path.join(segment_folder, 'inliers.pkl'), 'rb') as f:
        inliers = pickle.load(f)
        best_trans = pickle.load(f)
    
    with open(os.path.join(segment_folder, 'initial_translation.pkl'), 'rb') as f:
        initial_translation = pickle.load(f)
        
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
    plane_indices = list(range(len(target_planes)))
    random.shuffle(plane_indices)
    for idx, plane in zip(plane_indices, target_planes):
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
    
    if use_aligned:
        aligned_pcd = copy.deepcopy(scene_pcd).transform(best_trans)
        aligned_raw_pcd = copy.deepcopy(raw_pcd).transform(best_trans)
    else:
        aligned_pcd = scene_pcd
        aligned_raw_pcd = raw_pcd
        
    # progress evaluation
    built_model = o3d.geometry.TriangleMesh()
    unbuilt_model = o3d.geometry.TriangleMesh()
    evaluated_pcd = copy.deepcopy(aligned_raw_pcd)
    pcd_tree = o3d.geometry.KDTreeFlann(aligned_raw_pcd)
    vol = o3d.visualization.SelectionPolygonVolume()
    vol.orthogonal_axis = "Z"
    removed_indices = list()
    for mesh in tqdm(model_meshes, leave=False):
        vertex_points = np.asarray(mesh.vertices)
        mesh_grid = mesh.voxelized(0.27)
        obb = o3d.geometry.OrientedBoundingBox.create_from_points(o3d.utility.Vector3dVector(vertex_points))
        search_box = o3d.geometry.OrientedBoundingBox(obb.center, obb.R, obb.extent + 0.2)
        cropped_pcd = aligned_raw_pcd.crop(search_box)
        for point_idx, point in enumerate(np.asarray(cropped_pcd.points)):
            [k, idx, _] = pcd_tree.search_knn_vector_3d(point, 1)
            removed_indices.append(idx[0])
        
        filled_set = set([tuple(index) for index in mesh_grid.sparse_indices])
        occupied_voxel_set = set([tuple(index) for index in 
                                  mesh_grid.points_to_indices(np.asarray(cropped_pcd.points)) 
                                  if tuple(index) in filled_set])
        o3d_mesh = mesh.as_open3d
        o3d_mesh.compute_vertex_normals()
        if (len(occupied_voxel_set)/mesh_grid.filled_count > 0.3):
            built_model += o3d_mesh
        else:
            unbuilt_model += o3d_mesh
            
    built_pcd = evaluated_pcd.select_by_index(removed_indices, invert=False)
    built_pcd.paint_uniform_color(np.array([0, 255, 0])/256)
    unbuilt_pcd = evaluated_pcd.select_by_index(removed_indices, invert=True)
    unbuilt_pcd.paint_uniform_color(np.array([255, 48, 48])/256)
    built_model.paint_uniform_color(np.array([102, 205, 0])/256)
    unbuilt_model.paint_uniform_color(np.array([255, 48, 48])/256)
        
    # built_mat_bim = set_material(np.array([0, 255, 0])/256)
    # unbuilt_mat_bim = set_material(np.array([255, 48, 48])/256)
    # geoms = [{'name': 'built_bim_model', 'geometry': built_model, 'material': built_mat_bim}, 
    #         {'name': 'unbuilt_bim_model', 'geometry': unbuilt_model, 'material': unbuilt_mat_bim}, 
    #         # {'name': 'built_pcd', 'geometry': built_pcd},
    #         # {'name': 'unbuilt_pcd', 'geometry': unbuilt_pcd},
    #         ]
    # vis.draw(geoms,
    #         bg_color=(1.0, 1.0, 1.0, 1.0),
    #         show_ui=True,
    #         width=1920,
    #         height=1080)
    o3d.visualization.draw_geometries([o3d_model_mesh, built_pcd, unbuilt_pcd, ])# built_model, unbuilt_model