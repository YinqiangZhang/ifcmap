import os 
import glob
import copy
import random
import pickle
import trimesh
import argparse
import numpy as np
import open3d as o3d
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
    
def set_material():
    mat_bim = vis.rendering.MaterialRecord()
    # mat_box.shader = 'defaultLitTransparency'
    mat_bim.shader = 'defaultLitSSR'
    mat_bim.base_color = [0.1, 0.1, 0.2, 1.0]
    mat_bim.base_roughness = 0.1
    mat_bim.base_reflectance = 0.0
    mat_bim.base_clearcoat = 1.0
    mat_bim.thickness = 1.0
    mat_bim.transmission = 0.5
    mat_bim.absorption_distance = 10
    mat_bim.absorption_color = [0.1, 0.1, 0.2]
    return mat_bim

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Select target model:')
    parser.add_argument('--index', type=int, default=0, help='Model Index')
    parser.add_argument('--aligned', type=bool, default=False, help='If use aligned points')
    parser.add_argument('--vis', type=bool, default=False, help='If use visualization')
    args = parser.parse_args()
    
    use_aligned = args.aligned
    has_visualization = args.vis 
    # read segmented planes
    root_path = os.path.dirname(os.path.abspath(__file__))
    data_folder = os.path.join(root_path, 'RealData')
    model_folder = os.path.join(data_folder, 'mesh_models')
    segment_folder = data_folder
    
    with open(os.path.join(segment_folder, 'inliers.pkl'), 'rb') as f:
        inliers = pickle.load(f)
        best_trans = pickle.load(f)
        
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
    else:
        aligned_pcd = scene_pcd
    
    # if has_visualization:
    #     mat_bim = set_material()
    #     geoms = [{'name': 'bim_model', 'geometry': o3d_model_mesh, 'material': mat_bim}, 
    #             {'name': 'scene_pcd', 'geometry': aligned_pcd}]
    #     vis.draw(geoms,
    #             bg_color=(1.0, 1.0, 1.0, 1.0),
    #             show_ui=True,
    #             width=1920,
    #             height=1080)
    # else:
    o3d.visualization.draw_geometries([o3d_model_mesh, aligned_pcd])