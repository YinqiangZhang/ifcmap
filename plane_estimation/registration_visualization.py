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
        color = plt.get_cmap('rainbow')(cmap_norm(idx))[0:3]
        o3d_points.paint_uniform_color(color)
        tri_points = trimesh.PointCloud(vertices=plane.points)
        real_points.append(tri_points)
        o3d_real_points.append(o3d_points)
        scene_pcd += o3d_points

    camera_parameters = o3d.io.read_pinhole_camera_parameters(os.path.join(root_path, 'configs', 'new_video_configs.json'))
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name='Registration', 
                    width=960, 
                    height=540, 
                    left=0, 
                    top=0, 
                    visible=True)
    # vis.add_geometry(scene_pcd)
    # vis.add_geometry(o3d_model_mesh)
    # vis.get_view_control().convert_from_pinhole_camera_parameters(camera_parameters)
    # vis.get_render_option().load_from_json(os.path.join(root_path, 'configs', 'render_opt.json'))
    # vis.run()
    img_count = 0
    registor = PrimitiveRegistor(model_meshes, real_points, [], 0.005)
    for idx, inlier in enumerate(inliers):
        registor.add_correspondence(inlier)
        registor.set_damping()
        for n_iter in range(600):
            result_trans, total_V, is_done = registor.optimize(total_iter_num=5)
            average_V = registor.get_average_potential()
            line_set, ancher_points_set, key_points_set = registor.get_inlier_lineset()
            print('Add Inlier: {}, Total V: {}, Average V: {}'.format(inlier, total_V, average_V))
            aligned_pcd = copy.deepcopy(scene_pcd).transform(result_trans)
            vis.clear_geometries()
            vis.add_geometry(line_set)
            vis.add_geometry(ancher_points_set)
            vis.add_geometry(key_points_set)
            vis.add_geometry(aligned_pcd)
            vis.add_geometry(o3d_model_mesh)
            vis.get_view_control().convert_from_pinhole_camera_parameters(camera_parameters)
            vis.get_render_option().load_from_json(os.path.join(root_path, 'configs', 'render_opt.json'))
            vis.update_renderer()
            vis.poll_events()
            # image = vis.capture_screen_float_buffer(False)
            # plt.imsave(os.path.join(root_path, 'video_figs', "{:05d}.png".format(img_count)), np.asarray(image))
            img_count += 1
            if is_done:
                break
        registor.reset_to_still()


if __name__ == '__main__':
    main()
    