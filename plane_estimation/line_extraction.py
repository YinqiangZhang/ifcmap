import os 
import glob
import pickle
import numpy as np 
import open3d as o3d 
import matplotlib.pyplot as plt 
import matplotlib as mpl
import numpy as np
import itertools

def get_intersection_line(plane1, plane2):
    pass


if __name__ == '__main__':
    # read segmented plane
    root_path = os.path.dirname(os.path.abspath(__file__))    
    plane_data_path = os.path.join(root_path, 'plane_data')
    with open(os.path.join(plane_data_path, 'plane_objects.pkl'), 'rb') as f:
        plane_data = pickle.load(f)

    cmap_norm = mpl.colors.Normalize(vmin=0.0, vmax=len(plane_data))
    sorted_plane_data = sorted(plane_data, key=lambda x:np.sum(x.inliers==1), reverse=True)

    # plane processing
    o3d_planes = o3d.geometry.PointCloud()
    o3d_plane_list = list()
    obb_list = list()
    ground_planes = o3d.geometry.PointCloud()
    for idx, plane in enumerate(sorted_plane_data):
        plane_points = o3d.geometry.PointCloud()
        inlier_points = plane.points[np.squeeze(plane.inliers==1), :]
        plane_points.points = o3d.utility.Vector3dVector(inlier_points)
        plane_points.normals = o3d.utility.Vector3dVector(
            np.repeat(plane.plane_params[:, :-1], inlier_points.shape[0], axis=0)
            )
        color = plt.get_cmap('nipy_spectral')(cmap_norm(plane.id))[0:3]
        plane_points.paint_uniform_color(color)
        obb = plane_points.get_oriented_bounding_box()
        obb.color = (0,0,0)
        obb_list.append(obb)
        o3d_plane_list.append(plane_points)
        o3d_planes += plane_points
    
    candidate_id_pairs = list()
    for idx_1 in range(len(obb_list)):
        for idx_2 in range(idx_1+1, len(obb_list)):
            center_dist = np.linalg.norm(obb_list[idx_1].center - obb_list[idx_2].center)
            radius_0 = np.linalg.norm(np.asarray(obb_list[idx_1].get_box_points())[0] - obb_list[idx_1].center)
            radius_1 = np.linalg.norm(np.asarray(obb_list[idx_2].get_box_points())[0] - obb_list[idx_2].center)
            if (radius_0+radius_1 > center_dist):
                candidate_id_pairs.append([idx_1, idx_2])
    
    overlap_id_pairs = list()
    for idx_1, idx_2 in candidate_id_pairs:
        diff_01 = obb_list[idx_1].get_point_indices_within_bounding_box(o3d_plane_list[idx_2].points)
        diff_10 = obb_list[idx_2].get_point_indices_within_bounding_box(o3d_plane_list[idx_1].points)
        if len(diff_01) != 0 and len(diff_10) != 0:
            overlap_id_pairs.append((idx_1, idx_2))
    
    plane_param_pairs = list()
    for idx_1, idx_2 in overlap_id_pairs:
        plane_param1 = sorted_plane_data[idx_1].plane_params
        plane_param2 = sorted_plane_data[idx_2].plane_params
        
        relative_angle = np.arccos(np.dot(plane_param1[:, :-1], plane_param2[:, :-1].T))
        print(relative_angle *180.0/ np.pi)
        param_set = np.vstack((plane_param1, plane_param2))
    
    # included planes visualization
    plane_ids = set()
    for idx_1, idx_2 in overlap_id_pairs:
        plane_ids.add(idx_1)
        plane_ids.add(idx_2)
    
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    # vis.add_geometry(o3d_planes)
    # [vis.add_geometry(obb) for obb in obb_list]
    [vis.add_geometry(o3d_plane_list[plane_id]) for plane_id in plane_ids]
    vis.run()
    vis.destroy_window()