import trimesh
import numpy as np
import open3d as o3d
import matplotlib as mpl
import matplotlib.pyplot as plt
from utils.plane import PlaneCandidate
from utils.primitive_registor import PrimitiveRegistor

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
    print('Reduction rate: {:.2f} %'.format(correspondence_num / (len(point_list)*len(mesh_list)) * 100))
    return correspondence_dict

def get_model_meshes(model_paths):
    model_meshes = list()
    o3d_model_mesh = o3d.geometry.TriangleMesh()
    for model_path in model_paths:
        model_mesh = trimesh.load_mesh(model_path)
        o3d_mesh = o3d.io.read_triangle_mesh(model_path)
        o3d_mesh.compute_vertex_normals()
        model_meshes.append(model_mesh.split())
        o3d_model_mesh += o3d_mesh
    return model_meshes, o3d_model_mesh

def get_real_points(segments_paths, init_trans=np.identity(4)):
    real_points = list()
    o3d_real_points = list()
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
            o3d_real_points.append(valid_pcd)
            scene_points_list.append(trimesh_points)
        scene_points_list = sorted(scene_points_list, key=lambda x: len(x.vertices), reverse=True)
        o3d_real_points = sorted(o3d_real_points, key=lambda x: len(x.points), reverse=True)
        real_points.append(scene_points_list)
    return real_points, o3d_real_points, scene_pcd