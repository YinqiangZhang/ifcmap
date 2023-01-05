import numpy as np
import open3d as o3d

def generate_plane_points(paramters, point_num, range=[-2.0, 2.0]):
    normal_vector = np.array(paramters[:-1]) 
    normal_vector = normal_vector / np.linalg.norm(normal_vector)
    distance = paramters[-1]
    xy_points = np.random.uniform(range[0], range[1], size=(point_num, 2))
    xyz_points = list()
    for xy_point in xy_points:
        error = np.random.normal(0, 0.05)
        z = (error - distance - np.dot(xy_point, normal_vector[:-1]))/normal_vector[-1]
        xyz_points.append([xy_point[0], xy_point[1], z])
    
    return np.array(xyz_points)

def generate_plane_o3d_points(paramters, point_num):
    points = generate_plane_points(paramters, point_num)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    return pcd

def plain_parameter_estimate(points):
    homo_points = np.column_stack((points, np.ones((points.shape[0], 1))))
    point_cluster = np.matmul(homo_points.T, homo_points)
    eig_values, eig_vectors = np.linalg.eig(point_cluster)
    plane_cost = np.min(eig_values)
    estimated_parameters = eig_vectors[:, np.argmin(eig_values)]
    temp_norm = np.linalg.norm(estimated_parameters[:-1])
    estimated_parameters = estimated_parameters / temp_norm 
    return plane_cost, estimated_parameters

def rotate_view(vis):
    ctr = vis.get_view_control()
    ctr.rotate(2.0, 0.0)
    return False