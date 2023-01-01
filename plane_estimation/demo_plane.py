import os
import time
import numpy as np
import open3d as o3d
import matplotlib as mpl
import matplotlib.pyplot as plt


def generate_plane_points(paramters, point_num):
    normal_vector = np.array(paramters[:-1]) 
    normal_vector = normal_vector / np.linalg.norm(normal_vector)
    distance = paramters[-1]

    xy_points = np.random.randn(point_num, 2)
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
    # add all
    homo_points = np.column_stack((points, np.ones((points.shape[0], 1))))
    point_cluster = np.matmul(homo_points.T, homo_points)
    eig_values, eig_vectors = np.linalg.eig(point_cluster)
    plane_cost = np.min(eig_values)
    estimated_parameters = eig_vectors[:, np.argmin(eig_values)]
    temp_norm = np.linalg.norm(estimated_parameters[:-1])
    estimated_parameters = estimated_parameters / temp_norm 
    return plane_cost, estimated_parameters

def robust_parameter_estimate(points, iter_num = 10, recover_factor=1.3):
    weights = np.ones((points.shape[0], 1))
    homo_points = np.column_stack((points, np.ones((points.shape[0], 1))))
    mu = 5
    
    for _ in range(iter_num):
        # get weighted points
        weighted_points = np.multiply(np.repeat(np.sqrt(weights), points.shape[1], axis=1), points)
        homo_weighted_points = np.column_stack((weighted_points, np.ones((points.shape[0], 1))))
        point_cluster = np.matmul(homo_points.T, homo_weighted_points)
        # cost is point cluster + regularization
        eig_values, eig_vectors = np.linalg.eig(point_cluster)
        estimated_parameters = eig_vectors[:, np.argmin(eig_values)]
        temp_norm = np.linalg.norm(estimated_parameters[:-1])
        estimated_parameters = np.atleast_2d(estimated_parameters / temp_norm)
        # update weight
        squared_errors = np.square(np.dot(estimated_parameters, homo_points.T))
        weights = np.square(mu / (mu + squared_errors)).T
        mu = np.max([1.5, mu / recover_factor])
    return plane_cost, estimated_parameters, weights

if __name__ == '__main__':
    root_path = os.path.dirname(os.path.abspath(__file__))

    plane1 = generate_plane_points([1, 2, 3, 3], 100) 
    plane2 = generate_plane_points([1, 1, 1, 1], 1000) 
    plane3 = generate_plane_points([1, 3, 2, 0], 100) 
    
    noise_plane = np.vstack((plane2, plane1, plane3))
    
    plane_cost, estimated_parameters = plain_parameter_estimate(noise_plane)
    print(plane_cost, estimated_parameters)
    
    plane_cost2, estimated_parameters2, weights = robust_parameter_estimate(noise_plane)
    print(plane_cost2, estimated_parameters2)
    
    # visualization
    pcd1 = o3d.geometry.PointCloud()
    pcd1.points = o3d.utility.Vector3dVector(plane1)
    pcd1.paint_uniform_color([1, 0.706, 0])
    pcd2 = o3d.geometry.PointCloud()
    pcd2.points = o3d.utility.Vector3dVector(plane2)
    pcd2.paint_uniform_color([0, 0, 1.0])
    
    # color mapping
    cmap_norm = mpl.colors.Normalize(vmin=0.0, vmax=1.0)
    point_colors = plt.get_cmap('brg')(cmap_norm(np.squeeze(weights)))[:, 0:3]
    
    pcd_noise = o3d.geometry.PointCloud()
    pcd_noise.points = o3d.utility.Vector3dVector(noise_plane)
    pcd_noise.colors = o3d.utility.Vector3dVector(point_colors)
    
    o3d.visualization.draw_geometries([pcd_noise])
    
    