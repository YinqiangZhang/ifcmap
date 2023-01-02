import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import open3d as o3d
import matplotlib as mpl
import matplotlib.pyplot as plt
from utils.plane_utils import generate_plane_points


class PlaneCandidate():
    def __init__(self, plane_id, points, weights, inliers=None, mu0=5):
        self.id = int(plane_id)
        self.homo_points = np.column_stack((points, np.ones((points.shape[0], 1))))
        self.weights = weights
        
        self.inliers = inliers if inliers is not None else np.ones_like(weights)
        self.inlier_homo_points = self.homo_points
        self.inlier_weights = self.weights
        
        self.mu = mu0
        self.recover_factor = 1.4
        self.mu_min = 0.1
        
        self.reset()
    
    def reset(self):
        self.update()
    
    def update(self, update_mu=True):
        self.plane_params = self.plane_estimate()
        self.weights = self.GM_weight_estimate()
        self.inliers, self.inlier_homo_points, self.inlier_weights = self.inliner_estimate()
        if update_mu:
            self.mu = self.mu_update()
        
    def plane_estimate(self):
        point_cluster = self.inlier_homo_points.T @ np.diag(np.squeeze(self.inlier_weights)) @ self.inlier_homo_points
        eig_values, eig_vectors = np.linalg.eig(point_cluster)
        parameters = eig_vectors[:, np.argmin(eig_values)]
        parameters = np.atleast_2d(parameters / np.linalg.norm(parameters[:-1]))
        return parameters
    
    def GM_weight_estimate(self):
        squared_errors = np.square(np.dot(self.plane_params, self.homo_points.T))
        weights = np.square(self.mu / (self.mu + squared_errors)).T
        return weights
    
    def TLS_weight_update(self):
        pass
    
    def mu_update(self):
        temp_mu = np.max([self.mu_min, self.mu / self.recover_factor])
        return temp_mu
    
    def inliner_estimate(self):
        inliers = np.round(self.weights)
        inlier_indices = np.squeeze(inliers == 1)
        inlier_homo_points = self.homo_points[inlier_indices, :]
        inlier_weights = self.weights[inlier_indices, :]
        return inliers, inlier_homo_points, inlier_weights
    
    def get_inlier_points(self):
        return self.inlier_homo_points[:, :-1]


if __name__ == '__main__':
    root_path = os.path.dirname(os.path.abspath(__file__))

    plane1 = generate_plane_points([1, 2, 3, 3], 300) 
    plane2 = generate_plane_points([1, 1, 1, 1], 1000) 
    plane3 = generate_plane_points([1, 3, 2, 0], 200) 
    noise_plane = np.vstack((plane2, plane1, plane3))
    
    weights = np.ones((noise_plane.shape[0], 1))
    plane_obj = PlaneCandidate(1, noise_plane, weights)
    
    for _ in range(10):
        plane_obj.update()
    print(plane_obj.plane_params)
    
    # other planes test
    new_weights = weights[np.squeeze(plane_obj.inliers == 0)]
    new_noise_plane = noise_plane[np.squeeze(plane_obj.inliers == 0), :]
    plane_obj2 = PlaneCandidate(2, new_noise_plane, new_weights)
    for _ in range(10):
        plane_obj2.update()
    print(plane_obj2.plane_params)
    
    # color mapping
    colormap_name = ['Greens', 'Purples']
    cmap_norm = mpl.colors.Normalize(vmin=0.0, vmax=1.0)
    
    # plane 1 
    inlier_points1 = plane_obj.get_inlier_points()
    point_colors1 = plt.get_cmap(colormap_name[0])(cmap_norm(np.squeeze(plane_obj.inlier_weights)))[:, 0:3]
    pcd_noise = o3d.geometry.PointCloud()
    pcd_noise.points = o3d.utility.Vector3dVector(inlier_points1)
    pcd_noise.colors = o3d.utility.Vector3dVector(point_colors1)
    
    # plane 2 
    inlier_points2 = plane_obj2.get_inlier_points()
    point_colors2 = plt.get_cmap(colormap_name[1])(cmap_norm(np.squeeze(plane_obj2.inlier_weights)))[:, 0:3]
    pcd_noise2 = o3d.geometry.PointCloud()
    pcd_noise2.points = o3d.utility.Vector3dVector(inlier_points2)
    pcd_noise2.colors = o3d.utility.Vector3dVector(point_colors2)
    
    o3d.visualization.draw_geometries([pcd_noise, pcd_noise2])
    
    # EM optimization
    # unified framework for plane extraction, plane refinement, registration, and reconstruction. 
    # plane-based global registration
    
    