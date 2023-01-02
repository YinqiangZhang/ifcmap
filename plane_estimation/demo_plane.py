import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import open3d as o3d
import matplotlib as mpl
import matplotlib.pyplot as plt
from utils.plane_utils import generate_plane_points


class PlaneCandidate():
    def __init__(self, plane_id, points, weights, 
                 inlier_weights=None, 
                 mu_init=5):
        self.id = int(plane_id)
        self.homo_points = np.column_stack((points, np.ones((points.shape[0], 1))))
        self.weights = weights
        if inlier_weights is not None:
            self.inlier_weights = inlier_weights
        else:
            self.inlier_weights = np.ones_like(weights)
        self.point_labels = np.zeros_like(weights)
        self.mu = mu_init
        self.is_converged = False
        self.recover_factor = 1.4
        self.mu_min = 0.1
        self.reset()
    
    def reset(self):
        self.update()
    
    def update(self, update_mu=True):
        self.plane_params = self.plane_estimate()
        self.weights = self.GM_weight_estimate()
        self.point_labels = self.label_estimate()
        if update_mu:
            self.mu = self.mu_update()
        
    def plane_estimate(self):
        point_cluster = self.homo_points.T @ np.diag(np.squeeze(self.weights * self.inlier_weights)) @ self.homo_points
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
    
    def label_estimate(self):
        self.inlier_weights = np.round(self.weights)
        point_labels = np.zeros_like(self.weights)
        point_labels[self.inlier_weights == 1] = self.id
        return point_labels


if __name__ == '__main__':
    root_path = os.path.dirname(os.path.abspath(__file__))

    plane1 = generate_plane_points([1, 2, 3, 3], 200) 
    plane2 = generate_plane_points([1, 1, 1, 1], 1000) 
    plane3 = generate_plane_points([1, 3, 2, 0], 200) 
    noise_plane = np.vstack((plane2, plane1, plane3))
    
    weights = np.ones((noise_plane.shape[0], 1))
    plane_obj = PlaneCandidate(1, noise_plane, weights)
    
    for _ in range(10):
        plane_obj.update()
    print(plane_obj.plane_params)
    
    # other planes test
    new_weights = weights[np.squeeze(plane_obj.point_labels == 0)]
    new_noise_plane = noise_plane[np.squeeze(plane_obj.point_labels == 0), :]
    plane_obj2 = PlaneCandidate(2, new_noise_plane, new_weights)
    for _ in range(10):
        plane_obj2.update()
    print(plane_obj2.plane_params)
    
    # outlier_points = noise_plane[plane_obj.inlier_weights == 0]
    # plane_obj2 = PlaneCandidate(noise_plane, weights, mu_init=5)
    
    # visualization
    # pcd1 = o3d.geometry.PointCloud()
    # pcd1.points = o3d.utility.Vector3dVector(plane1)
    # pcd1.paint_uniform_color([1, 0.706, 0])
    # pcd2 = o3d.geometry.PointCloud()
    # pcd2.points = o3d.utility.Vector3dVector(plane2)
    # pcd2.paint_uniform_color([0, 0, 1.0])
    
    # color mapping
    colormap_name = ['PiYG', 'PiYG']
    cmap_norm = mpl.colors.Normalize(vmin=0.0, vmax=1.0)
    
    point_colors = plt.get_cmap('PiYG')(cmap_norm(np.squeeze(plane_obj.weights)))[:, 0:3]
    # point_colors2 = plt.get_cmap('PiYG')(cmap_norm(np.squeeze(plane_obj2.weights)))[:, 0:3]
    
    pcd_noise = o3d.geometry.PointCloud()
    pcd_noise.points = o3d.utility.Vector3dVector(noise_plane)
    pcd_noise.colors = o3d.utility.Vector3dVector(point_colors)
    
    o3d.visualization.draw_geometries([pcd_noise])
    
    # EM optimization
    # unified framework for plane extraction, plane refinement, registration, and reconstruction. 
    # plane-based global registration
    
    