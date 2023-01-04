import numpy as np

class PlaneCandidate():
    def __init__(self, plane_id, points, weights, mu0=20):
        self.id = int(plane_id)
        self.points = points
        self.homo_points = self.generate_homo_points(points)
        self.weights = weights
        
        self.inliers = np.ones_like(weights)
        self.inlier_points = self.points
        self.inlier_weights = self.weights
        
        self.mu = mu0
        self.recover_factor = 1.4
        self.mu_min = 0.1
        
        self.update()
        
    def generate_homo_points(self, points):
        return np.column_stack((points, np.ones((points.shape[0], 1))))
    
    def update(self, update_mu=True):
        self.plane_params = self.plane_estimate()
        self.weights = self.GM_weight_estimate()
        self.inliers, self.inlier_points, self.inlier_weights = self.inliner_estimate()
        if update_mu:
            self.mu = self.mu_update()
        
    def plane_estimate(self):
        inlier_homo_points = self.generate_homo_points(self.inlier_points)
        weighted_inlier_homo_points = np.multiply(inlier_homo_points, self.inlier_weights)
        point_cluster = np.matmul(weighted_inlier_homo_points.T, weighted_inlier_homo_points)
        eig_values, eig_vectors = np.linalg.eig(point_cluster)
        parameters = eig_vectors[:, np.argmin(eig_values)]
        parameters = np.atleast_2d(parameters / np.linalg.norm(parameters[:-1]))
        return parameters
    
    def GM_weight_estimate(self):
        squared_errors = np.square(np.dot(self.plane_params, self.homo_points.T))
        weights = np.square(self.mu / (self.mu + squared_errors)).T
        return weights
    
    def TLS_weight_update(self):
        errors = np.dot(self.plane_params, self.homo_points.T)
    
    def mu_update(self):
        return np.max([self.mu_min, self.mu / self.recover_factor])
    
    def inliner_estimate(self):
        inliers = np.round(self.weights)
        inlier_indices = np.squeeze(inliers == 1)
        inlier_points = self.points[inlier_indices, :]
        inlier_weights = self.weights[inlier_indices, :]
        return inliers, inlier_points, inlier_weights
    
    def get_inlier_points(self):
        return self.inlier_points
    
    def get_outlier_indices(self):
        inlier_labels = np.round(self.weights)
        outlier_indices = np.squeeze(inlier_labels == 0)
        return outlier_indices