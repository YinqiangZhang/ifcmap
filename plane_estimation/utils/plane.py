import numpy as np

class PlaneCandidate():
    def __init__(self, plane_id, points, weights, mu0=10):
        self.id = int(plane_id)
        self.points = points
        self.homo_points = self.generate_homo_points(points)
        self.weights = weights
        self.point_cluster = self.generate_point_cluster(self.points, self.weights)
        self.plane_params = None
        
        self.inliers = np.ones_like(weights)
        
        self.mu0 = mu0
        self.mu = mu0
        self.recover_factor = 1.4
        self.mu_min = 0.3
        
        # self.update()
        
    def generate_homo_points(self, points):
        return np.column_stack((points, np.ones((points.shape[0], 1))))
    
    def update(self, new_points=None, direct_upate=False):
        if not direct_upate:
            self.plane_params, min_eigenvalue = self.plane_estimate(new_points, weight_update=False)
            if min_eigenvalue < self.points.shape[0] * 0.00001:
                return
        
        pre_cost = np.inf
        self.mu = self.mu0
        for out_idx in range(10):
            in_idx = 0
            while True:
                self.plane_params, cost = self.plane_estimate(weight_update=True)
                self.weights = self.GM_weight_estimate()
                self.inliers = self.inliner_estimate()                
                print('Outer loop: {}, inner loop: {}, cost: {}, mu: {}'.format(out_idx, in_idx, cost, self.mu))
                if not np.isinf(pre_cost) and abs(cost - pre_cost)/cost < 0.01:
                    break
                in_idx += 1
                pre_cost = cost
                
            self.mu = self.mu_update()
        
    def plane_estimate(self, new_points=None, weight_update=False):
        new_point_cluster = np.zeros((4, 4))
        if new_points is not None:
            new_weights = self.evaluate(new_points)
            self.points = np.vstack((self.points, new_points))
            self.weights = np.vstack((self.weights, new_weights))
            new_point_cluster = self.generate_point_cluster(new_points, new_weights)
        if not weight_update:
            self.point_cluster += new_point_cluster
        else:
            self.point_cluster = self.generate_point_cluster(self.points, self.weights)
        eig_values, eig_vectors = np.linalg.eig(self.point_cluster)
        parameters = eig_vectors[:, np.argmin(eig_values)]
        parameters = np.atleast_2d(parameters / np.linalg.norm(parameters[:-1]))
        return parameters, np.min(eig_values)

    def generate_point_cluster(self, points, weights):
        homo_points = self.generate_homo_points(points)
        weighted_homo_points = np.multiply(homo_points, weights)
        point_cluster = np.matmul(weighted_homo_points.T, weighted_homo_points)
        return point_cluster
    
    def GM_weight_estimate(self):
        homo_points = self.generate_homo_points(self.points)
        squared_errors = np.square(np.dot(self.plane_params, homo_points.T))
        weights = np.square(self.mu / (self.mu + squared_errors)).T
        return weights
    
    def TLS_weight_update(self):
        errors = np.dot(self.plane_params, self.homo_points.T)
    
    def mu_update(self):
        return np.max([self.mu_min, self.mu / self.recover_factor])
    
    def inliner_estimate(self, threshold=0.8):
        inliers = np.zeros_like(self.weights)
        inliers[self.weights>=threshold] = 1
        return inliers
    
    def get_inliers(self):
        inlier_points = self.points[np.squeeze(self.inliers == 1), :]
        inlier_weights = self.weights[np.squeeze(self.inliers == 1), :]
        return inlier_points, inlier_weights
    
    def evaluate(self, points):
        weights = np.ones((points.shape[0], 1))
        if self.plane_params is not None:
            homo_points = self.generate_homo_points(points)
            squared_errors = np.square(np.dot(self.plane_params, homo_points.T))
            weights = np.square(self.mu / (self.mu + squared_errors)).T
        return weights
         