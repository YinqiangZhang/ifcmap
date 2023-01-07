import os
import numpy as np
import open3d as o3d
import matplotlib as mpl
import matplotlib.pyplot as plt
from utils.plane import PlaneCandidate
from utils.range_image_utils import project_to_range_image


class PlaneManager():
    def __init__(self, points):
        self.min_point_num = 600
        self.colormap_name = ['winter', 'Wistia', 'cool'] # TODO: label plane with different colors
        self.cmap_norm = mpl.colors.Normalize(vmin=0.0, vmax=1.0)
        # range_image, vertex_map = project_to_range_image(cloud, w, h, max_range=75)
        self.reset(points)
        
    def reset(self, points):
        self.points = points
        self.plane_dict = dict()
        self.plane_point_indices = dict()
        self.multi_weights = None
        self.labels = np.full((self.points.shape[0], 1), np.nan)
        self.weights = np.ones((self.points.shape[0], 1))
        self.iter_num = 0
        
    def step(self):
        print("Try to segment plane {}".format(self.iter_num))
        self.plane_dict.clear()
        self.plane_point_indices.clear()
        
        # expectation
        for label_type in np.unique(self.labels):
            if np.isnan(label_type):
                indices = np.where(np.isnan(self.labels))[0]
            else:
                indices = np.where(self.labels==label_type)[0]
            points = self.points[indices, :]
            weights = np.ones((points.shape[0], 1))
            
            plane_id = self.iter_num if np.isnan(label_type) else label_type
            plane_obj = PlaneCandidate(plane_id, points, weights)
            plane_obj.update()
            valid_indices = indices[np.where(plane_obj.inliers == 1)[0]]
            self.plane_dict[plane_obj.id] = plane_obj
            self.plane_point_indices[plane_obj.id] = valid_indices
                
        # maximization
        labels = np.full_like(self.labels, np.nan)
        for plane_id, indices in self.plane_point_indices.items():
            labels[indices] = plane_id
        self.labels = labels
        self.iter_num += 1
    
    def render(self, vis):
        # create o3d points
        plane_points_list = list()
        for idx, (_, plane_obj) in enumerate(self.plane_dict.items()):
            inlier_points, inlier_weights = plane_obj.get_inliers()
            point_colors = plt.get_cmap(self.colormap_name[idx%3])(self.cmap_norm(np.squeeze(inlier_weights)))[:, 0:3]
            pcd_points = o3d.geometry.PointCloud()
            pcd_points.points = o3d.utility.Vector3dVector(inlier_points)
            pcd_points.colors = o3d.utility.Vector3dVector(point_colors)
            plane_points_list.append(pcd_points)
        # visualize
        for plane_points in plane_points_list:
            vis.add_geometry(plane_points)
    

if __name__ == "__main__":
    pass 