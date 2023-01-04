import os
import numpy as np
import open3d as o3d
import matplotlib as mpl
import matplotlib.pyplot as plt
from plane import PlaneCandidate


class PlaneManager():
    def __init__(self, points):
        self.min_point_num = 300
        self.colormap_name = ['winter', 'Wistia', 'cool'] # TODO: label plane with different colors
        self.cmap_norm = mpl.colors.Normalize(vmin=0.0, vmax=1.0)
        self.reset(points)
        
    def reset(self, points):
        self.points = points
        self.plane_dict = dict()
        self.outlier_points = points
        self.rest_point_num = self.outlier_points.shape[0]
        self.iter_num = 0
        
    def step(self):
        while rest_point_num > self.min_point_num:
            weights = np.ones((self.outlier_points.shape[0], 1))
            plane_obj = PlaneCandidate(self.iter_num, outlier_points, weights)
            for _ in range(15):
                plane_obj.update()
            if np.sum(plane_obj.inliers) > self.min_point_num:
                self.plane_dict[plane_obj.id] = plane_obj
            outlier_points = outlier_points[np.squeeze(plane_obj.inliers == 0), :]
            rest_point_num = outlier_points.shape[0]
            self.iter_num += 1
    
    def render(self):
        # create o3d points
        plane_points_list = list()
        for idx, (_, plane_obj) in enumerate(self.plane_dict.items()):
            inlier_points = plane_obj.get_inlier_points()
            point_colors = plt.get_cmap(self.colormap_name[idx])(self.cmap_norm(np.squeeze(plane_obj.inlier_weights)))[:, 0:3]
            pcd_points = o3d.geometry.PointCloud()
            pcd_points.points = o3d.utility.Vector3dVector(inlier_points)
            pcd_points.colors = o3d.utility.Vector3dVector(point_colors)
            plane_points_list.append(pcd_points)
        # visualize
        vis = o3d.visualization.Visualizer()
        vis.create_window()
        for plane_points in plane_points_list:
            vis.add_geometry(plane_points)
        vis.run()
        vis.destroy_window()
    

if __name__ == "__main__":
    pass 