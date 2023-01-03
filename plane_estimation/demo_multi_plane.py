import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import time
import numpy as np
import open3d as o3d
import matplotlib as mpl
import matplotlib.pyplot as plt
from utils.plane_utils import generate_plane_points, rotate_view
from utils.plane import PlaneCandidate


# how to leverage the update of mu

if __name__ == '__main__':
    root_path = os.path.dirname(os.path.abspath(__file__))
    
    # color mapping
    colormap_name = ['Greens', 'Purples']
    cmap_norm = mpl.colors.Normalize(vmin=0.0, vmax=1.0)

    # generate original data
    plane1 = generate_plane_points([1, 2, 3, 3], 300) 
    plane2 = generate_plane_points([1, 1, 1, 1], 1000) 
    plane3 = generate_plane_points([1, 3, 2, 0], 200) 
    points = np.vstack((plane2, plane1, plane3))
    
    weights = np.ones((points.shape[0], 1))
    
    rest_point_num = weights.shape[0]
    plane_dict = dict()
    
    # initialized
    plane_obj = PlaneCandidate(1, points, weights)
    plane_dict[plane_obj.id] = plane_obj
    
    for _ in range(20):
        # reset point labels
        point_labels = np.zeros((points.shape[0], len(plane_dict)))
        # plane parameter estimation
        for idx, (plane_id, plane_obj) in enumerate(plane_dict.items()):
            plane_obj.update()
            inlier_indices = np.squeeze(plane_obj.inliers == 1)
            point_labels[inlier_indices, idx] = plane_obj.id
        
        outlier_indices = np.all(point_labels==0, axis= 1)
        outlier_num = np.sum(outlier_indices)
        if (outlier_num > 100):
            target_points = points[outlier_indices, :]
            target_weights = np.ones((target_points.shape[0], 1))
            new_plane_obj = PlaneCandidate(plane_id+1, target_points, target_weights)
            new_plane_obj.update()
            plane_dict[new_plane_obj.id] = new_plane_obj

# initialization
# point exchange for plane refinement 