import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import glob
import open3d as o3d
import numpy as np 
from utils.plane_manager import PlaneManager

# Lesson: we need to first get a initial result
if __name__ == '__main__':
    root_path = os.path.dirname(os.path.abspath(__file__))
    
    data_path_list = glob.glob(os.path.join(root_path, 'site_cloud_data', '*.ply'))
    data_path = data_path_list[0]
    site_pcd_cloud = np.asarray(o3d.io.read_point_cloud(data_path).points)
    
    plane_manager = PlaneManager(site_pcd_cloud)
    
    for _ in range(2):
        plane_manager.step()
    plane_manager.render()