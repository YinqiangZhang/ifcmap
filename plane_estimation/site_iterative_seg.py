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
    
    cloud = o3d.io.read_point_cloud(data_path)
    cloud.paint_uniform_color(np.array([211, 211, 211])/255.0)
    points = np.asarray(cloud.points)
    
    plane_manager = PlaneManager(points)
    for _ in range(10):
        plane_manager.step()
        
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    plane_manager.render(vis)
    vis.add_geometry(cloud)
    vis.run()
    vis.destroy_window()