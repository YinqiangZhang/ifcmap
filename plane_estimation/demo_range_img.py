import os 
import glob
from copy import deepcopy
import time
import numpy as np 
import open3d as o3d
import matplotlib.pyplot as plt
from utils.range_image_utils import project_to_range_image, get_normal_map
from utils.plane import PlaneCandidate
import matplotlib as mpl

w = 512
h = 64
dpi = 500
cmap_norm = mpl.colors.Normalize(vmin=0.0, vmax=1.0)

def segment_range_img(vertex_map):
    H = vertex_map.shape[0]
    W = vertex_map.shape[1]
    label_image = np.full((H, W), np.nan)
    neighbors = np.array([(-1, 0), (0, -1), (0, 1), (1, 0)])
    segment_label = 0
    non_coords = np.where(np.any(np.isnan(vertex_map), axis=2))
    non_vertexs = set([(x, y) for x, y in zip(non_coords[0], non_coords[1])])
    while True:
        h_seed = np.random.randint(0, H)
        w_seed = np.random.randint(0, W)
        if (h_seed, w_seed) in non_vertexs:
            continue
        openlist = [(h_seed, w_seed)]
        closed_set = set()
        open_set = set()
        open_set.add((h_seed, w_seed))
        
        point_list = list()
        point_indices = list()
        point_list.append(vertex_map[h_seed, w_seed, :])
        point_indices.append((h_seed, w_seed))
        plane_obj = None
        while len(openlist) != 0:
            temp_set = set()
            while len(openlist) != 0:
                center = tuple(openlist.pop())
                open_set.remove(center)
                center_point = vertex_map[center[0], center[1], :]
                for delta in neighbors:
                    h_idx = center[0] + delta[0]
                    w_idx = (center[1] + delta[1]) % W
                    if h_idx >= H or h_idx < 0:
                        continue
                    if ((h_idx, w_idx) not in non_vertexs 
                        and (h_idx, w_idx) not in (open_set | closed_set | temp_set)
                    ):
                        neighbor_point = vertex_map[h_idx, w_idx, :]
                        # if (np.linalg.norm(neighbor_point - center_point) > 1.5):
                        #     continue
                        # if plane_obj is not None and plane_obj.evaluate(np.atleast_2d(neighbor_point)).item() < 0.04:
                        #     continue
                        temp_set.add((h_idx, w_idx))
                        point_list.append(vertex_map[h_idx, w_idx, :])
                        point_indices.append((h_idx, w_idx))
                            
                closed_set.add(center)
            
            if len(point_list) >= 3:
                if plane_obj is None:
                    weights = np.ones((len(point_list), 1))
                    plane_obj = PlaneCandidate(1, np.array(point_list), weights)
                else:
                    plane_obj.update(np.array(point_list))
                weights = plane_obj.evaluate(np.array(point_list))
                valid_indices = np.array(point_indices)[np.squeeze(weights>=0.5),:]
                invalid_indices = np.array(point_indices)[np.squeeze(weights<0.5),:]
                for valid_ind in valid_indices:
                    if tuple(valid_ind) not in (open_set | closed_set):
                        openlist.append(tuple(valid_ind))
                        open_set.add(tuple(valid_ind))
                for invalid_ind in invalid_indices:
                    closed_set.add(tuple(invalid_ind))
            point_list.clear()
            point_indices.clear()
            if len(openlist) == 0:
                print('new plane')
            yield plane_obj
         
                     

root_path = os.path.dirname(os.path.abspath(__file__))
data_path_list = glob.glob(os.path.join(root_path, 'site_cloud_data', '*.ply'))

# test one of them 
data_path = data_path_list[0]

cloud = o3d.io.read_point_cloud(data_path)
range_image, vertex_map = project_to_range_image(cloud, w, h, max_range=75)
# normal_map = get_normal_map(vertex_map)
pcd_noise = deepcopy(cloud)
cloud.paint_uniform_color(np.array([107, 142, 35])/255)
vis = o3d.visualization.Visualizer()
vis.create_window()
vis.add_geometry(pcd_noise)
# vis.add_geometry(cloud) 
for idx, plane_obj in enumerate(segment_range_img(vertex_map)):
    if plane_obj is None:
        continue
    point_colors = plt.get_cmap('RdBu')(cmap_norm(np.squeeze(plane_obj.weights)))[:, 0:3]
    pcd_noise.points = o3d.utility.Vector3dVector(plane_obj.points)
    pcd_noise.colors = o3d.utility.Vector3dVector(point_colors)
    vis.update_geometry(pcd_noise)
    vis.poll_events()
    vis.update_renderer()
    time.sleep(0.00005)
vis.run()
vis.destroy_window()
# filtered_cloud_points = vertex_map.reshape([-1, 3])
# pcd_points = o3d.geometry.PointCloud()
# point_colors = plt.get_cmap('magma')(cmap_norm(np.squeeze(range_image.reshape([-1, 1]))))[:, 0:3]
# pcd_points.points = o3d.utility.Vector3dVector(filtered_cloud_points)
# pcd_points.colors = o3d.utility.Vector3dVector(point_colors)
# pcd_points.normals = o3d.utility.Vector3dVector(normal_map.reshape(-1, 3))

# vis = o3d.visualization.Visualizer()
# vis.create_window()
# vis.add_geometry(pcd_points)
# vis.run()
# vis.destroy_window()

# fig = plt.figure(frameon=False, figsize=(w / dpi, h / dpi), dpi=dpi)
# ax = plt.Axes(fig, [0.0, 0.0, 1.0, 1.0])
# ax.set_axis_off()
# fig.add_axes(ax)

# # Then draw your image on it :
# ax.imshow(range_image, aspect="auto", cmap="magma")
# plt.show()
# fig.savefig(os.path.join(root_path, "range_image.png"), dpi=dpi)