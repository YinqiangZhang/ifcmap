import numpy as np
import open3d as o3d

def generate_plane_points(paramters, point_num, range=[-2.0, 2.0]):
    normal_vector = np.array(paramters[:-1]) 
    normal_vector = normal_vector / np.linalg.norm(normal_vector)
    distance = paramters[-1]
    xy_points = np.random.uniform(range[0], range[1], size=(point_num, 2))
    xyz_points = list()
    for xy_point in xy_points:
        error = np.random.normal(0, 0.05)
        z = (error - distance - np.dot(xy_point, normal_vector[:-1]))/normal_vector[-1]
        xyz_points.append([xy_point[0], xy_point[1], z])
    
    return np.array(xyz_points)

def generate_plane_o3d_points(paramters, point_num):
    points = generate_plane_points(paramters, point_num)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    return pcd

def plain_parameter_estimate(points):
    homo_points = np.column_stack((points, np.ones((points.shape[0], 1))))
    point_cluster = np.matmul(homo_points.T, homo_points)
    eig_values, eig_vectors = np.linalg.eig(point_cluster)
    plane_cost = np.min(eig_values)
    estimated_parameters = eig_vectors[:, np.argmin(eig_values)]
    temp_norm = np.linalg.norm(estimated_parameters[:-1])
    estimated_parameters = estimated_parameters / temp_norm 
    return plane_cost, estimated_parameters

def rotate_view(vis):
    ctr = vis.get_view_control()
    ctr.rotate(2.0, 0.0)
    return False

def project_to_range_image(cloud, W=1024, H=128, max_range=50):
    """Project a pointcloud into a spherical projection image.projection.
    Function takes no arguments because it can be also called externally if the
    value of the constructor was not set (in case you change your mind about
    wanting the projection)
    """
    current_vertex = np.asarray(cloud.points)

    # ouster
    fov_up = 45 / 180.0 * np.pi  # field of view up in radians
    fov_down = -45 / 180.0 * np.pi  # field of view down in radians
    fov = abs(fov_down) + abs(fov_up)  # get field of view total in radians

    # get depth of all points
    depth = np.linalg.norm(current_vertex, axis=1)
    current_vertex = current_vertex[(depth > 0) & (depth < max_range)]
    depth = depth[(depth > 0) & (depth < max_range)]

    # get scan components
    scan_x = current_vertex[:, 0]
    scan_y = current_vertex[:, 1]
    scan_z = current_vertex[:, 2]

    # get angles of all points
    yaw = -np.arctan2(scan_y, scan_x)
    pitch = np.arcsin(scan_z / depth)

    # get projections in image coords
    proj_x = 0.5 * (yaw / np.pi + 1.0)  # in [0.0, 1.0]
    proj_y = 1.0 - (pitch + abs(fov_down)) / fov  # in [0.0, 1.0]

    # scale to image size using angular resolution
    proj_x *= W  # in [0.0, W]
    proj_y *= H  # in [0.0, H]

    # round and clamp for use as index
    proj_x = np.floor(proj_x)
    proj_x = np.minimum(W - 1, proj_x)
    proj_x = np.maximum(0, proj_x).astype(np.int32)

    proj_y = np.floor(proj_y)
    proj_y = np.minimum(H - 1, proj_y)
    proj_y = np.maximum(0, proj_y).astype(np.int32)

    # order in decreasing depth
    order = np.argsort(depth)[::-1]
    depth = depth[order]
    proj_y = proj_y[order]
    proj_x = proj_x[order]

    scan_x = scan_x[order]
    scan_y = scan_y[order]
    scan_z = scan_z[order]

    proj_range = np.full((H, W), np.nan, dtype=np.float32)
    proj_vertex = np.full((H, W, 3), np.nan, dtype=np.float32)

    proj_range[proj_y, proj_x] = depth
    proj_vertex[proj_y, proj_x] = np.array([scan_x, scan_y, scan_z]).T

    return proj_range, proj_vertex

if __name__ == '__main__': 
    pass
    # pcd1 = o3d.geometry.PointCloud()
    # pcd1.points = o3d.utility.Vector3dVector(plane1)
    # pcd1.paint_uniform_color([1, 0.706, 0])
    # pcd2 = o3d.geometry.PointCloud()
    # pcd2.points = o3d.utility.Vector3dVector(plane2)
    # pcd2.paint_uniform_color([0, 0, 1.0])