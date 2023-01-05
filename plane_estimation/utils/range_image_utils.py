import numpy as np


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

def get_normal_map(vertex_map):
    H = vertex_map.shape[0]
    W = vertex_map.shape[1]
    normal_map = np.full((H, W, 3), np.nan, dtype=np.float32)
    for h_ind in range(vertex_map.shape[0]-1):
        for w_ind in range(vertex_map.shape[1]):
            point = vertex_map[h_ind, w_ind, :]
            h_neighbor = vertex_map[h_ind+1, w_ind, :]
            w_neighbor = vertex_map[h_ind, (w_ind+1) % W, :]
            if np.any(np.isnan(point)) or np.any(np.isnan(h_neighbor)) or np.any(np.isnan(w_neighbor)):
                continue
            delta_h = h_neighbor - point
            delta_w = w_neighbor - point
            normal_vector = np.cross(delta_h, delta_w)
            normal_map[h_ind, w_ind] = normal_vector/np.linalg.norm(normal_vector)
    return normal_map