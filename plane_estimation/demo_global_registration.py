import os 
import glob
import copy
import numpy as np
import open3d as o3d 


def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp])

def preprocess_point_cloud(pcd, voxel_size):
    print(":: Downsample with a voxel size %.3f." % voxel_size)
    pcd_down = pcd.voxel_down_sample(voxel_size)

    radius_normal = voxel_size * 2
    print(":: Estimate normal with search radius %.3f." % radius_normal)
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    radius_feature = voxel_size * 5
    print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd_down, pcd_fpfh

def execute_global_registration(source_down, target_down, source_fpfh,
                                target_fpfh, voxel_size):
    distance_threshold = voxel_size * 1.5
    print(":: RANSAC registration on downsampled point clouds.")
    print("   Since the downsampling voxel size is %.3f," % voxel_size)
    print("   we use a liberal distance threshold %.3f." % distance_threshold)
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, True, distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        4, [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(
                0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                distance_threshold)
        ], o3d.pipelines.registration.RANSACConvergenceCriteria(4000000, 500))
    return result

def refine_registration(source, target, trans_init, voxel_size):
    distance_threshold = voxel_size
    print(":: Point-to-plane ICP registration is applied on original point")
    print("   clouds to refine the alignment. This time we use a strict")
    print("   distance threshold %.3f." % distance_threshold)
    result = o3d.pipelines.registration.registration_icp(
        source, target, distance_threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPlane())
    return result

root_path = os.path.dirname(os.path.abspath(__file__))    
data_path_list = glob.glob(os.path.join(root_path, 'site_cloud_data', '*.ply'))

# read all clouds
cloud_list = list()
for idx, data_path in enumerate(data_path_list):
    cloud = o3d.io.read_point_cloud(data_path)
    cloud_list.append(cloud)
    if idx == 100:
        break

# coarse registration
cloud1, cloud2= cloud_list[0], cloud_list[50]
voxel_size = 0.8
cloud_down1, cloud_fpfh1 = preprocess_point_cloud(cloud1, voxel_size)
cloud_down2, cloud_fpfh2 = preprocess_point_cloud(cloud2, voxel_size)
result_ransac = execute_global_registration(cloud_down1, cloud_down2, cloud_fpfh1, cloud_fpfh2, voxel_size)
print(result_ransac)

# fine_registration
voxel_size = 0.2
cloud_down1, cloud_fpfh1 = preprocess_point_cloud(cloud1, voxel_size)
cloud_down2, cloud_fpfh2 = preprocess_point_cloud(cloud2, voxel_size)
result_icp = refine_registration(cloud_down1, cloud_down2, result_ransac.transformation, voxel_size)

draw_registration_result(cloud_down1, cloud_down2, result_icp.transformation)