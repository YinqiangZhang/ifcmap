import os 
import glob
import argparse
import open3d as o3d
import open3d.visualization as vis
from utils.registration_utils import get_model_meshes, get_real_points

def save_view_point(pcd, filename):
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pcd)
    vis.run()  # user changes the view and press "q" to terminate
    param = vis.get_view_control().convert_to_pinhole_camera_parameters()
    o3d.io.write_pinhole_camera_parameters(filename, param)
    vis.destroy_window()
    
def set_material():
    mat_bim = vis.rendering.MaterialRecord()
    # mat_box.shader = 'defaultLitTransparency'
    mat_bim.shader = 'defaultLitSSR'
    mat_bim.base_color = [0.1, 0.1, 0.2, 1.0]
    mat_bim.base_roughness = 0.0
    mat_bim.base_reflectance = 0.0
    mat_bim.base_clearcoat = 1.0
    mat_bim.thickness = 1.0
    mat_bim.transmission = 0.4
    mat_bim.absorption_distance = 1
    mat_bim.absorption_color = [0.1, 0.1, 0.2]
    return mat_bim

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Select target model:')
    parser.add_argument('--index', type=int, default=0, help='Model Index')
    args = parser.parse_args()
    
    # read segmented planes
    root_path = os.path.dirname(os.path.abspath(__file__))
    data_folder = os.path.join(root_path, 'CaseStudy')
    model_folder = os.path.join(data_folder, 'CaseStudyModels')
    segment_folder = os.path.join(data_folder, 'CaseStudySegments')
    
    model_paths = glob.glob(os.path.join(model_folder, '*.ply'))
    segments_paths = [glob.glob(os.path.join(segment_folder, folder, '*.ply')) \
                      for folder in os.listdir(segment_folder)]
    
    case_index = args.index
    model_meshes, o3d_mesh = get_model_meshes(model_paths[case_index:case_index+1])
    real_points, o3d_real_points, scene_pcd = get_real_points(segments_paths[case_index:case_index+1])
    mat_bim = set_material()
    
    geoms = [{'name': 'bim_model', 'geometry': o3d_mesh, 'material': mat_bim}, 
             {'name': 'scene_pcd', 'geometry': scene_pcd}]
    vis.draw(geoms,
             bg_color=(1.0, 1.0, 1.0, 1.0),
             show_ui=True,
             width=1920,
             height=1080)
    # o3d.visualization.draw_geometries([o3d_mesh, scene_pcd])