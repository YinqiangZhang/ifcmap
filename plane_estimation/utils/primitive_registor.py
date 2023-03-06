import time
import copy
import numpy as np
import open3d as o3d
from pysdf import SDF
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt 
from trimesh.proximity import ProximityQuery


class PrimitiveRegistor():
    def __init__(self, model_meshes, primitives, correspondences, s_dot_threshold = 0.001):
        self.model_list = model_meshes
        self.primitive_list = primitives
        self.correspondence_list = correspondences
        
        self.history_states = list()
        self.history_V = list()
        self.point_pairs = list()
        self.record_pair_lineset = True
        
        self.k = 20
        self.damping_c = 1.0
        self.s_damping_c = 0.1
        self.damping = self.damping_c
        self.s_damping = self.s_damping_c
        self.time_step = 0.3
        self.iteration_num = 600
        self.sample_num = 20
        self.s_dot_threshold = s_dot_threshold
        
        self.state = np.zeros((13,))
        self.centroid = None
        self.ref_points = None
        self.ref_points_list = None
        self.J = None
        
        self.mesh_sdf_list = self.generate_sdf_list()
        self.dws_primitives = self.primitive_downsample()
        self.mesh_query_list = self.generate_mesh_query()
        self.o3d_mesh_list = self.get_o3d_meshes()
        self.initial_state()
        
    def initial_state(self):
        target_points = np.vstack(self.dws_primitives)
        self.centroid = target_points.mean(axis=0)
        self.ref_points = target_points - self.centroid
        self.ref_points_list = list(points - self.centroid for points in self.dws_primitives)
        self.J = self.get_moment_inertia(self.ref_points)
        self.J_inv = np.linalg.inv(self.J)

        self.state[:3] = self.centroid
        self.state[3:7] = np.array([0, 0, 0, 1.0])
        self.state[7:10] = 0.0
        self.state[10:] = 0.0
    
    def get_moment_inertia(self, ref_points):
        curr_J = np.zeros((3, 3))
        for ref_point in ref_points:
            temp = self.hatmap(ref_point)
            curr_J -= np.matmul(temp, temp) 
        return curr_J
    
    def hatmap(self, u):
        hat_u = np.array([[0, -u[2], u[1]],
                          [u[2], 0, -u[0]],
                          [-u[1], u[0], 0],]
                         )
        return hat_u
    
    def quat_prod(self, a, b):
        A = np.array([[a[3], -a[2], a[1], a[0]],
                      [a[2], a[3], -a[0], a[1]],
                      [-a[1], a[0], a[3], a[2]],
                      [-a[0], -a[1], -a[2], a[3]]]
                     )
        return np.matmul(A, np.atleast_2d(b).T)
    
    def generate_sdf_list(self):
        mesh_sdf_list = list()
        for model in self.model_list:
            model_sdf = SDF(model.vertices, model.faces)
            mesh_sdf_list.append(model_sdf)
        return mesh_sdf_list
    
    def set_correspondence(self, correspondence_list):
        self.correspondence_list = correspondence_list
        
    def set_damping(self):
        target_indices = set()
        for correspondence_pair in self.correspondence_list:
            target_idx, _ = correspondence_pair
            target_indices.add(target_idx)
        point_mass = 0
        for idx in target_indices:
            point_mass += self.dws_primitives[idx].shape[0]
        self.damping = 1.0 * np.sqrt(point_mass/25)
        self.s_damping = 0.1 * np.sqrt(point_mass/25)
        # TODO: should be improved later for much faster
        # self.iteration_num = max(100, int(600 * np.sqrt(25/point_mass)))
        
    def add_correspondence(self, correspondence):
        self.correspondence_list.append(correspondence)
    
    def remove_correspondence(self):
        self.state = self.history_state
        self.correspondence_list.pop(-1)
    
    def set_model_primitives(self, model_meshes, primitives):
        self.model_list = model_meshes
        self.primitive_list = primitives
        self.mesh_query_list = self.generate_mesh_query()
        self.dws_primitives = self.primitive_downsample()
        self.o3d_mesh_list = self.get_o3d_meshes()
    
    def primitive_downsample(self):
        dws_primitive_list = list()
        for tri_pcd in self.primitive_list:
            o3d_pcd = o3d.geometry.PointCloud()
            o3d_pcd.points = o3d.utility.Vector3dVector(tri_pcd.vertices)
            sample_num = max(int(len(o3d_pcd.points) / 100000 * self.sample_num), self.sample_num)
            points = np.asarray(o3d_pcd.farthest_point_down_sample(sample_num).points)
            dws_primitive_list.append(points)
        return dws_primitive_list
    
    def generate_mesh_query(self):
        mesh_query_list = list()
        for mesh in self.model_list:
            mesh_query_list.append(ProximityQuery(mesh))
        return mesh_query_list
    
    def get_o3d_meshes(self):
        o3d_mesh_list = list()
        for mesh in self.model_list:
            o3d_mesh = mesh.as_open3d
            o3d_mesh.compute_vertex_normals()
            o3d_mesh_list.append(o3d_mesh)
        return o3d_mesh_list
    
    def reset_to_still(self):
        self.state[7:10] = 0.0
        self.state[10:] = 0.0
    
    def optimize(self, total_iter_num=None):
        is_done = False
        total_iter_num = self.iteration_num if total_iter_num is None else total_iter_num
        self.history_state = self.state
        # self.history_states = list()
        # self.history_V = list()
        for n_iter in range(total_iter_num):
            # TODO: the dynamics part require about 20ms
            s_dot, spring_energy = self.dynamics()
            _, _, V_total = self.compute_system_energy(spring_energy)
            self.state = self.state + self.time_step * s_dot
            self.state[3:7] = self.state[3:7] / np.linalg.norm(self.state[3:7])
            self.history_states.append(copy.deepcopy(self.state))
            self.history_V.append(V_total)
            if np.linalg.norm(s_dot) < self.s_dot_threshold:
                is_done = True
                break
        print(n_iter)
        self.reset_to_still()
        return self.get_transformation_matrix(), V_total, is_done
    
    def get_transformation_matrix(self):
        result_trans = np.identity(4)
        rot_mat = R.from_quat(self.state[3:7]).as_matrix()
        result_trans[:-1, :-1] = rot_mat
        result_trans[:-1, -1] = self.state[:3] - np.squeeze(rot_mat @ np.atleast_2d(self.centroid).T)
        return result_trans
    
    def dynamics(self):
        # extend different variables
        xbar = self.state[:3]
        q = self.state[3:7]
        vbar = self.state[7:10]
        omega = self.state[10:]
        rot = R.from_quat(q).as_matrix()
        curr_points = np.matmul(rot, self.ref_points.T).T + xbar
        # start_time = time.time() 
        # generate vector forces
        composite_force_dict = dict()
        self.point_pairs = list()
        for primitive_idx, model_idx in self.correspondence_list:
            primitive_points = np.matmul(rot, self.ref_points_list[primitive_idx].T).T + xbar
            # mesh_query = self.mesh_query_list[model_idx]
            # projected_points, forces = self.get_hausdorff_projective_points(mesh_query, primitive_points)
            mesh_sdf = self.mesh_sdf_list[model_idx]
            projected_points, forces = self.get_sdf_distance_gradient(mesh_sdf, primitive_points)
            if self.record_pair_lineset:
                self.point_pairs.append((projected_points, primitive_points))
            if composite_force_dict.get(primitive_idx, None) is None:
                composite_force_dict[primitive_idx] = [forces]
            else:
                composite_force_dict[primitive_idx].append(forces)
        # print('Force time: {}'.format(time.time()-start_time))
        # force composition
        force_vectors = list()
        spring_energy = list()
        for idx in range(len(self.dws_primitives)):
            sum_forces = np.zeros_like(self.dws_primitives[idx])
            if composite_force_dict.get(idx, None) is not None:
                force_list = composite_force_dict[idx]
                sum_forces = np.stack(force_list, axis=2).sum(axis=2)
                sum_energy = np.atleast_2d(np.square(np.stack(force_list, axis=2)).sum(axis=(1, 2))).T
                spring_energy.append(sum_energy)
            force_vectors.append(sum_forces)
        force_vectors = np.vstack(force_vectors)
        spring_energy = np.vstack(spring_energy)
        # dynamic equations
        f_spring = self.k * force_vectors
        vel = (rot @ (self.hatmap(omega) @ self.ref_points.T)).T + vbar
        vel_norm = np.linalg.norm(vel, axis=1)
        f_damping = -self.damping * vel - self.s_damping * (vel.T * vel_norm).T
        f_total = f_spring + f_damping
        acc_bar = f_total.mean(axis=0)
        q_dot = 0.5 * self.quat_prod(q, np.append(omega, 0))
        f_total_X = (rot.T @ f_total.T).T
        tau_X = np.cross((rot.T @ (curr_points - xbar).T).T, f_total_X)
        tau = np.atleast_2d(tau_X.sum(axis=0)).T
        domega = self.J_inv @ (tau - self.hatmap(omega) @ self.J @ np.atleast_2d(omega).T)
        # print('Dynamics time: {}'.format(time.time()-start_time))
        s_dot = np.zeros_like(self.state)
        s_dot[:3] = vbar
        s_dot[3:7] = np.squeeze(q_dot)
        s_dot[7:10] = acc_bar / np.linalg.norm(acc_bar) * min(np.linalg.norm(acc_bar), 1000)
        s_dot[10:] = np.squeeze(domega) / np.linalg.norm(domega) * min(np.linalg.norm(domega), 10)
        # print('Iteration time: {}'.format(time.time()-start_time))
        return s_dot, spring_energy
    
    def compute_system_energy(self, spring_energy):
        vbar = np.atleast_2d(self.state[7:10])
        omega = np.atleast_2d(self.state[10:])
        Vk = 0.5 * self.ref_points.shape[0] * np.dot(vbar, vbar.T) + 0.5 * (omega @ self.J @ omega.T)
        # Vp = 0.5 * spring_energy.sum() / self.k
        Vp = 0.5 * np.percentile(spring_energy, 75)/self.k * spring_energy.shape[0]
        V_total = Vk + Vp
        return Vp.item(), Vk.item(), V_total.item()
    
    def get_projective_points(self, plane_params, primitive_points):
        signed_distances = self.get_signed_distance(plane_params, primitive_points)
        target_plane_idx = np.argmin(np.abs(signed_distances).mean(axis=0))
        alpha = -signed_distances[:, target_plane_idx]
        target_plane_params = np.atleast_2d(plane_params[target_plane_idx, :-1])
        force_vectors = np.multiply(alpha, np.repeat(target_plane_params, primitive_points.shape[0], axis=0).T).T
        projected_points = primitive_points + force_vectors
        return projected_points, force_vectors
    
    def get_hausdorff_projective_points(self, mesh_query, points):
        projected_points, _, _ = mesh_query.on_surface(points)
        signed_distances = mesh_query.signed_distance(points)
        force_vectors = projected_points - points
        force_vectors[signed_distances>0] = np.zeros_like(force_vectors[signed_distances>0])
        return projected_points, force_vectors
    
    def get_sdf_distance_gradient(self, mesh_sdf, points):
        delta = 0.0001
        sd_0 = mesh_sdf(points)
        sd_x_shift = mesh_sdf(points + [delta, 0.0, 0.0])
        sd_y_shift = mesh_sdf(points + [0.0, delta, 0.0])
        sd_z_shift = mesh_sdf(points + [0.0, 0.0, delta])
        gradients = np.array([(sd_x_shift-sd_0), sd_y_shift-sd_0, sd_z_shift-sd_0])/(2*delta)
        force_vectors = -(gradients/np.linalg.norm(gradients, axis=0) * sd_0).T
        force_vectors[sd_0>0] = np.zeros_like(force_vectors[sd_0>0])
        projected_points = points + force_vectors
        return projected_points, force_vectors
    
    def get_signed_distance(self, plane_params, primitive_points):
        homo_points = np.column_stack((primitive_points, np.ones((primitive_points.shape[0], 1))))
        return homo_points @ plane_params.T
    
    def get_average_potential(self):
        target_indices = set()
        for correspondence_pair in self.correspondence_list:
            target_idx, _ = correspondence_pair
            target_indices.add(target_idx)
        
        point_mass = 0
        for idx in target_indices:
            point_mass += self.dws_primitives[idx].shape[0]
        
        return self.history_V[-1] / point_mass
    
    def visualize(self):
        # TODO: visualize the optimization procedures
        state_matrix = np.array(self.history_states)
        xbar = state_matrix[:, :3]
        q = state_matrix[:, 3:7]
        vbar = state_matrix[:, 7:10]
        omega = state_matrix[:, 10:]
        fig = plt.figure(figsize=(16, 6))
        ax1, ax2 = fig.subplots(1, 2)
        ax1.plot(range(vbar.shape[0]), np.linalg.norm(vbar, axis=1), 'b', linewidth=2)
        ax1.grid('-.')
        ax2.plot(range(omega.shape[0]), np.linalg.norm(omega, axis=1), 'r', linewidth=2)
        ax2.grid('-.')
        plt.show()
        plt.close()
        
    def get_inlier_lineset(self):
        key_points_set = o3d.geometry.PointCloud()
        ancher_points_set = o3d.geometry.PointCloud()
        line_set = o3d.geometry.LineSet()
        for idx, _ in enumerate(self.correspondence_list):
            key_points = o3d.geometry.PointCloud()
            ancher_points = o3d.geometry.PointCloud()
            point_pairs = self.point_pairs[idx]
            correspndence_set = list((num, num) for num in range(point_pairs[0].shape[0]))
            ancher_points.points = o3d.utility.Vector3dVector(point_pairs[0])
            key_points.points = o3d.utility.Vector3dVector(point_pairs[1])
            corres_lines = o3d.geometry.LineSet.create_from_point_cloud_correspondences(ancher_points, 
                                                                                        key_points, 
                                                                                        correspndence_set)
            line_set += corres_lines
            key_points_set += key_points
            ancher_points_set += ancher_points
        ancher_points_set.paint_uniform_color(np.array([255, 48, 48])/255)
        key_points_set.paint_uniform_color(np.array([138, 43, 226])/255)
        line_set.paint_uniform_color(np.array([255, 0, 0])/255)
        return [line_set, ancher_points_set, key_points_set]