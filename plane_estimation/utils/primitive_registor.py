import copy 
import scipy
import trimesh
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt 
from trimesh.proximity import ProximityQuery

class PrimitiveRegistor():
    def __init__(self, model_meshes, primitives, correspondences):
        self.model_list = model_meshes
        self.primitive_list = primitives
        self.correspondence_list = correspondences
        
        self.history_states = list()
        self.history_V = list()
        
        self.k = 20
        self.damping = 1.0
        self.s_damping = 0.1
        self.time_step = 0.3
        self.iteration_num = 1000
        self.sample_num = 20
        
        self.state = np.zeros((13,))
        self.centroid = None
        self.ref_points = None
        self.ref_points_list = None
        self.J = None
        
        self.dws_primitives = self.primitive_downsample()
        self.mesh_query_list = self.generate_mesh_query()
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
        self.state[7:10] = 0.001*np.random.randn(3)
        self.state[10:] = 0.001*np.random.randn(3)
    
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
    
    def primitive_downsample(self):
        dws_primitive_list = list()
        for tri_pcd in self.primitive_list:
            # ratio = 0.001 if len(pcd.points) > 500 else 2/len(pcd.points)
            # o3d_pcd = o3d.geometry.PointCloud()
            # o3d_pcd.points = o3d.utility.Vector3dVector(tri_pcd.vertices)
            # points = np.asarray(o3d_pcd.farthest_point_down_sample(self.sample_num).points)
            points = np.asarray(tri_pcd.vertices[np.random.choice(np.arange(len(tri_pcd.vertices)), self.sample_num, replace=False)])
            dws_primitive_list.append(points)
        return dws_primitive_list
    
    def generate_mesh_query(self):
        mesh_query_list = list()
        for mesh in self.model_list:
            mesh_query_list.append(ProximityQuery(mesh))
        return mesh_query_list
    
    def optimize(self):
        self.history_state = self.state
        self.history_states = list()
        self.history_V = list()
        self.state[7:10] = 0.0
        self.state[10:] = 0.0
        for n_iter in range(self.iteration_num):
            s_dot, spring_energy = self.dynamics()
            _, _, V_total = self.compute_system_energy(spring_energy)
            self.state = self.state + self.time_step * s_dot
            self.state[3:7] = self.state[3:7] / np.linalg.norm(self.state[3:7])
            self.history_states.append(copy.deepcopy(self.state))
            self.history_V.append(V_total)
            if np.linalg.norm(s_dot) < 0.01:
                break
        # print(n_iter)

        result_trans = np.identity(4)
        rot_mat = R.from_quat(self.state[3:7]).as_matrix()
        result_trans[:-1, :-1] = rot_mat
        result_trans[:-1, -1] = self.state[:3] - np.squeeze(rot_mat @ np.atleast_2d(self.centroid).T)
        return result_trans, V_total
    
    def dynamics(self):
        # extend different variables
        xbar = self.state[:3]
        q = self.state[3:7]
        vbar = self.state[7:10]
        omega = self.state[10:]
        rot = R.from_quat(q).as_matrix()
        curr_points = np.matmul(rot, self.ref_points.T).T + xbar
        
        # generate vector forces
        composite_force_dict = dict()
        for primitive_idx, model_idx in self.correspondence_list:
            mesh_query = self.mesh_query_list[model_idx]
            primitive_points = np.matmul(rot, self.ref_points_list[primitive_idx].T).T + xbar
            _, forces = self.get_hausdorff_projective_points(mesh_query, primitive_points)
            if composite_force_dict.get(primitive_idx, None) is None:
                composite_force_dict[primitive_idx] = [forces]
            else:
                composite_force_dict[primitive_idx].append(forces)
        
        # force composition
        force_vectors = list()
        spring_energy = list()
        for idx in range(len(self.dws_primitives)):
            sum_forces = np.zeros_like(self.dws_primitives[idx])
            sum_energy = np.zeros((sum_forces.shape[0], 1))
            if composite_force_dict.get(idx, None) is not None:
                force_list = composite_force_dict[idx]
                sum_forces = np.stack(force_list, axis=2).sum(axis=2)
                sum_energy = np.atleast_2d(np.square(np.stack(force_list, axis=2)).sum(axis=(1, 2))).T
            force_vectors.append(sum_forces)
            spring_energy.append(sum_energy)
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
        
        s_dot = np.zeros_like(self.state)
        s_dot[:3] = vbar
        s_dot[3:7] = np.squeeze(q_dot)
        s_dot[7:10] = acc_bar / np.linalg.norm(acc_bar) * min(np.linalg.norm(acc_bar), 1000)
        s_dot[10:] = np.squeeze(domega) / np.linalg.norm(domega) * min(np.linalg.norm(domega), 10)
        return s_dot, spring_energy
    
    def compute_system_energy(self, spring_energy):
        vbar = np.atleast_2d(self.state[7:10])
        omega = np.atleast_2d(self.state[10:])
        Vk = spring_energy.shape[0]/2 * np.dot(vbar, vbar.T) + 0.5 * (omega @ self.J @ omega.T)
        Vp = spring_energy.sum() / (2 * self.k)
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
        force_vectors = projected_points - points
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
        