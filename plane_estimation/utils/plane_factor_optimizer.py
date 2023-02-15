import copy 
import numpy as np


class PlaneFactorOptimizer():
    def __init__(self):
        self.factor_list = list()
        self.init_T = np.identity(4)
        
    def set_init_T(self, init_T):
        self.init_T = init_T

    def add_factor(self, plane_params, Q_matrix, point_num):
        self.factor_list.append([plane_params, Q_matrix, point_num])
    
    def optimize(self):
        self.curr_T = copy.deepcopy(self.init_T)
        for plane_params, Q_matrix, point_num in self.factor_list:
            updated_Q = self.curr_T @ Q_matrix @ self.curr_T.T 
            plane_id = np.argmin(np.diag(plane_params @ updated_Q @ plane_params.T))
            target_plane = np.atleast_2d(plane_params[plane_id])
            dQ_dv = self.get_dQdv(updated_Q)
            dL_dv = np.zeros((6,))
            for idx, dQ_dv_k in enumerate(dQ_dv):
                dL_dv[idx] = target_plane @ dQ_dv_k @ target_plane.T
            alpha = 0.2 / point_num
            updated_T = (np.identity(4) + self.exp_map(alpha * dL_dv)) @ self.curr_T
            pass
    
    def exp_map(self, v):
        exp_map = np.zeros((4, 4))
        exp_map[0] = np.array([0, -v[2], v[1], v[3]])
        exp_map[1] = np.array([v[2], 0, -v[0], v[4]])
        exp_map[2] = np.array([-v[1], v[0], 0, v[5]])
        exp_map[3] = np.array([0, 0, 0, 0])
        return exp_map
        
    def get_dQdv(self, Q_matrix):
        q1, q2, q3, q4 = Q_matrix[0], Q_matrix[1], Q_matrix[2], Q_matrix[3]
        dQ_dv = np.zeros((6, 4, 4))
        # 1
        dQ_dv[0][1] = -q3
        dQ_dv[0][2] = q2
        dQ_dv[0] += dQ_dv[0].T
        # 2
        dQ_dv[1][0] = q3
        dQ_dv[1][2] = -q1
        dQ_dv[1] += dQ_dv[1].T
        # 3
        dQ_dv[2][0] = -q2
        dQ_dv[2][1] = q1
        dQ_dv[2] += dQ_dv[2].T
        # 4
        dQ_dv[3][0] = q4
        dQ_dv[3] += dQ_dv[3].T
        # 5
        dQ_dv[4][1] = q4
        dQ_dv[4] += dQ_dv[4].T
        # 6
        dQ_dv[5][2] = q4
        dQ_dv[5] += dQ_dv[5].T
        return dQ_dv