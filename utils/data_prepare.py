import numpy as np
from Keypoint_Selection_KD import KeypointSelection
import math
import time


def transform_volume(nx, ny, nphi, nx_step, ny_step, nphi_step):
    phi = np.linspace(-(nphi - 1) / 2 * nphi_step, (nphi - 1) / 2 * nphi_step, nphi)  # phi is radian
    sin_table = np.zeros(nphi)
    cos_table = np.zeros(nphi)
    for i in range(nphi):
        sin_table[i] = math.sin(phi[i])
        cos_table[i] = math.cos(phi[i])

    transform_volume = np.zeros([nx * ny * nphi, 3, 4])
    n = 0
    for i in range(nx):
        for j in range(ny):
            for k in range(nphi):
                transform_volume[n, 2, :] = np.array([0, 0, 1, 0])
                transform_volume[n, 0:2, 0:2] = np.array([[cos_table[k], -sin_table[k]],
                                                          [sin_table[k], cos_table[k]]])
                transform_volume[n, 0, 3] = -nx_step * ((nx - 1) // 2 - i)
                transform_volume[n, 1, 3] = -ny_step * ((ny - 1) // 2 - j)
                transform_volume[n, (0,1), 2] = np.array([0, 0])
                n = n + 1
    print(transform_volume[0, :, :])
    return transform_volume


# there are 4701 frames pcd in dataset, so there are 4700 pairs
class DataPrepare():
    def __init__(self, threshold, neighborhood_distance, k_neighbor, keypoint_num,
                 nx, ny, nphi, nx_step, ny_step, nphi_step, gt_pose_path, init_pose_path):
        self.threshold = threshold                          # density threshold to be selected as kp candidate
        self.neighborhood_distance = neighborhood_distance  # neighborhood radius to calculate density
        self.k_neighbor = k_neighbor                        # the knn's parameter k to calculate 3D structure tensor
        self.N = keypoint_num                               # number of key points
        self.nx, self.ny, self.nphi = nx, ny, nphi
        self.nx_step, self.ny_step, self.nphi_step = nx_step, ny_step, nphi_step
        self.transform_volume = transform_volume(nx, ny, nphi, nx_step, ny_step, nphi_step)
        self.gt_pose_path = gt_pose_path
        self.init_pose_path = init_pose_path

    def map_feature_volume(self, key_points_map_j, KDTree_map, pcd_map):      # index in last is 1 smaller than index in current
        feature_volume = np.zeros([self.nx*self.ny*self.nphi, 64, 3])
        n = 0
        for i in range(self.nx):
            # time1 = time.time()
            for j in range(self.ny):
                for k in range(self.nphi):
                    key_point_position = key_points_map_j[n, :]
                    key_point_position = np.resize(key_point_position, (3,))
                    [kk, idx_o, _] = KDTree_map.search_knn_vector_3d(key_point_position,64)  # knn $no cost
                    neighbor_of_kp_map = pcd_map.select_down_sample(idx_o)                   # pick neighbour $no cost
                    neighbor_of_kp_map = np.asarray(neighbor_of_kp_map.points)               # pcd to ndarray
                    feature_volume[n, :, :] = neighbor_of_kp_map
                    n = n + 1
            # time11 = time.time()
            # time1_ = time11- time1
            # print("Time of one nx loop: %.5f s!" %time1_)
        return feature_volume
                  # n*64*3

    def volume_calculate(self, KDTree_map, KDTree_online, pcd_map, pcd_online):
        select_KP = KeypointSelection(self.threshold, self.neighborhood_distance, self.k_neighbor, self.N, pcd_online)
        time2 = time.time()
        key_points, keypoint_index = select_KP.combinatorially_geometric_characteristic()        # KeyPoints, list
        time22 = time.time()
        time2_ = time22 - time2
        print("Time of keypoint selection: %.4f s!" % time2_)
        key_points_array = np.asarray(key_points.points)
        key_points_array_T = key_points_array.T                                                   # 3*N array
        key_points_array_T = np.r_[key_points_array_T, np.ones([1, self.N])]                      # 4*N array
        key_points_array_map = np.matmul(self.transform_volume, key_points_array_T)               # n*3*N = n*3*4 . 4*N
        current_feature_compute_points = np.zeros([self.N, 64, 3])
        map_feature_compute_points_spaced = np.zeros([self.N, self.nx*self.ny*self.nphi, 64, 3])  # attention for dim

        for j in range(self.N):
            time3 = time.time()
            if j % 32 == 0:
                print("--->processing No." + str(j+1) + "key point")
            [kk, idx_o, _] = KDTree_online.search_knn_vector_3d(pcd_online.points[keypoint_index[j]], 64)  # knn
            neighbor_of_kp_online = pcd_online.select_down_sample(idx_o)                           # pick neighbour
            neighbor_of_kp_online = np.asarray(neighbor_of_kp_online.points)                       # pcd to ndarray
            current_feature_compute_points[j, :, :] = neighbor_of_kp_online                        # 64*3

            map_feature_compute_points_spaced[j, :, :, :] = self.map_feature_volume(key_points_array_map[:, :, j], KDTree_map, pcd_map)  # (nx*ny*nphi)*64*3
            # time33 = time.time()
            # time3_ = time33 - time3
            # print("Time of processing one key point: %.4f s" % time3_)
        return current_feature_compute_points, map_feature_compute_points_spaced
        #                N*64*3                       N*(nx*ny*nphi)*64*3

    def quaternion_to_euler(self, x, y, z, w):
        yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (z * z + y * y))
        yaw = yaw * 180 / math.pi
        print(yaw.shape)
        return yaw                        # (4700,) array

    def gt_offset_calculate(self):
        gt_poses = np.loadtxt(self.gt_pose_path)
        init_poses = np.loadtxt(self.init_pose_path)
        gt_xy = gt_poses[:, (2, 3)]       # 4700*2 array
        init_xy = init_poses[:, (2, 3)]   # 4700*2 array
        gt_yaw = self.quaternion_to_euler(gt_poses[:, 5], gt_poses[:, 6], gt_poses[:, 7], gt_poses[:, 8])
        init_yaw = self.quaternion_to_euler(init_poses[:, 5], init_poses[:, 6], init_poses[:, 7], init_poses[:, 8])
        gt_offset = np.c_[gt_xy - init_xy, gt_yaw - init_yaw]
        return gt_offset                  # (4700,3) array
