import open3d as o3d
import numpy as np
import math
import time

class KeypointSelection:
    def __init__(self, threshold, neighborhood_distance, k_neighbor, keypoint_num, pcd):
        self.pcd = pcd
        print(pcd)
        point = np.asarray(pcd.points)
        self.point = point
        self.threshold = threshold                          # density threshold to be selected as kp candidate
        self.neighborhood_distance = neighborhood_distance  # neighborhood radius to calculate density
        self.k_neighbor = k_neighbor                        # the knn's parameter k to calculate 3D structure tensor
        self.keypoint_num = keypoint_num                    # number of key points

    def traverse_density(self):
        t1 = time.time()
        pcd = self.pcd
        pcd.paint_uniform_color([0.5, 0.5, 0.5])    # color the points
        pcd_tree = o3d.geometry.KDTreeFlann(pcd)    # init a KDTree class
        pointNum = self.point.shape[0]
        selected_index = []                         # store the index of the candidate points

        for i in range(pointNum):
            # if i % 5000 == 0:
            #     print("candidate selection: " + str(round(i/1208.83,3)) + "%")
            [k, idx, _] = pcd_tree.search_radius_vector_3d(pcd.points[i], self.neighborhood_distance)  # RNN
            if k >= self.threshold:
                selected_index.append(i)

        np.asarray(pcd.colors)[selected_index[1:], :] = [0, 1, 0]     # color the candidate points
        t2 = time.time()
        print(t2 - t1)

        return pcd, pcd_tree, selected_index

    def structure_tensor(self):
        pcd, pcd_tree, selected_index = self.traverse_density()
        structure_tensor = np.zeros((len(selected_index), 3, 3))
        print('totally' + str(len(selected_index)) + 'candidate points')

        for i in range(len(selected_index)):
            [k, idx, _] = pcd_tree.search_knn_vector_3d(pcd.points[selected_index[i]], self.k_neighbor)  # KNN
            neighbor_point = pcd.select_down_sample(idx)
            neighbor_array = np.asarray(neighbor_point.points)
            mean = k * neighbor_array.mean(axis=0) / (k+1)            # the average coordinate of K neighour points
            # print(mean)
            if i % 20000 == 0:
                print("structure_tensor computing，No. "+str(i) + "/" + str(len(selected_index)) + " candidate point")

            S = np.zeros((3, 3))                                      # to store 3D structure tensor
            for j in range(k):
                temp_point = neighbor_array[j] - mean
                temp_point1 = temp_point.reshape(3, 1)
                # print(temp_point1)
                temp_point2 = temp_point1.reshape(1, 3)
                S = S + temp_point1.dot(temp_point2)
            structure_tensor[i] = S / (k + 1)
            # print(structure_tensor[i])

        return pcd, selected_index, structure_tensor

    def combinatorially_geometric_characteristic(self):
        scatter_w = 0  # scatter相较于linear的权重
        pcd, selected_index, structure_tensor = self.structure_tensor()
        point = np.asarray(pcd.points)
        # lowest = np.min(point, axis=0)[2]
        # point = point + lowest
        selected_point = pcd.select_down_sample(selected_index)
        number_of_point = structure_tensor.shape[0]
        combined_geometric_characteristic = np.zeros(number_of_point)   # combinatorially geometric characteristic
        geometrical_feature = np.zeros((number_of_point, 4))            # row i is the linear, planar, scatter of No.i canditate point

        print("combinatorially geometric characteristic computing")
        for i in range(number_of_point):
            # if i % 20000 == 0:
            #     print("combinatorially geometric characteristic computing，No." + str(i) + "candidate")
            e_vals, e_vecs = np.linalg.eig(structure_tensor[i])
            e_vals_sort = np.sort(e_vals)[::-1]
            # print(e_vals_sort)
            sigma_1 = e_vals_sort[0]
            sigma_2 = e_vals_sort[1]
            sigma_3 = e_vals_sort[2]
            geometrical_feature[i][0] = (sigma_1 - sigma_2) / sigma_1   # neighbor's linear
            geometrical_feature[i][1] = (sigma_2 - sigma_3) / sigma_1   # neighbor's planar
            geometrical_feature[i][2] = (sigma_3 / sigma_1)             # neighbor's scatter
            geometrical_feature[i][3] = sigma_3 / (sigma_1 + sigma_2 + sigma_3)
            # combined_geometric_characteristic[i] = geometrical_feature[i][0] + scatter_w * geometrical_feature[i][2]
            combined_geometric_characteristic[i] = geometrical_feature[i][0] + scatter_w * geometrical_feature[i][2] - 8000*geometrical_feature[i][3]
            # combined_geometric_characteristic[i] = geometrical_feature[i][0] + scatter_w * geometrical_feature[i][2]
            # if -0.5 <= point[selected_index[i]][2] <= 0.1:
            #     combined_geometric_characteristic[i] = combined_geometric_characteristic[i] - 1/point[selected_index[i]][2]

        temp_index = np.argsort(combined_geometric_characteristic)
        temp_index = temp_index[0:self.keypoint_num]
        keypoints = selected_point.select_down_sample(temp_index)

        # print(np.max(geometrical_feature, axis=0))
        # print(np.min(geometrical_feature, axis=0))

        keypoint_index = []
        for i in range(len(temp_index)):
            keypoint_index.append(selected_index[temp_index[i]])

        np.asarray(pcd.colors)[keypoint_index[1:], :] = [1, 0, 0]     # color the keypoints as red
        # o3d.visualization.draw_geometries([pcd])

        return keypoints, keypoint_index
