import open3d as o3d
from data_prepare import DataPrepare
import numpy as np
import time
import yaml


def load_hparam(filenam):
    stream = open(filenam, 'r')
    docs = yaml.load_all(stream, Loader=yaml.FullLoader)
    haparam_dict = {}
    for doc in docs:
        for k,v in doc.items():
            haparam_dict[k] = v
    return haparam_dict

if __name__ == '__main__':
    filename = './config/samples/sample_L3Net_data_prepare/root_config.yaml'  # config file root
    haparam_dict = load_hparam(filename)
    data_prepare = DataPrepare(haparam_dict['threshold'], haparam_dict['neighborhood_distance'],
                               haparam_dict['k_neighbor'], haparam_dict['N'],
                               haparam_dict['nx'], haparam_dict['ny'], haparam_dict['nphi'],
                               haparam_dict['nx_step'], haparam_dict['ny_step'], haparam_dict['nphi_step'],
                               haparam_dict['gt_pose_path'], haparam_dict['init_pose_path'])

    n = haparam_dict['nx'] * haparam_dict['ny'] * haparam_dict['nphi']
    map_data = np.zeros([haparam_dict['N'], n, 64, 3])
    online_data = np.zeros([haparam_dict['N'], 64, 3])
    KDTree_dict = {}
    pcd_dict = {}

    for i in range(4701):
        if i%400 == 0:
            print("processed %s/4700 pcd to KDTree" %i)
        temp_pcd = o3d.io.read_point_cloud(haparam_dict['dataroot_load'] + str(i + 1) + ".pcd")
        KDTree_dict[i + 1] = o3d.geometry.KDTreeFlann(temp_pcd)
        pcd_dict[i + 1] = temp_pcd

    for i in range(4700):
        time1 = time.time()
        print()
        print("----------------------procession No." + str(i+1), "pcd pair----------------------")
        online_data_volume, map_data_volume = data_prepare.volume_calculate(KDTree_dict[i+1], KDTree_dict[i+2], pcd_dict[i+1], pcd_dict[i+2])
        np.save(haparam_dict['save_root_online'] + str(i+1) + '.npy', online_data_volume)
        np.save(haparam_dict['save_root_map'] + str(i+1) + '.npy', map_data_volume)
        time11 = time.time()
        time1_ = time11 - time1
        print("time of one frame: %.3f s!" % time1_)
    gt_offset = data_prepare.gt_offset_calculate()  # 4700*3 ndarray
    np.save(haparam_dict['save_root_offset'], gt_offset)
    print(gt_offset.shape)

