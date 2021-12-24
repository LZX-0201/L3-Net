import logging
import traceback
from utils.config.Configuration import Configuration
from factory.model_factory import ModelFactory
from factory.dataset_factory import DatasetFactory
import torch
import os
import math
import numpy as np


def evaluate(offset_delta):  # (Num of test data, 3)
    num_of_data = offset_delta.shape[0]
    offset_delta[:, 2] = offset_delta[:, 2] * 180 / math.pi
    lat_RMS = np.sqrt((sum(offset_delta[:, 0] * (offset_delta[:, 0])) / num_of_data))
    long_RMS = np.sqrt((sum(offset_delta[:, 1] * (offset_delta[:, 1])) / num_of_data))
    yaw_RMS = np.sqrt((sum(offset_delta[:, 2] * (offset_delta[:, 2])) / num_of_data))
    yaw_max = max(offset_delta[:, 2])
    horiz_error = np.sqrt((offset_delta[:, 0] * (offset_delta[:, 0]) + offset_delta[:, 1] * (offset_delta[:, 1])))
    horiz_RMS = np.sqrt((sum(horiz_error * (horiz_error)) / num_of_data))
    horiz_max = max(horiz_error)
    pct_distance_1 = np.sum(horiz_error < 0.1) / num_of_data
    pct_distance_2 = np.sum(horiz_error < 0.2) / num_of_data
    pct_distance_3 = np.sum(horiz_error < 0.3) / num_of_data
    pct_degree_1 = np.sum(offset_delta[:, 2] < 0.1) / num_of_data
    pct_degree_3 = np.sum(offset_delta[:, 2] < 0.3) / num_of_data
    pct_degree_6 = np.sum(offset_delta[:, 2] < 0.6) / num_of_data
    evaluation = np.array([horiz_RMS, horiz_max, long_RMS, lat_RMS, pct_distance_1, pct_distance_2, pct_distance_3,
                            yaw_RMS, yaw_max, pct_degree_1, pct_degree_3, pct_degree_6])
    return evaluation


if __name__ == '__main__':
    try:
        # manage config
        logging_logger = logging.getLogger()
        logging_logger.setLevel(logging.NOTSET)

        config = Configuration()
        args = config.get_shell_args_train()

        # default test settings
        args.for_train = False
        args.shuffle = False

        config.load_config(args.cfg_dir)
        config.overwrite_config_by_shell_args(args)

        # instantiating all modules by non-singleton factory
        dataset = DatasetFactory.get_singleton_dataset(config.dataset_config)
        data_loader = dataset.get_data_loader()

        model = ModelFactory.get_model(config.model_config)
        model.load_model_paras_from_file(args.check_point_file)
        model.set_eval()
        model.set_device("cuda:0")

        with torch.no_grad():
            for step, data in enumerate(data_loader):
                logging.info(f"*****************************************************step:{step}/{len(data_loader)}********************************************************")

                online_data_batch, map_data_batch, gt_data_batch = dataset.load_data_to_gpu(data)

                pred_offset = model(online_data_batch, map_data_batch, gt_data_batch)  # (batchsize, 3)
                print("gt_offset")
                print(gt_data_batch)
                print("pred_offset")
                print(pred_offset)
                offset_delta_batch = pred_offset - gt_data_batch  # (batchsize, 3)
                offset_delta_batch = offset_delta_batch.cpu()
                offset_delta_batch_np = offset_delta_batch.numpy()

                if step == 0:
                    offset_delta = offset_delta_batch_np
                else:
                    offset_delta = np.concatenate((offset_delta, offset_delta_batch_np), axis=0)

            evaluation = evaluate(offset_delta)
            print(evaluation)
            np.save("evaluation.npy", evaluation)


    except Exception as e:
        logging.exception(traceback.format_exc())
        exit(-1)
