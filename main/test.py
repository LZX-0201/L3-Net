import logging
import pickle
import traceback
from utils.config.Configuration import Configuration
from factory.model_factory import ModelFactory
from factory.dataset_factory import DatasetFactory
import torch
import os
import datetime
from trainer.vae_gan_trainer_pvrcnn_instance import VAEGANTrainerPVRCNNInstance
# todo: support 1.distributed testing 2.logger custimized for testing
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
        output_data = {
            "predictions":[],
            "model_para_file": args.check_point_file
        }
        output_dir = "../output"
        date_time_str = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        output_file = os.path.join(output_dir, "test_log-" + date_time_str + '.pkl')

        for step, data in enumerate(data_loader):
            logging.info(f"step:{step}/{len(data_loader)}")
            # 0.data preparation
            # trans all data to gpu device
            data_loader.dataset.load_data_to_gpu(data)

            # get target model output
            target_res = model.target_model(data)
            target_boxes = target_res['dt_lidar_box']

            # align the gt_boxes and target_res_processed
            gt_box = data['gt_boxes']
            max_obj = 25
            gt_box_ext = torch.zeros([gt_box.shape[0], max_obj, 8])
            gt_box_ext[:, :gt_box.shape[1], :] = gt_box[:, :max_obj, :]
            gt_box_ext = gt_box_ext.to(model.device)

            gt_valid_mask = (gt_box[:, :, -1] > 0)
            gt_valid_elements = gt_valid_mask.sum()
            if not gt_valid_elements > 0:
                raise ZeroDivisionError("wrong gt valid number")

            if gt_box_ext.shape != target_boxes.shape:
                raise TypeError("gt_box and target_box must have same shape")

            with torch.no_grad():
                # generator_input = data['points']
                generator_input = VAEGANTrainerPVRCNNInstance.get_instance_cloud(data)
                generator_output, point_feature, _, _ = model.generator(generator_input, gt_box_ext)
                # save by frm
                prediction_inx = 0
                for batch_inx in range(gt_valid_mask.shape[0]):
                    box_indices = gt_valid_mask[batch_inx].nonzero()
                    cur_frm_result = {'ini_gt_inx': [],
                                      'gt_box': [],
                                      'prediction': [],
                                      'target_box': [],
                                      'frame_id': data['frame_id'][batch_inx]
                                      }
                    for box_inx in box_indices:
                        box_gt = gt_box[batch_inx][box_inx].cpu().numpy()
                        prediction = generator_output[prediction_inx].cpu().numpy()
                        prediction_inx += 1
                        target_box = target_boxes[batch_inx][box_inx].cpu().numpy()
                        print("box_gt", box_gt)
                        print("prediction", prediction)
                        print("target_box", target_box)
                        print("===========================")
                        cur_frm_result['ini_gt_inx'].append(box_inx.item())
                        cur_frm_result['gt_box'].append(box_gt)
                        cur_frm_result['prediction'].append(prediction)
                        cur_frm_result['target_box'].append(target_box)

                    output_data['predictions'].append(cur_frm_result)
        with open(output_file, 'wb') as f:
            pickle.dump(output_data, f)

    except Exception as e:
        logging.exception(traceback.format_exc())
        exit(-1)
