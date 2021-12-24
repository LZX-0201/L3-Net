from trainer.trainer_base import TrainerBase
import torch
import logging
from petrel_client.client import Client
from petrel_client.utils.data import DataLoader
import time
from datetime import datetime
import numpy as np


trainAcc_txt = "/mnt/cache/lizhaoxin.vendor/ADModel_Pro/train_acc.txt"

def save_checkpoint_state(epoch, model, optimizer):
    checkpoint = {
            'epoch': epoch,
            'model_paras': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            # 'scheduler_state_dict':scheduler.state_dict(),
                }
    path = "/mnt/cache/lizhaoxin.vendor/ADModel_Pro/checkpoints/epoch_" + str(epoch) + "_check_point.pth"
    torch.save(checkpoint, path)


def get_loss(pred_offset, gt_offset, alpha):         # tensor(batchsize, 3)
    delta = pred_offset - gt_offset                  # tensor(batchsize, 3)
    print('delta')
    print(delta)
    loss_each_batch = delta[:, 0].mul(delta[:, 0]) + delta[:, 1].mul(delta[:, 1]) + alpha * delta[:, 2].mul(delta[:, 2])  # tensor([batchsize])
    # print('loss_each_batch')
    # print(loss_each_batch)
    loss_batch = loss_each_batch.sum()                       # tensor([])
    # print("loss in get_loss")
    # print(loss_batch)
    loss_batch = loss_batch.cuda()
    return loss_batch


class PoseErrRegressionTrainer(TrainerBase):
    def __init__(self, config):
        super(PoseErrRegressionTrainer, self).__init__()
        self.max_epoch = config['epoch']
        self.optimizer_config = config['optimizer']
        self.device = torch.device(config['device'])
        self.alpha = config['alpha']
        self.data_loader = None
        client = Client('/mnt/cache/lizhaoxin.vendor/petreloss.conf')

    def set_optimizer(self, optimizer_config):
        optimizer_ref = torch.optim.__dict__[self.optimizer_config['type']]
        self.optimizer = optimizer_ref(self.model.parameters(), **optimizer_config['paras'])

    def run(self):
        if not self.check_ready():
            raise ModuleNotFoundError("The trainer not ready. Plz set model/dataset first")
        torch.autograd.set_detect_anomaly(True)
        self.set_optimizer(self.optimizer_config)
        # self.model.set_device(self.device)  # send the model to GPU
        self.model.cuda()
        # dataloader = DataLoader(dataset=self.dataset, prefetch_factor=2, persistent_workers=True)
        time1 = time.time()
        self.data_loader = self.dataset.get_data_loader()
        time11 = time.time()
        print("--------------------------------------------------------------------------------------------------------------------")
        print("the time to init dataloader is: " + str(time11 - time1))
        print("--------------------------------------------------------------------------------------------------------------------")
        # sd = self.model.state_dict()
        # print(sd.values())

        # Training Loop
        # Lists to keep track of progress

        self.global_step = 0

        for epoch in range(self.max_epoch):
            print(epoch)
            self.epoch = epoch
            for step, data in enumerate(self.data_loader):
                # continue
                self.optimizer.zero_grad()
                # print(data[0].shape)
                time2 = time.time()
                online_data_batch, map_data_batch, gt_data_batch = self.dataset.load_data_to_gpu(data)
                time22 = time.time()
                print("*******************************************Time to load data to GPU is: " + str(time22 - time2) + " s************************************************************************")
                # print(online_data_batch[0, 1, :, :])
                time3 = time.time()
                pred_offset = self.model(online_data_batch, map_data_batch, gt_data_batch)
                # pred_offset = torch.ones([1, 3])
                # pred_offset = pred_offset.cuda()
                print("gt_offset")
                print(gt_data_batch)
                time33 = time.time()
                print("*******************************************Time of forward propagation is: " + str(time33 - time3) + "***********************************************************************")
                print("pred_offset")
                print(pred_offset)
                loss = get_loss(pred_offset, gt_data_batch, self.alpha)
                # continue
                time4 = time.time()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters=self.model.parameters(), max_norm=2, norm_type=3)
                self.optimizer.step()
                time44 = time.time()
                print("*******************************************Time of backword propagation is: " + str(time44 - time4) + "**********************************************************************")

                # print current status and logging
                print("****************************************************************************one iteration done*************************************************************************")
                if not self.distributed or self.rank == 0:
                    logging.info(f'********************************[loss] Epoch={epoch}/{self.max_epoch}, step={step}/{len(self.data_loader)}\t'
                                 f'loss={loss:.6f}\t'
                                 )
                print("***********************************************************************************************************************************************************************")
                    # for i in loss_dict.keys():
                    #     self.logger.log_data(i, loss_dict[i].item(), True)
                self.step = step
                self.global_step += 1
                time5 = time.time()
                output = "%s：Epoch [%d] Step [%d]  train Loss : %f" % (datetime.now(), epoch, step, loss)
                with open(trainAcc_txt,"a+") as f:
                    f.write(output+'\n')
                    f.close
                print("*******************************************Time of one iteration is: " + str(time5-time2) + "**************************************************************************")
            save_checkpoint_state(epoch, self.model, self.optimizer)
            if not self.distributed or self.rank == 0:
                self.logger.log_model_params(self.model)
