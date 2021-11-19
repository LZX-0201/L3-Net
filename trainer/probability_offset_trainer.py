from trainer.trainer_base import TrainerBase
import torch
import logging


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
        self.data_loader = self.dataset.get_data_loader()
        # sd = self.model.state_dict()
        # print(sd.values())

        # Training Loop
        # Lists to keep track of progress

        self.global_step = 0

        for epoch in range(self.max_epoch):
            print(epoch)
            self.epoch = epoch
            for step, data in enumerate(self.data_loader):
                self.optimizer.zero_grad()
                # print(data[0].shape)
                online_data_batch, map_data_batch, gt_data_batch = self.dataset.load_data_to_gpu(data)
                # print(online_data_batch[0, 1, :, :])
                pred_offset = self.model(online_data_batch, map_data_batch, gt_data_batch)
                print("gt_offset")
                print(gt_data_batch)
                print("pred_offset")
                print(pred_offset)
                loss = get_loss(pred_offset, gt_data_batch, self.alpha)
                loss.backward()
                self.optimizer.step()

                # print current status and logging
                if not self.distributed or self.rank == 0:
                    logging.info(f'[loss] Epoch={epoch}/{self.max_epoch}, step={step}/{len(self.data_loader)}\t'
                                 f'loss={loss:.6f}\t'
                                 )
                    # for i in loss_dict.keys():
                    #     self.logger.log_data(i, loss_dict[i].item(), True)
                self.step = step
                self.global_step += 1
            if not self.distributed or self.rank == 0:
                self.logger.log_model_params(self.model)
