import os
import pickle
import logging
from model.model_base import ModelBase
from dataset.dataset_base import DatasetBase
from utils.logger.basic_logger import BasicLogger
from utils.torch_distributed_config import init_distributed_device
import torch
import torch.nn as nn


class TrainerBase:
    def __init__(self):
        self.model = None
        self.dataset = None
        self.data_loader = None
        self.device = None
        self.optimizer = None
        self.optimizer_config = None
        self.max_epoch = 0
        self.epoch = 0
        self.step = 0
        self.global_step = 0
        self.logger = None
        self.distributed = False
        self.total_gpus = 0
        self.rank = None
        self.sync_bn = False
        self.launcher = 'none'

    def get_training_status(self):
        training_status = {
            "max_epoch": self.max_epoch,
            "epoch": self.epoch,
            "step": self.step,
            "global_step": self.global_step
        }
        return training_status

    def check_ready(self):
        if self.model is None:
            return False
        if self.dataset is None:
            return False
        return True

    def run(self):
        if not self.check_ready():
            raise ModuleNotFoundError("The trainer not ready. Plz set model/dataset first")
        self.set_optimizer(self.optimizer_config)
        if self.sync_bn:
            self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
        self.model.set_device(self.device)
        self.data_loader = self.dataset.get_data_loader(distributed=self.distributed)
        if self.distributed:
            self.model = nn.parallel.DistributedDataParallel(self.model,
                                                             device_ids=[self.rank % torch.cuda.device_count()])
            # why not set self.model = model.module?
            # In case the need to use model(data) directly.
        # [Instruction] add code in sub class and using super to run this function for general preparation

    def set_model(self, model):
        if not isinstance(model, ModelBase):
            raise TypeError
        self.model = model

    def set_dataset(self, dataset):
        if not isinstance(dataset, DatasetBase):
            raise TypeError
        self.dataset = dataset

    def set_optimizer(self, optimizer_config):
        raise NotImplementedError

    def load_state(self, log_file):
        if not os.path.exists(log_file):
            raise FileNotFoundError(f'file not exist:{log_file}')
        params = None
        try:
            params = torch.load(log_file)
        except Exception as e:
            with open(log_file, 'rb') as f:
                params = pickle.load(f)
        if params is not None:
            if self.model is not None:
                self.model.load_model_paras(params)
            else:
                raise AssertionError('model does not exist.')
            logging.info(f"loaded model params:{log_file}")
            # todo: retrive all status including: optimizer epoch log folder...
            status = params['status']
        else:
            raise AssertionError('Fail to load params for model.')


    def set_logger(self, logger):
        if not isinstance(logger, BasicLogger):
            raise TypeError("logger must be with the type: BasicLogger")
        self.logger = logger
        self.logger.register_status_hook(self.get_training_status)

    def config_distributed_computing(self, launcher, tcp_port=None, local_rank=None):
        self.launcher = launcher
        if self.launcher == 'none':
            self.distributed = False
            self.total_gpus = 1
        else:
            self.total_gpus, self.rank = \
                init_distributed_device(self.launcher, tcp_port, local_rank, backend='nccl')
            self.distributed = True
            device_id = self.rank % torch.cuda.device_count()
            self.device = torch.device(device_id)


