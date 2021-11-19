import builtins
import os
import datetime
import logging
import pickle
import shutil
import torch
import subprocess
from model.model_base import ModelBase
from utils.config.Configuration import Configuration
from tensorboardX import SummaryWriter

# todo: add lock while used in multiprocessing...


class BasicLogger:
    logger = None

    def __new__(cls, *args, **kwargs):
        if cls.logger is None:
            cls.logger = super(BasicLogger, cls).__new__(cls)
            cls.logger.__initialized = False
        return cls.logger

    def __init__(self, config):
        if self.__initialized:
            return
        if not isinstance(config, Configuration):
            raise TypeError("input must be the Configuration type!")
        config_dict = config.get_complete_config()
        if "logging" not in config_dict.keys():
            raise KeyError("Not config on logger has been found!")
        self._program_version = None
        self._monitor_dict = {}
        self._status_hook = None
        self.root_log_dir = config_dict['logging']['path']
        self.log_suffix = config_dict['logging']['suffix']
        self._ckpt_eph_interval = config_dict['logging']['ckpt_eph_interval']
        if not isinstance(self._ckpt_eph_interval, int):
            self._ckpt_eph_interval = 0
        date_time_str = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        if self.log_suffix is None or len(self.log_suffix) == 0:
            self._cur_instance_root_log_dir = date_time_str
        else:
            self._cur_instance_root_log_dir = "-".join([date_time_str, self.log_suffix])
        self._tensor_board_log_dir = os.path.join(self.root_log_dir, self._cur_instance_root_log_dir, "tensor_board")
        self._data_log_dir = os.path.join(self.root_log_dir, self._cur_instance_root_log_dir, "data_log")
        self._model_para_log_dir = os.path.join(self.root_log_dir, self._cur_instance_root_log_dir, "model_paras_log")
        os.makedirs(os.path.join(self.root_log_dir, self._cur_instance_root_log_dir))
        os.makedirs(self._tensor_board_log_dir)
        os.makedirs(self._data_log_dir)
        os.makedirs(self._model_para_log_dir)
        # add version file
        version_file = os.path.join(self.root_log_dir, self._cur_instance_root_log_dir, "version.txt")
        self.get_program_version()
        self.log_version_info(version_file)
        self._tensor_board_writer = SummaryWriter(self._tensor_board_log_dir)
        self._data_pickle_file = os.path.join(self._data_log_dir, "data_bin.pkl")
        self.__initialized = True

    def get_program_version(self):
        git_version = None
        try:
            git_version = subprocess.check_output(["git", 'rev-parse', 'HEAD']).strip().decode()
            if self._program_version is None:
                self._program_version = git_version
        except:
            pass
        return git_version

    def log_version_info(self, file_name):
        with open(file_name, 'w') as f:
            f.write(f'Current version:{self._program_version}')

    def log_config(self, config):
        if not isinstance(config, Configuration):
            raise TypeError("Please input a valid Configuration instance or reference")
        config.pack_configurations(os.path.join(self.root_log_dir, self._cur_instance_root_log_dir))

    def log_data(self, data_name, data_content, add_to_tensorboard=False):
        status = self._status_hook()
        if isinstance(data_content, builtins.float):
            self._log_scalar(status, data_name, data_content, add_to_tensorboard)
            return
        if isinstance(data_content, builtins.int):
            self._log_scalar(status, data_name, data_content, add_to_tensorboard)
            return

    def _add_to_pickle(self, status, data_name, data_content):
        with open(self._data_pickle_file, 'ab') as f:
            pickle.dump({
                "status": status,
                "name": data_name,
                "content": data_content
            }, f)

    def _log_scalar(self, status, data_name, data_content, add_to_tensorboard=False):
        if data_name not in self._monitor_dict.keys():
            self._monitor_dict[data_name] = []
        self._monitor_dict[data_name].append((status, data_content))
        if add_to_tensorboard:
            self._tensor_board_writer.add_scalar(data_name, data_content, global_step=self._status_hook()['global_step'])
        self._add_to_pickle(status, data_name, data_content)

    def log_model_params(self, model):
        # skip when ckpt_eph_interval is set
        status = self._status_hook()
        epoch = status['epoch']
        step = status['step']
        if epoch % self._ckpt_eph_interval != 0:
            return
        if model is not None:
            if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                model = model.module

        if not isinstance(model, ModelBase):
            raise TypeError("input type must have class attribute of ModelBase!")
        para_dict = {
            "status": status,
            "model_paras": model.state_dict()
        }
        date_time_str = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        pickle_name = "-".join([f'model_ckpt-epoth{epoch}-step{step}', date_time_str]) + ".pt"
        with open(os.path.join(self._model_para_log_dir, pickle_name), 'wb') as f:
            # pickle.dump(para_dict, f)
            torch.save(para_dict, f)
        logging.info(f'Log model state dict as: {pickle_name}')

    def register_status_hook(self, fn):
        if not callable(fn):
            raise TypeError(f'input must be a function!')
        self._status_hook = fn

    @classmethod
    def get_logger(cls, config=None):
        if cls.logger is not None:
            if config is not None:
                logging.warning("input config for logger will be ignored")
            return cls.logger
        if config is None:
            raise ValueError("config must be set")
        else:
            cls.logger = BasicLogger(config)
            return cls.logger

    def copy_screen_log(self, file_path):
        shutil.copy(file_path, os.path.join(self.root_log_dir,
                                            self._cur_instance_root_log_dir, os.path.basename(file_path)))






