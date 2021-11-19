import os
import logging
import signal
import traceback
import subprocess
from utils.config.Configuration import Configuration
from factory.model_factory import ModelFactory
from factory.dataset_factory import DatasetFactory
from factory.trainer_factory import TrainerFactory
from utils.logger.basic_logger import BasicLogger
from utils.logger.mute_logger import MuteLogger


def sigint_handler(sig, frm):
    print("You kill the program.")
    try:
        if args.screen_log is not None and logger is not None:
            logger.copy_screen_log(args.screen_log)
            logger.log_model_params(model)
        exit(0)
    except Exception as e_:
        logging.exception(traceback.format_exc())
        exit(-1)


if __name__ == '__main__':
    signal.signal(signal.SIGINT, sigint_handler)
    git_version = subprocess.check_output(["git", 'rev-parse', 'HEAD']).strip().decode()
    logging.info(f'Your program version is {git_version}')
    try:
        # manage config
        logging_logger = logging.getLogger()
        logging_logger.setLevel(logging.NOTSET)
        ch = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(name)s: %(message)s')
        ch.setFormatter(formatter)

        config = Configuration()
        args = config.get_shell_args_train()
        config.load_config(args.cfg_dir)
        config.overwrite_config_by_shell_args(args)

        # instantiating all modules by non-singleton factory
        dataset = DatasetFactory.get_singleton_dataset(config.dataset_config)
        print(config.dataset_config)
        print(config.model_config)
        model = ModelFactory.get_model(config.model_config)
        trainer = TrainerFactory.get_trainer(config.training_config)
        if config.extra_config['distributed']:
            logging.info("using distributed training......")
            trainer.config_distributed_computing(launcher=config.extra_config['launcher'],
                                                 tcp_port=config.extra_config['tcp_port'],
                                                 local_rank=config.extra_config['local_rank'])
        logger = None
        if (not config.extra_config['distributed']) or (os.environ['RANK'] == str(0)):
            logger = BasicLogger.get_logger(config)
            logger.log_config(config)
        else:
            logger = MuteLogger.get_logger(config)
        trainer.set_model(model)
        trainer.set_dataset(dataset)
        trainer.set_logger(logger)
        if args.check_point_file is not None:
            trainer.load_state(args.check_point_file)
        logging.info("Preparation done! Trainer run!")
        trainer.run()
        if args.screen_log is not None:
            logger.copy_screen_log(args.screen_log)
    except Exception as e:
        if args.screen_log is not None and logger is not None:
            logger.copy_screen_log(args.screen_log)
            logger.log_model_params(model)
        logging.exception(traceback.format_exc())
        exit(-1)
