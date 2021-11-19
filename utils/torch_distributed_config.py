import logging
import os
import pickle
import random
import shutil
import subprocess

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def init_distributed_device(launcher, tcp_port, local_rank=None, backend='nccl'):
    """
    modified from https://github.com/open-mmlab/mmdetection
    Args:
        tcp_port:
        backend:

    Returns:

    """
    if launcher == 'slurm':
        logging.info(f"config distributed training with launcher: {launcher}")
        proc_id = int(os.environ['SLURM_PROCID'])
        ntasks = int(os.environ['SLURM_NTASKS'])
        node_list = os.environ['SLURM_NODELIST']
        num_gpus = torch.cuda.device_count()
        torch.cuda.set_device(proc_id % num_gpus)
        addr = subprocess.getoutput('scontrol show hostname {} | head -n1'.format(node_list))
        os.environ['MASTER_PORT'] = str(tcp_port)
        os.environ['MASTER_ADDR'] = addr
        os.environ['WORLD_SIZE'] = str(ntasks)
        os.environ['RANK'] = str(proc_id)
        dist.init_process_group(backend=backend)

        total_gpus = dist.get_world_size()
        rank = dist.get_rank()
        return total_gpus, rank
    elif launcher == 'pytorch':
        logging.info(f"config distributed training with launcher: {launcher}")
        assert local_rank is not None
        if mp.get_start_method(allow_none=True) is None:
            mp.set_start_method('spawn')

        num_gpus = torch.cuda.device_count()
        torch.cuda.set_device(local_rank % num_gpus)
        dist.init_process_group(
            backend=backend,
            init_method='tcp://127.0.0.1:%d' % tcp_port,
            rank=local_rank,
            world_size=num_gpus
        )
        rank = dist.get_rank()
        os.environ['WORLD_SIZE'] = str(num_gpus)
        os.environ['RANK'] = str(local_rank)
        return num_gpus, rank
    else:
        raise NotImplementedError


def get_dist_info():
    if torch.__version__ < '1.0':
        initialized = dist._initialized
    else:
        if dist.is_available():
            initialized = dist.is_initialized()
        else:
            initialized = False
    if initialized:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1
    return rank, world_size


def merge_results_dist(result_part, size, tmpdir):
    rank, world_size = get_dist_info()
    os.makedirs(tmpdir, exist_ok=True)

    dist.barrier()
    pickle.dump(result_part, open(os.path.join(tmpdir, 'result_part_{}.pkl'.format(rank)), 'wb'))
    dist.barrier()

    if rank != 0:
        return None

    part_list = []
    for i in range(world_size):
        part_file = os.path.join(tmpdir, 'result_part_{}.pkl'.format(i))
        part_list.append(pickle.load(open(part_file, 'rb')))

    ordered_results = []
    for res in zip(*part_list):
        ordered_results.extend(list(res))
    ordered_results = ordered_results[:size]
    shutil.rmtree(tmpdir)
    return ordered_results





