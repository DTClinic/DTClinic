import sys
import os

import datetime
import logging
#import portpicker
#import requests
import torch.distributed
#import pkg_resources
import 

from .data import DataLoader, DistributedSampler
from .parallel import DistributedDataParallel, DataParallel
from .loss import loss
sys.path.append("..")
sys.path.append(".")
import status

STATUS = status.STATUS

LOG_FORMAT = '{name}:{levelname} [{asctime}] {message}'
DATE_FORMAT = '%Y-%m-%d %H:%M:%S'

LOG = logging.getLogger(__name__)
LOG.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT,
                                            style='{'))
LOG.addHandler(ch)

def check_port(ip, port=80):
    s = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
    try:
        s.connect((ip, port))
        s.shutdown(2)
        print('%s:%d is used' % (ip, port))
        return True
    except socket.error as e:
        print('%s:%d is unused' % (ip, port))
        return False

def debug_test(func, *args, **kwargs):
    try:
        os.environ['NCCL_DEBUG'] = 'INFO'
        func(*args, **kwargs)
    except Exception as e:
        print(repr(e))
        print_summary()
        print("If got out of memory error, please use a smaller batch size or use more GPU devices.")



def init_process_group(backend,
                       init_method=None,
                       timeout=datetime.timedelta(0, 1800), 
                       world_size=-1, 
                       rank=-1, 
                       store=None, 
                       group_name='', 
                       pg_options=None):
    """
    Initializes the default distributed process group.
    Args:
        backend (str or Backend): The backend to use. Use "nccl" for multi-GPU
            training else "gloo".
        init_method (str, optional): URL specifying how to initialize the
                                     process group.
        world_size (int, optional): Number of processes participating in
                                    the job
        rank (int, optional): Rank of the current process (it should be a
                              number between 0 and ``world_size``-1).
    If init_method, world_size and rank is NOT provided, typically in the environment.
    We will try to infer them through environment
    variables MASTER_ADDR, MASTER_PORT, WORLD_SIZE, and RANK respectively.
    """
    STATUS.backend = backend
    if backend == 'nccl':
        if not torch.cuda.is_available():
            error = "CUDA not available to torch. Please use gloo backend. If you want to use GPU, please reinstall torch or check the versions of NVIDIA driver and NVIDIA manager."
            STATUS.errors.append(error)
            LOG.error(error)

    if not torch.distributed.is_available():
        msg = "torch.distributed not available. If you are using MacOS, please set rebiuld torch from source USE_DISTRIBUTED=1, or change your device. Otherwise, please reinstall torch or rebuild torch from source with USE_DISTRIBUTED=1."
        STATUS.vulnarables.append(msg)
        LOG.info(msg)

    # check environment variables
    master_addr = None
    if 'MASTER_ADDR' not in os.environ:
        error = "please set environmental variable MASTER_ADDR."
        STATUS.errors.append(error)
        LOG.error(error)
    else:
        master_addr = os.environ["MASTER_ADDR"]
    STATUS.master_addr = master_addr
    LOG.info(f"MASTER_ADDR={master_addr}")
    
    master_port = None
    if 'MASTER_PORT' not in os.environ:
        error = "please set environmental variable MASTER_PORT."
        STATUS.errors.append(error)
        LOG.error(error)
    else:
        master_port = os.environ["MASTER_PORT"]
    STATUS.master_port = master_port
    LOG.info(f"MASTER_PORT={master_port}")

    if not check_port(master_addr, master_port):
        error = f"Address or port already in use. Please use an availble port for {master_addr}"
        STATUS.errors.append(error)
        LOG.error(error)
        
    if world_size == -1:
        if "WORLD_SIZE" in os.environ:
            world_size = int(os.environ["WORLD_SIZE"])
        else:
            error = "please set environmental variable WORLD_SIZE."
            STATUS.errors.append(error)
            LOG.error(error)
    STATUS.world_size = world_size
    LOG.info(f"WORLD_SIZE={world_size}")
    if world_size < 1:
        error = f"WORLD_SIZE must be more than 0, got {world_size}"
        STATUS.errors.append(error)
        LOG.error(error)
    if world_size > 1:
        msg = "Please check if all processes has the same master_addr, master_port, and world_size."
        STATUS.vulnarables.append(msg)
        LOG.info(msg)
        if backend == 'nccl':
            msg = "Please check if all processes has different cuda devices and different ranks."
            STATUS.vulnarables.append(msg)
            LOG.info(msg)
    
    if rank == -1:
        if "RANK" in os.environ:
            rank = int(os.environ["RANK"])
        else:
            error = "please set environmental variable RANK or set rank in init_process_group."
            STATUS.errors.append(error)
            LOG.error(error)
    STATUS.rank  = rank
    LOG.info(f"RANK={rank}")
    if rank < 0 or rank >= world_size:
        LOG.error("RANK must be no less than 0 and no more than WORLD_SIZE")

    torch.distributed.init_process_group(backend, init_method, timeout, world_size, rank, store, group_name, pg_options)

    LOG.info("torch.distributed initialized.")
    STATUS.status_summary()

def print_summary():
    STATUS.status_summary()


__all__ = [
    "debug_test",
    "init_process_group",
    "DataLoader",
    "DistributedSampler",
    "DistributedDataParallel",
    "DataParallel"
    "loss",
    "print_summary",
]
