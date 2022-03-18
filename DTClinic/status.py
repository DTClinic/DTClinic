import logging
import os

import torch

LOG_FORMAT = '{name}:{levelname} [{asctime}] {message}'
DATE_FORMAT = '%Y-%m-%d %H:%M:%S'

LOG = logging.getLogger(__name__)
LOG.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT,
                                            style='{'))
LOG.addHandler(ch)

class Status():
    def __init__(self):
        self.multi_process = None
        self.backend = None
        self.master_addr = None
        self.master_port = None
        self.rank = -1
        self.world_size = -1
        self.drop_last = False
        self.batch_size = -1
        self.devices = None
        self.vulnarables = list()
        self.warnings = list()
        self.errors = list()
        self.module_device_error = False
        self.input_device_error = False
        self.target_device_error = False
    def status_summary(self, print_vulnarables=True, print_warnings=True, print_errors=True):
        print("=========== summary ===========")
        print("- Versions:")
        print(f"Pytorch: {torch.__version__}")
        print(f"CUDA: {torch.version.cuda}")
        nccl_version = torch.cuda.nccl.version()
        print(f"NCCL: {nccl_version[0]}.{nccl_version[1]},{nccl_version[2]}")
        if torch.__version__ < '1.10' or nccl_version[1] < 10:
            print("Please update to the newest version of PyTorch if possible.")
        print("- Environmental variables:")
        for key in os.environ:
            print(f"{key}={os.environ[key]}")
            print("If set NCCL_SOCKET_IFNAME or GLOO_SOCKET_IFNAME, please make sure it is in 'if_config'.")
            print("If no need to use multi-threads, please set OMP_NUM_THREADS=1 to speed up computation.")
        if self.multi_process == True:
            if self.backend == 'nccl':
                print("Please export NCCL_DEBUG=INFO to check NCCL versions and settings.")
            print()
            print("If you encounter communication problems, please check dependency versions and environmental variables one by one.")
            print()
            print("- process group information:")
            LOG.info(f"Using backend {self.backend}.")
            LOG.info(f"MASTER_ADDR={self.master_addr}")
            LOG.info(f"MASTER_PORT={self.master_port}")
            LOG.info(f"WORLD_SIZE={self.world_size}")
            LOG.info(f"RANK={self.rank}")
            LOG.info(f"drop_set is {self.drop_last} in dataloader / sampler.")
            if self.devices is not None:
            	LOG.info(f'DistributedDataParallel model on GPU devices {self.devices}.')
        elif self.multi_process == False:
            # todo
            if self.devices is not None:
                LOG.info(f'DataParallel model on GPU devices {self.devices}.')
        print()
        if print_vulnarables:
        	for each_vulnarable in self.vulnarables:
        		LOG.info(each_vulnarable)
        print()
        if print_warnings:
        	for each_warning in self.warnings:
        		LOG.info(each_warning)
        print()
        if print_errors:
        	for each_error in self.errors:
        		LOG.error(each_error)

STATUS = Status()
_allowed_symbols = [
    'STATUS'
]