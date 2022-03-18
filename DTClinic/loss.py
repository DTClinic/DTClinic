import logging
import sys
import torch

sys.path.append("..")
sys.path.append(".")
import status

STATUS = status.STATUS

LOG_FORMAT = '{name}:{levelname} [{asctime}] {message}'
DATE_FORMAT = '%Y-%m-%d %H:%M:%S'

LOG = logging.getLogger(__name__)
LOG.setLevel(logging.INFO)

def loss(loss_function, *args):
    if STATUS.backend == 'nccl' or STATUS.multi_process == False:
        args = list(args)
        output = args[0]
        targets = args[1]
        if output.device != targets.device:
            if not STATUS.target_device_error:
                error = f"Targets must be on the same device with output. Got {targets.device} and {output.device}"
                STATUS.errors.append(error)
                LOG.error(error)
                STATUS.target_device_error = True
            args[1] = args[1].to(args[0].device)
        args = tuple(args)
    return loss_function(*args)
