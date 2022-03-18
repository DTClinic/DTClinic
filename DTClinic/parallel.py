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

# todo: dataparallel

class DistributedDataParallel(torch.nn.parallel.DistributedDataParallel):
    def __init__(self, module, device_ids=None, output_device=None, dim=0, 
        broadcast_buffers=True, process_group=None, bucket_cap_mb=25, 
        find_unused_parameters=True, check_reduction=False, 
        gradient_as_bucket_view=False, device_check=True):
        self.device_check = device_check
        STATUS.multi_process = True
        if STATUS.backend is None:
            error = "init_process_group is not successfully called before calling DistributedDataParallel."
            STATUS.errors.append(error)
            LOG.error(error)
            LOG.info("trying to call init_process_group and use nccl by default")
            DTClinic.init_process_group("nccl")
        if STATUS.backend != 'nccl':
            msg = "NCCL backend recommended for DDP."
            STATUS.vulnarables.append(msg)
            LOG.info(msg)
        LOG.info("Set find_unused_parameters to True to avoid communication errors. To stop finding unused paprameters, set find_unused_parameters=False in DistributedDataParallel")
        if not isinstance(module, torch.nn.Module):
            error = f"The Module passed to DistributedDataParallel is not torch.nn.Module. Got {type(module)}"
            STATUS.errors.append(error)
            LOG.error(error)
        devices = list()
        state_dict = module.state_dict()
        on_cpu = False
        if device_check:
        	if STATUS.backend == 'nccl':
		        for param_name in state_dict:
		        	if not state_dict[param_name].is_cuda:
		        		if state_dict[param_name].device not in devices:
		        			devices.append(state_dict[param_name].device)
		        			error = f'model not on CUDA device, got {state_dict[param_name].device}.'
		        			STATUS.errors.append(error)
		        			LOG.error(error)
		        			on_cpu = True
		        	else:
		        		if state_dict[param_name].device not in devices:
		        			devices.append(state_dict[param_name].device)
		        if len(devices) == 1 and on_cpu:
		        	msg = f"Putting model on cuda:{STATUS.rank % torch.cuda.device_count()} by default. To disable device check, set device_check to False."
		        	LOG.info(msg)
		        	module = module.to("cuda:" + str(STATUS.rank % torch.cuda.device_count()))
		        	for param_name in module.state_dict():
		        		devices = [module.state_dict()[param_name].device]
		        		break
		        STATUS.devices = devices
		        LOG.info(f'DistributedDataParallel model on GPU devices {devices}.')
        
	        if len(devices) > 1:
	        	# then, For multi-device modules and CPU modules, device_ids must be None
	        	if device_ids is not None:
	        		error = f"For multi-device modules (and CPU modules), device_ids must be None, got{device_ids}"
	        		LOG.error(error)
	        		msg = "Using None for device_ids in DistributedDataParallel by default."
	        		LOG.info(msg)
	        		STATUS.vulnarables.append(msg)
	        		device_ids = None
            torch.cuda.set_device(devices[0])
        super().__init__(module, device_ids=device_ids, output_device=output_device,
        dim=dim, broadcast_buffers=broadcast_buffers, process_group=process_group, 
        bucket_cap_mb=bucket_cap_mb, find_unused_parameters=find_unused_parameters, 
        check_reduction=check_reduction, gradient_as_bucket_view=gradient_as_bucket_view)

    def forward(self, *args, **kwargs):
        if self.device_check and STATUS.backend == 'nccl':
            args = list(args)
            idx = 0
            for arg in args:
                if arg.device != STATUS.devices[0]:
                    error = f"Inputs must be on the same device with model. Got {arg.device} and {STATUS.devices[0]}"
                    STATUS.errors.append(error)
                    LOG.error(error)
                    args[idx] = args[idx].to(STATUS.devices[0])
                    idx += 1
            args = tuple(args)
        return super().forward(*args, **kwargs)


class DataParallel(torch.nn.DataParallel):
    def __init__(self, module, device_ids=None, output_device=None, dim=0):
        STATUS.multi_process = False
        msg = "It is recommended to use DistributedDataParallel instead of DataParallel."
        LOG.warning(msg)
        if device_ids is None:
            error = "DataParallel requires module to be on a CUDA device"
            LOG.error(error)
            STATUS.errors.append(error)
            msg = "Using all visible CUDA devices by default."
            LOG.info(msg)
            STATUS.vulnarables.append(msg)
            device_ids = list(range(torch.cuda.device_count()))
        STATUS.devices = [torch.device(f'cuda:{device_ids[id]}') for id in device_ids]
        self.default_device = STATUS.devices[0]
        state_dict = module.state_dict()
        device_error = False
        for key in state_dict:
            if state_dict[key].device != self.default_device:
                if not STATUS.module_device_error:
                    error = "DataParallel requires every input tensor be provided on the first device in its device_ids list."
                    STATUS.errors.append(error)
                    LOG.error(error)
                    STATUS.module_device_error = True
                device_error = True
                break
        if device_error:
            module = module.to(self.default_device)
            LOG.info(f"Putting module on {self.default_device} by default.")
        super().__init__(module, device_ids=device_ids, output_device=output_device, dim=dim)

    def forward(self, *args, **kwargs):
        # DataParallel requires every input tensor be provided on the first device in its device_ids list.
        args = list(args)
        idx = 0
        for arg in args:
            if arg.device != self.default_device:
                if not STATUS.input_device_error:
                    error = f"Inputs must be on the same device with model. Got {arg.device} and {self.default_device}"
                    STATUS.errors.append(error)
                    LOG.error(error)
                    STATUS.input_device_error = True
                args[idx] = args[idx].to(self.default_device)
                idx += 1
        args = tuple(args)
        return super().forward(*args, **kwargs)
