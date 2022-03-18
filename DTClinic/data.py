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

class DistributedSampler(torch.utils.data.distributed.DistributedSampler):
    """
    Arguments:
        dataset: Dataset used for sampling.
        num_replicas (int, optional): Number of processes participating in distributed 
            training. By default, world_size is retrieved from the current distributed group.
        rank (int, optional): Rank of the current process within num_replicas. By 
            default, rank is retrieved from the current distributed group.
        shuffle (bool, optional): If True (default), sampler will shuffle the indices.
        seed (int, optional): random seed used to shuffle the sampler if shuffle=True. 
            This number should be identical across all processes in the distributed group. 
            Default: 0.
        drop_last (bool, optional): if True, then the sampler will drop the tail of 
            the data to make it evenly divisible across the number of replicas. If False, the sampler will add extra indices to make the data evenly divisible across the replicas. Default: False.
    """
    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True, 
        seed=0, drop_last=False):
        if STATUS.backend is None:
            error = "init_process_group is not successfully called before calling DistributedDataParallel."
            STATUS.errors.append(error)
            LOG.error(error)
            LOG.info("trying to call init_process_group and use nccl by default")
            DTClinic.init_process_group("nccl")
        STATUS.drop_last = drop_last

        if num_replicas is not None and num_replicas != STATUS.world_size:
            warning = f"num_replicas for DistributedSampler is not consistence with WORLD_SIZE. Expected {STATUS.WORLD_SIZE}, got {num_replicas}."
            STATUS.warnings.append(warning)
            LOG.warning(warning)

        if rank is not None and rank != STATUS.rank:
            warning = f"rank for DistributedSampler is not consistence with RANK. Expected {STATUS.RANK}, got {rank}."
            STATUS.warnings.append(warning)
            LOG.warning(warning)

        super().__init__(dataset, num_replicas=num_replicas, rank=rank,
            shuffle=shuffle, seed=seed, drop_last=drop_last)

class DataLoader(torch.utils.data.DataLoader):
    """
    This class is a PyTorch DataLoader. Applications can typically use objects
    of this class as direct replacements for PyTorch DataLoaders. 
    Arguments:
        dataset (torch.util.data.Dataset): Dataset from which to load the data.
        batch_size (int): The target total batch size across all replicas. The
            actual total batch size may be different due to rounding (each
            replica must have the same local batch size), or being scaled up
            using adaptive batch sizes.
        shuffle (bool): Whether the data is reshuffled at every epoch.
        ...
        **kwargs: Keyword arguments passed to ``torch.util.data.Dataloader``.
    Other arguments not supported for the moment. batch_sampler not well supported.
    """
    def __init__(self, dataset, batch_size=1, shuffle=False, sampler=None,
           batch_sampler=None, num_workers=0, collate_fn=None,
           pin_memory=True, drop_last=False, timeout=0,
           worker_init_fn=None, prefetch_factor=2,
           persistent_workers=True):
        LOG.info("Set pin_memory to True to speed up loading data. To stop using pinned memory, set pin_memory=False in DataLoader")
        LOG.info("Set persistent_workers to True to speed up loading data. To skill worker processes timely, set persistent_workers=False in DataLoader")
        STATUS.batch_size = batch_size
        # batch size need to be the same in different processes
        msg = "Check if batch sizes in different processes are the same."
        STATUS.vulnarables.append(msg)
        LOG.info(msg)
        if STATUS.world_size > 1 and not drop_last:
            warning = "drop_last is False, might have unmatched shape in the communication of the last iteration of each epoch."
            STATUS.warnings.append(warning)
            LOG.warning(warning)
        elif not STATUS.drop_last:
            STATUS.drop_last = drop_last
            if not STATUS.drop_last:
                warning = "drop_last is False, might have unmatched shape in the communication of the last iteration of each epoch."
                STATUS.warnings.append(warning)
                LOG.warning(warning)

        if STATUS.world_size > 1 and (sampler is None or not isinstance(sampler, DistributedSampler)):
            error = f"Expected DistributedSampler for the sampler. Got {type(sampler)}"
            STATUS.errors.append(error)
            LOG.error(error)
        
        super().__init__(dataset, batch_size=batch_size, shuffle=shuffle, sampler=sampler,
           batch_sampler=batch_sampler, num_workers=num_workers, collate_fn=collate_fn,
           pin_memory=pin_memory, drop_last=drop_last, timeout=timeout,
           worker_init_fn=worker_init_fn, prefetch_factor=prefetch_factor,
           persistent_workers=persistent_workers)
