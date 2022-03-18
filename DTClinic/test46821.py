import os
import numpy as np
import sys
import torch

from torch.utils.data import Dataset
from torch.utils.data.distributed import DistributedSampler
sys.path.append("..")
sys.path.append(".")
import DTClinic


class LRIterableDataset(Dataset):
    def __init__(self, size, true_values, noise):
        input_values = np.random.uniform(-5.0, 5.0, size)
        bias_input_values = np.stack([np.ones(size), input_values])
        target_values = (
            np.dot(true_values, bias_input_values)
            + np.random.normal(0.0, noise, size=(size,)))
        self._values = list(zip(input_values, target_values))
        self._len = size

    def __getitem__(self, index):
        return self._values[index]

    def __len__(self):
        return self._len


def test_issue46821():
    DTClinic.init_process_group("nccl")
    true_values = np.asarray([3.0, 4.0])
    dataset = LRIterableDataset(1000, true_values, 1.0)
    sampler = DTClinic.DistributedSampler(dataset, shuffle=True)
    dataloader = DTClinic.DataLoader(
        dataset, batch_size=32, num_workers=1, drop_last=False, sampler=sampler)
    model = torch.nn.Linear(1, 1, bias=True)
    device = 'cuda:' + os.environ['RANK']
    print('Gpu setting...', os.environ['RANK'])
    torch.cuda.set_device(int(os.environ['RANK']))
    #model = model.to(device)
    params = [model.bias, model.weight]
    sgd = torch.optim.SGD(
        [{"params": [param]} for param in params],
        lr=0.01)
    schedule = torch.optim.lr_scheduler.MultiStepLR(sgd, [50])
    model = DTClinic.DistributedDataParallel(
        model, device_ids=[int(os.environ['RANK'])], find_unused_parameters=True)
    #model =  torch.nn.parallel.DistributedDataParallel(
    #    model, device_ids=[int(os.environ['RANK'])], find_unused_parameters=True)
    loss = torch.nn.MSELoss()
    for epoch in range(0, 10):
        for inputs, targets in dataloader:
            inputs = inputs.float().to(device)
            targets = targets.float().to(device)
            sgd.zero_grad()
            output = model(torch.reshape(inputs, (-1, 1)))
            targets = torch.reshape(targets, (-1, 1))
            loss_value = DTClinic.loss(loss, output, targets)
            loss_value.backward()
            sgd.step()
        schedule.step()


DTClinic.debug_test(test_issue46821)
