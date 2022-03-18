# DTClinic

## Introduction

DTClinic  is a demo debugging tool for distributed training using PyTorch based on the findings in Rise of Distributed Training: From Developer’s Perspective. DTClinic is capable of detecting and fixing commonly encountered faults and recording logs and environmental information if it detects any vulnerability. According to the recorded logs and information, DTClinic provides debugging suggestions to developers as a supplement of logs from the original framework, which further helps developers to resolve their encountered issues.

## Usage
DTClinic provides out-of-box APIs for developers to debug their distributed training programs. The APIs are compatible with the APIs provided in PyTorch. To use DTClinic, developers need to (1) locate their code where uses the APIs that are specific to distributed training, and change to the corresponding APIs provided by Dist-
Debug; (2) wrap the code to test in a function. Here is an example:
```Python
import DTClinic

def test_function():
    # initialize process group
    DTClinic.init_process_group(backend="nccl", rank=args.rank, world_size=2)
    # set data sampler and train loader
    sampler = DTClinic.DistributedSampler(...)
    train_loader = DTClinic.DataLoader(...)
    # put the model to the correct device
    model = model.to(device)
    # wrap the model to a distributed data parallel model
    model = DTClinic.DistributedDataParallel(...)
    # train the model with PyTorch training code

DTClinic.debug_test(test_function)
```

## Implemented APIs
- init_process_groups
- DataLoader
- DistributedSampler
- DistributedDataParallel
- DataParallel
- loss


## Example
We provide examples for running DTClinic. For example, to reproduce the fault of PyTorch issue #46821 and fix it, run:
```bash
python test46821.py
```
