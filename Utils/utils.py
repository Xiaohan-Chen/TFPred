import logging
import os
import time
import torch
import pickle
import numpy as np

def save_log(obj, path):
    with open(path, "wb") as f:
        pickle.dump(obj, f)

def read_pkl(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data

def save_model(model, args):
    if not os.path.exists("./checkpoints"):
        os.makedirs("./checkpoints")
    if not args.fft:
        torch.save(model.state_dict(), "./checkpoints/{}.tar".format(args.backbone))
    else:
        torch.save(model.state_dict(), "./checkpoints/{}FFT.tar".format(args.backbone))

def accuracy(outputs, targets):
    """
    Compute the labeled data accuracy.  

    Args:
    - outputs (tensor): model outputs
    - targets (tensor): ground truth

    Return:
    - accuracy (float)
    """
    labeled_minibatch_size = max(targets.ne(-1).sum(), 1e-8)

    pre = torch.max(outputs.cpu(), 1)[1].numpy()
    y = targets.data.cpu().numpy()

    acc = ((pre == y).sum() / labeled_minibatch_size) * 100
    return acc

def randomseed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

class Meter(object):
    def __init__(self):
        self.meter = {}
    
    def update(self, key, value):
        if key in self.meter:
            self.meter[key].append(value)
        else:
            self.meter[key] = [value]
    
    def reset(self):
        self.meter = {}

class AverageMeter(object):
    def __init__(self, name) -> None:
        self.name = name
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.history = []
    
    def update(self, val, n=1):
        self.sum += val * n
        self.count += n
        self.avg = self.sum /self.count
        self.history.append(val)