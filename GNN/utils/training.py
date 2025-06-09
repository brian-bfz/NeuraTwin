"""
Training utilities and helper classes.
"""

import random
import numpy as np
import torch
import sys
from torch.autograd import Variable


def set_seed(seed):
    """
    Set random seeds for reproducible training.
    
    Args:
        seed: int - random seed value
    """
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def count_trainable_parameters(model):
    """
    Count the number of trainable parameters in a PyTorch model.
    
    Args:
        model: torch.nn.Module - PyTorch model
        
    Returns:
        int - number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ============================================================================
# DEPRECATED
# ============================================================================

def rand_float(lo, hi):
    """
    Generate random float in range [lo, hi).
    
    Args:
        lo: float - lower bound
        hi: float - upper bound
        
    Returns:
        float - random value in range
    """
    return np.random.rand() * (hi - lo) + lo


def rand_int(lo, hi):
    """
    Generate random integer in range [lo, hi).
    
    Args:
        lo: int - lower bound
        hi: int - upper bound (exclusive)
        
    Returns:
        int - random integer in range
    """
    return np.random.randint(lo, hi)


def count_all_parameters(model):
    """
    Count the total number of parameters in a PyTorch model.
    
    Args:
        model: torch.nn.Module - PyTorch model
        
    Returns:
        int - total number of parameters
    """
    return sum(p.numel() for p in model.parameters())


def count_non_trainable_parameters(model):
    """
    Count the number of non-trainable parameters in a PyTorch model.
    
    Args:
        model: torch.nn.Module - PyTorch model
        
    Returns:
        int - number of non-trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if not p.requires_grad)


def to_var(tensor, use_gpu, requires_grad=False):
    """
    Convert tensor to Variable (legacy PyTorch).
    
    Args:
        tensor: numpy array - input tensor
        use_gpu: bool - whether to use GPU
        requires_grad: bool - whether to track gradients
        
    Returns:
        Variable - PyTorch variable
    """
    if use_gpu:
        return Variable(torch.FloatTensor(tensor).cuda(), requires_grad=requires_grad)
    else:
        return Variable(torch.FloatTensor(tensor), requires_grad=requires_grad)


def to_np(x):
    """
    Convert PyTorch tensor to numpy array.
    
    Args:
        x: torch.Tensor - input tensor
        
    Returns:
        numpy array - detached numpy array
    """
    return x.detach().cpu().numpy()


class AverageMeter(object):
    """
    Computes and stores the average and current value.
    """
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        """Reset all statistics."""
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """
        Update statistics with new value.
        
        Args:
            val: float - new value
            n: int - weight for the value (default: 1)
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count 

# ============================================================================
# DEPRECATED
# ============================================================================

def get_lr(optimizer):
    """
    Get current learning rate from optimizer.
    
    Args:
        optimizer: torch.optim.Optimizer - PyTorch optimizer
        
    Returns:
        float - current learning rate
    """
    for param_group in optimizer.param_groups:
        return param_group['lr']


def combine_stat(stat_0, stat_1):
    """
    Combine statistics from two datasets.
    
    Args:
        stat_0: numpy array - statistics [mean, std, count] for dataset 0
        stat_1: numpy array - statistics [mean, std, count] for dataset 1
        
    Returns:
        numpy array - combined statistics
    """
    mean_0, std_0, n_0 = stat_0[:, 0], stat_0[:, 1], stat_0[:, 2]
    mean_1, std_1, n_1 = stat_1[:, 0], stat_1[:, 1], stat_1[:, 2]

    mean = (mean_0 * n_0 + mean_1 * n_1) / (n_0 + n_1)
    std = np.sqrt(
        (std_0 ** 2 * n_0 + std_1 ** 2 * n_1 + (mean_0 - mean) ** 2 * n_0 + (mean_1 - mean) ** 2 * n_1) / (n_0 + n_1))
    n = n_0 + n_1

    return np.stack([mean, std, n], axis=-1)


def init_stat(dim):
    """
    Initialize statistics array.
    
    Args:
        dim: int - dimensionality
        
    Returns:
        numpy array - initialized statistics [mean=0, std=0, count=0]
    """
    return np.zeros((dim, 3))


class Tee(object):
    """
    Redirect stdout to both file and console.
    """
    def __init__(self, name, mode):
        self.file = open(name, mode)
        self.stdout = sys.stdout
        sys.stdout = self

    def __del__(self):
        sys.stdout = self.stdout
        self.file.close()

    def write(self, data):
        self.file.write(data)
        self.stdout.write(data)

    def flush(self):
        self.file.flush()

    def close(self):
        self.__del__()
