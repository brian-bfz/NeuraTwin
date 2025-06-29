"""
Training utilities and helper classes.
"""

import random
import numpy as np
import torch
import sys
from torch.autograd import Variable
import torch.nn.functional as F


def collate_fn(data):
    """
    Custom collation function for batching variable-sized particle data.
    Pads sequences to maximum particle count in batch and handles temporal structure.
    """
    states, states_delta, attrs, particle_num, topological_edges, first_states = zip(*data)
    max_len = max(particle_num)
    batch_size = len(data)
    
    # Check if states is not empty to get shape
    if not states or states[0] is None:
        return None, None, None, None, None, None

    n_time, _, n_dim = states[0].shape
    
    states_tensor = torch.zeros((batch_size, n_time, max_len, n_dim), dtype=torch.float32)
    states_delta_tensor = torch.zeros((batch_size, n_time - 1, max_len, n_dim), dtype=torch.float32)
    attr = torch.zeros((batch_size, n_time, max_len), dtype=torch.float32)
    particle_num_tensor = torch.tensor(particle_num, dtype=torch.int32)
    topological_edges_tensor = torch.zeros((batch_size, max_len, max_len), dtype=torch.float32)
    first_states_tensor = torch.zeros((batch_size, max_len, n_dim), dtype=torch.float32)

    for i in range(len(data)):
        states_tensor[i, :, :particle_num[i], :] = states[i]
        states_delta_tensor[i, :, :particle_num[i], :] = states_delta[i]
        attr[i, :, :particle_num[i]] = attrs[i]
        topological_edges_tensor[i, :particle_num[i], :particle_num[i]] = topological_edges[i]
        first_states_tensor[i, :particle_num[i], :] = first_states[i]

    return states_tensor, states_delta_tensor, attr, particle_num_tensor, topological_edges_tensor, first_states_tensor


def compute_per_sample_mse_sum(s_pred, s_nxt, particle_nums):
    """
    Computes the sum of mean squared errors for each sample in a batch.
    This function is designed to replicate the behavior of a looped MSE calculation
    in a vectorized and efficient manner.
    """
    # Create a mask to select only the valid particles for each sample.
    max_particles = s_pred.shape[1]
    arange = torch.arange(max_particles, device=s_pred.device)
    # Unsqueeze to enable broadcasting over feature dimension.
    mask = (arange[None, :] < particle_nums[:, None]).unsqueeze(-1)

    # Compute MSE loss without reduction to get per-element squared errors.
    loss_matrix = F.mse_loss(s_pred, s_nxt, reduction='none')
    
    # Apply mask and sum the squared errors for each sample.
    sum_sq_err_per_sample = torch.sum(loss_matrix * mask, dim=(1, 2))
    
    # Normalize by the number of valid elements to get the mean for each sample.
    elements_per_sample = particle_nums * s_pred.shape[2]
    mse_per_sample = sum_sq_err_per_sample / (elements_per_sample + 1e-9)
    
    # Return the sum of per-sample MSEs, matching the original loop's behavior.
    return torch.sum(mse_per_sample)


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
# NOT USED
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