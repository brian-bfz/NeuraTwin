import functools
import torch
from numpy import pi


def get_dist_info():
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()
    else:
        rank = 0
        world_size = 1
    return rank, world_size


def master_only(func):

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        rank, _ = get_dist_info()
        if rank == 0:
            return func(*args, **kwargs)

    return wrapper


def singleton(cls):
    _instance = {}

    @functools.wraps(cls)
    def inner(*args, **kwargs):
        if cls not in _instance:
            _instance[cls] = cls(*args, **kwargs)
        return _instance[cls]

    return inner

def random_direction(device):
    random_angle = torch.rand(1, device=device) * 2 * pi - pi
    random_direction = torch.tensor([
        torch.cos(random_angle),
        torch.sin(random_angle),
        torch.tensor(0.0, device=device)  # Assuming movement in the XY plane
    ], device=device)
    random_direction = random_direction / torch.norm(random_direction)
    return random_direction
