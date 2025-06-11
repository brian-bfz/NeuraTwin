"""
Input/output utilities for file handling and time formatting.
"""

import datetime
import yaml
import os
from torch.distributed import init_process_group, destroy_process_group
from torch.cuda import set_device

def ddp_setup(rank, world_size):
    """
    Setup distributed training environment.

    Args:
        rank: int - unique identifier for current process
        world_size: int - total number of processes
    """
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    set_device(rank)
    init_process_group(backend='nccl', rank=rank, world_size=world_size)


def YYYY_MM_DD_hh_mm_ss_ms():
    """
    Returns a string identifying the current:
    - year, month, day, hour, minute, second, microsecond

    Using this format: YYYY-MM-DD-hh-mm-ss-microsecond

    For example: 2018-04-07-19-02-50-123456

    Note: this function will always return strings of the same length.

    Returns:
        str - current time formatted as a string
    """
    now = datetime.datetime.now()
    string = "%0.4d-%0.2d-%0.2d-%0.2d-%0.2d-%0.2d-%0.6d" % (
        now.year, now.month, now.day, now.hour, now.minute, now.second, now.microsecond)
    return string


def load_yaml(filename):
    """
    Load YAML file and return its contents.
    
    Args:
        filename: str - path to YAML file
        
    Returns:
        dict - loaded YAML contents
    """
    return yaml.safe_load(open(filename, 'r'))


def save_yaml(data, filename):
    """
    Save data to YAML file.
    
    Args:
        data: dict - data to save
        filename: str - output file path
    """
    with open(filename, 'w') as outfile:
        yaml.dump(data, outfile, default_flow_style=False) 