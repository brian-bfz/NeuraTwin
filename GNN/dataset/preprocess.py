import h5py
from ..utils import fps_rad
import numpy as np
import torch

def sample_points(data_file, output_file, config):
    """
    Sample the points in data_file using FPS
    Save sampled points to a hdf5 file

    Args: 
        data_file: str - path to HDF5 dataset file
        output_file: str - path to a new / empty HDF5 file
        config: dict - training configuration containing dataset parameters
    """

    # Extract dataset parameters
    n_episode = config['dataset']['n_episode']
    fps_radius = config['train']['particle']['fps_radius']

    # Load data
    with h5py.File(data_file, 'r') as f:
        with h5py.File(output_file, 'w') as f_out:
            for i in range(n_episode):
                episode_group = f[f'episode_{i:06d}']
                object_data = episode_group['object'] # [timesteps, n_obj, 3] numpy array
                robot_data = episode_group['robot'] # [timesteps, n_bot, 3] numpy array

                # Sample points using FPS on first frame
                sampled_object = fps_rad(object_data[0], fps_radius)
                sampled_robot = fps_rad(robot_data[0], fps_radius)

                # Save sampled points
                episode_group = f_out.create_group(f'episode_{i:06d}')
                episode_group.create_dataset(f'object', data=sampled_object)
                episode_group.create_dataset(f'robot', data=sampled_robot)

def construct_edges_from_dataset(points, adj_thresh, topk):
    """
    Construct topological edges for points in object dataset

    Args:
        points: h5py.Dataset - object dataset, skip if not
        adj_thresh: float - radius of neighborhood
        topk: int - maximum number of neighbors

    Returns:
        adj_matrix: [n_points, n_points] numpy array - adjacency matrix
    """
    # TODO: do this in numpy entirely
    if isinstance(points, h5py.Dataset) and "object" in points.name:
        # Convert points to torch tensor
        points = torch.from_numpy(points[:]).float()
        N, state_dim = points.shape

        # Create pairwise particle combinations
        s_receiv = points[:, None, :].repeat(1, N, 1)
        s_sender = points[None, :, :].repeat(N, 1, 1)

        # Create adjacency matrix based on distance threshold
        threshold = adj_thresh * adj_thresh
        s_diff = s_receiv - s_sender  # Position differences
        dis = torch.sum(s_diff ** 2, -1)  # Squared distances (N, N)
        adj_matrix = ((dis - threshold) < 0).float()

        # Apply topk constraint
        topk = min(dis.shape[-1], topk)
        topk_idx = torch.topk(dis, k=topk, dim=-1, largest=False)[1]
        topk_matrix = torch.zeros_like(adj_matrix)
        topk_matrix.scatter_(-1, topk_idx, 1)
        adj_matrix = adj_matrix * topk_matrix

        # Convert to numpy array 
        adj_matrix = adj_matrix.numpy()

        # Save to parent group
        name = points.name.split('/')[-1]
        points.parent.create_dataset(f'{name}_edges', data=adj_matrix)


def construct_edges_from_file(data_file, config):
    """
    Construct topological edges for points in data_file
    Only construct edges between points in the same dataset
    Save edges to the same file 

    Args:
        data_file: str - path to HDF5 dataset file
        config: dict - training configuration containing dataset parameters
    """

    # Extract dataset parameters
    adj_thresh = config['train']['edges']['topological']['adj_thresh']
    topk = config['train']['edges']['topological']['topk']

    # Load data
    with h5py.File(data_file, 'r+') as f:
        # iterate over all groups in file 
        f.visititems(lambda name, obj: construct_edges_from_dataset(obj, adj_thresh, topk))