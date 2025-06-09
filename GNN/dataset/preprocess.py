import h5py
from ..utils import fps_rad, load_config
import numpy as np
import torch
import argparse


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


def construct_edges_from_dataset(name, points, adj_thresh, topk):
    """
    Construct topological edges for points in object dataset
    Saves edges to parent group

    Args:
        name: str - path to the dataset
        points: h5py.Dataset - object dataset, skip if not
        adj_thresh: float - radius of neighborhood
        topk: int - maximum number of neighbors
    """
    if isinstance(points, h5py.Dataset) and "object" in name:
        points_data = points[:]
        N = points_data.shape[0]
        
        # Compute pairwise squared distances
        diff = points_data[:, None, :] - points_data[None, :, :]
        distances_sq = np.sum(diff ** 2, axis=-1)
        
        # Threshold-based adjacency
        adj_matrix = (distances_sq < adj_thresh ** 2).astype(float)
        
        # Apply topk constraint
        topk = min(N, topk)
        topk_idx = np.argpartition(distances_sq, topk, axis=-1)[:, :topk]
        topk_matrix = np.zeros_like(adj_matrix)
        np.put_along_axis(topk_matrix, topk_idx, 1, axis=-1)
        adj_matrix = adj_matrix * topk_matrix
        
        # Save edges
        name = name.split('/')[-1]
        edge_name = f'{name}_edges'
        if edge_name in points.parent:
            del points.parent[edge_name]
        points.parent.create_dataset(edge_name, data=adj_matrix)


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
        f.visititems(lambda name, obj: construct_edges_from_dataset(name, obj, adj_thresh, topk))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_file', type=str, required=True)
    parser.add_argument('--output_file', type=str, required=True)
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Sample points and create output file
    sample_points(args.data_file, args.output_file, config)

    # Construct edges
    construct_edges_from_file(args.output_file, config)
