import h5py
from ..utils import fps_rad, load_yaml, construct_edges_from_numpy
import numpy as np
import argparse
import os
from ..paths import CONFIG_TRAIN_GNN_DYN


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

                # Get original episode metadata
                n_frames = episode_group.attrs['n_frames']
                n_obj_particles = episode_group.attrs['n_obj_particles']
                n_bot_particles = episode_group.attrs['n_bot_particles']

                # Sample points using FPS on first frame to get indices
                object_indices = fps_rad(object_data[0], fps_radius)
                robot_indices = fps_rad(robot_data[0], fps_radius)
                
                # Extract full trajectories for sampled particles
                sampled_object_trajectory = object_data[:, object_indices, :]  # [timesteps, n_sampled_obj, 3]
                sampled_robot_trajectory = robot_data[:, robot_indices, :]     # [timesteps, n_sampled_robot, 3]

                # Create output episode group and save metadata
                episode_group_out = f_out.create_group(f'episode_{i:06d}')
                episode_group_out.attrs['n_frames'] = n_frames
                episode_group_out.attrs['n_obj_particles'] = len(object_indices) 
                episode_group_out.attrs['n_bot_particles'] = len(robot_indices)

                # Save full sampled trajectories
                episode_group_out.create_dataset('object', data=sampled_object_trajectory)
                episode_group_out.create_dataset('robot', data=sampled_robot_trajectory)


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

    # First, collect all object datasets that need edge construction
    object_datasets = []
    with h5py.File(data_file, 'r') as f:
        def collect_object_datasets(name, obj):
            if isinstance(obj, h5py.Dataset) and "object" in name:
                object_datasets.append(name)
        f.visititems(collect_object_datasets)

    # Now process each dataset and create edges
    with h5py.File(data_file, 'r+') as f:
        for dataset_name in object_datasets:
            points = f[dataset_name]
            # Use first frame for computing topological edges
            points_data = points[0, :, :]  # [n_sampled_obj, 3] - first frame positions
            adj_matrix = construct_edges_from_numpy(points_data, adj_thresh, topk)
            
            # Save edges
            name = dataset_name.split('/')[-1]
            edge_name = f'{name}_edges'
            if edge_name in points.parent:
                del points.parent[edge_name]
            points.parent.create_dataset(edge_name, data=adj_matrix)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default="PhysTwin/generated_data")
    parser.add_argument('--data_file', type=str, required=True)
    parser.add_argument('--output_file', type=str, default=None)
    parser.add_argument('--config', type=str, default=str(CONFIG_TRAIN_GNN_DYN))
    args = parser.parse_args()

    # Load config
    config = load_yaml(args.config)

    # Create data files paths
    args.data_file = args.data_dir + '/' + args.data_file
    if args.output_file is None:
        args.output_file = config['dataset']['file']
    else: 
        args.output_file = args.data_dir + '/' + args.output_file
    
    # Check that output file is empty
    if os.path.exists(args.output_file):
        raise ValueError(f"Output file {args.output_file} already exists")

    # Sample points and create output file
    sample_points(args.data_file, args.output_file, config)

    # Construct edges
    if config['train']['edges']['topological']['enabled']:
        construct_edges_from_file(args.output_file, config)
