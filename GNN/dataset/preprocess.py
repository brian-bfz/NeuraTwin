import h5py
from ..utils import fps_rad

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

def construct_edges_from_dataset(dataset):
    if isinstance(dataset, h5py.Dataset) and "object" in dataset.name:
        
        parent = dataset.parent


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
        
