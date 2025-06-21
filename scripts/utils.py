import torch
import h5py
import os
import numpy as np
import random
import math
from GNN.utils import fps_rad_tensor

def parse_episodes(episodes_arg):
    """
    Parse episode specification supporting both list and range formats.
    
    Args:
        episodes_arg: list[str] - episode specification from command line
        
    Returns:
        list[int] - parsed episode numbers
        
    Examples:
        ['0', '1', '2', '3', '4'] -> [0, 1, 2, 3, 4]
        ['0-4'] -> [0, 1, 2, 3, 4] (inclusive range)
    """
    if len(episodes_arg) == 1 and '-' in episodes_arg[0]:
        # Range format: "0-4"
        try:
            start, end = episodes_arg[0].split('-')
            start, end = int(start), int(end)
            if start > end:
                raise ValueError(f"Invalid range: start ({start}) > end ({end})")
            episodes = list(range(start, end + 1))  # inclusive
            print(f"Using episode range: {start} to {end} (inclusive) -> {episodes}")
            return episodes
        except ValueError as e:
            raise ValueError(f"Invalid range format '{episodes_arg[0]}': {e}")
    else:
        # Space-separated list: ["0", "1", "2", "3", "4"]
        try:
            episodes = [int(ep) for ep in episodes_arg]
            print(f"Using explicit episode list: {episodes}")
            return episodes
        except ValueError as e:
            raise ValueError(f"Invalid episode numbers: {e}")


def load_mpc_data(episode_idx, data_file, device):
    """
    Load MPC data from the specified episode.
    
    Args:
        episode_idx: int - episode to load
        data_file: str - path to data file
        device: torch.device - device to load tensors to
        
    Returns:
        tuple: (first_states, robot_mask, topological_edges) where
            first_states: [n_particles, 3] - first frame particle positions
            robot_mask: [n_particles] - boolean tensor for robot particles
            topological_edges: [n_particles, n_particles] or None - topological edges if available
    """
    with h5py.File(data_file, 'r') as f:
        # Read the first frame
        episode_group = f[f'episode_{episode_idx:06d}']
        object_data = episode_group['object'][0]
        robot_data = episode_group['robot'][0]
        n_object = object_data.shape[0]
        n_robot = robot_data.shape[0]

        # Convert to tensors
        object_data = torch.tensor(object_data, dtype=torch.float32, device=device)
        robot_data = torch.tensor(robot_data, dtype=torch.float32, device=device)
        first_states = torch.cat([object_data, robot_data], dim=0)
        
        robot_mask = torch.cat([torch.zeros(n_object, dtype=torch.bool, device=device), 
                              torch.ones(n_robot, dtype=torch.bool, device=device)], dim=0)

        # Check if topological edges exist
        topological_edges = None
        if 'object_edges' in episode_group:
            object_edges = episode_group['object_edges'][:]
            n_particles = n_object + n_robot
            topological_edges = torch.zeros(n_particles, n_particles, dtype=torch.float32, device=device)
            topological_edges[:n_object, :n_object] = torch.tensor(object_edges, dtype=torch.float32, device=device)

        return first_states, robot_mask, topological_edges
        
def load_mpc_data2(data_file, device):
    """
    Load a random episode from the specified data file using load_mpc_data
    Produce a target point cloud by downsampling the object point cloud with fps_rad_tensor
    Apply a random rotation to the target point cloud
    translate the target point cloud so that it's on the other side of the object from the robot

    Args:
        data_file: str - path to data file
        device: torch.device - device to load tensors to
        
    Returns:
        tuple: (first_states, robot_mask, target) where
            first_states: [n_particles, 3] - first frame particle positions
            robot_mask: [n_particles] - boolean tensor for robot particles
            target: [n_downsampled, 3] - target point cloud
    """
    # Determine the number of episodes available
    with h5py.File(data_file, 'r') as f:
        episode_keys = [k for k in f.keys() if k.startswith('episode_')]
        n_episodes = len(episode_keys)
    
    if n_episodes == 0:
        raise ValueError(f"No episodes found in {data_file}")
    
    # Select a random episode
    random_episode_idx = random.randint(0, n_episodes - 1)
    
    # Load the episode data using load_mpc_data
    first_states, robot_mask, topological_edges = load_mpc_data(random_episode_idx, data_file, device)
    
    # Separate object and robot particles
    object_particles = first_states[~robot_mask]  # [n_object, 3]
    robot_particles = first_states[robot_mask]    # [n_robot, 3]
    
    # Downsample the object point cloud using fps_rad_tensor
    # Use a reasonable radius for downsampling (0.03 is used in other parts of the codebase)
    downsampled_indices = fps_rad_tensor(object_particles, radius=0.03)
    target = object_particles[downsampled_indices]  # [n_downsampled, 3]
    
    # Apply random rotation to the target point cloud
    # Generate random rotation angle around z-axis (similar to dataset augmentation)
    angle = random.random() * 2 * math.pi
    cos_angle = math.cos(angle)
    sin_angle = math.sin(angle)
    
    # Create 3D rotation matrix for rotation around z-axis
    rotation_matrix = torch.tensor([
        [cos_angle, -sin_angle, 0],
        [sin_angle,  cos_angle, 0],
        [0,          0,         1]
    ], dtype=torch.float32, device=device)
    
    # Apply rotation: [n_downsampled, 3] @ [3, 3] -> [n_downsampled, 3]
    target = torch.matmul(target, rotation_matrix.T)
    
    # Position target on the opposite side of the object from the robot
    # Calculate centroids
    object_centroid = object_particles.mean(dim=0)  # [3]
    robot_centroid = robot_particles.mean(dim=0)    # [3]
    
    # Calculate vector from object centroid to robot centroid
    object_to_robot_vector = robot_centroid - object_centroid  # [3]
    
    # Calculate target centroid
    target_centroid = target.mean(dim=0)  # [3]
    
    # Position target on the opposite side by placing it at:
    # object_centroid - object_to_robot_vector (opposite direction from robot)
    # Add some additional distance to ensure separation
    separation_factor = 1.5  # Increase distance for better separation
    desired_target_centroid = object_centroid - object_to_robot_vector * separation_factor
    
    # Translate target to desired position
    translation = desired_target_centroid - target_centroid
    # Add noise
    translation = translation + torch.randn_like(translation)
    target = target + translation
    
    return first_states, robot_mask, target


def setup_task_directory(dir_name, mpc_config_path, device, model_type):
    """
    Set up task directory with target and config files.
    
    Args:
        dir_name: str - subdirectory name within tasks folder
        mpc_config_path: str - path to current MPC config
        device: torch device
        model_type: str - "GNN" or "PhysTwin" to determine base directory
        
    Returns:
        tuple: (save_dir, target_pcd) where save_dir is the full path and target_pcd is loaded/created target
    """
    from scripts.reward import create_default_target
    
    # Create task subdirectory in model-specific folder
    save_dir = os.path.join(f"{model_type}/tasks", dir_name)
    os.makedirs(save_dir, exist_ok=True)
    
    # Check for existing target
    target_path = os.path.join(save_dir, "target.npz")
    if os.path.exists(target_path):
        print(f"Loading existing target from: {target_path}")
        target_data = np.load(target_path)
        target_pcd = torch.tensor(target_data['target'], dtype=torch.float32, device=device)
    else:
        print(f"Creating new target for: {dir_name}")
        # target_pcd = create_default_target(np.array([-0.1, 0.0, 0.0]))
        target_pcd = np.load("targets/2.npz")['points']
        np.savez(target_path, target=target_pcd)
        target_pcd = torch.tensor(target_pcd, dtype=torch.float32, device=device)
        print(f"Target saved to: {target_path}")
    
    # Check for existing config
    config_path = os.path.join(save_dir, "mpc_config.yaml")
    if not os.path.exists(config_path):
        import shutil
        if os.path.exists(mpc_config_path):
            shutil.copy2(mpc_config_path, config_path)
            print(f"MPC config copied to: {config_path}")
        else:
            raise FileNotFoundError(f"MPC config file not found: {mpc_config_path}")
    
    return save_dir, target_pcd


def chamfer_distance(points, target):
    """
    Compute mean Chamfer distance between batch of point clouds and target point cloud

    Args: 
        points: [n_sample, n_particles, 3] torch tensor
        target: [n_sample, n_particles, 3] torch tensor

    Returns:
        [n_sample] torch tensor - Chamfer distance (scalar)
    """
    dist_matrix = torch.cdist(points, target, p=2) # [n_sample, n_particles, n_particles]
    
    # For each point in points, find minimum distance to any target point
    min_dist_points_to_target = torch.min(dist_matrix, dim=2)[0]  # [n_sample, n_particles]
    
    # For each point in target, find minimum distance to any points point
    min_dist_target_to_points = torch.min(dist_matrix, dim=1)[0]  # [n_sample, n_particles]
    
    # Chamfer distance is the mean of both directions
    chamfer_dist = (torch.mean(min_dist_points_to_target, dim=1) + 
                   torch.mean(min_dist_target_to_points, dim=1)) / 2.0
    
    return chamfer_dist
