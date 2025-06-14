import torch

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
