import torch
import h5py
from .utils import chamfer_distance

class RewardFn:
    def __init__(self, ap_weight, fsp_weight, robot_mask):
        """
        load the target point cloud and other configs
        
        Args:
            ap_weight: float - action penalty weight
            robot_mask: [n_particles] boolean tensor - mask for robot particles (optional)
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.ap_weight = ap_weight  # action penalty weight
        self.fsp_weight = fsp_weight # final speed penalty weight
        self.robot_mask = robot_mask.to(self.device)
        
        # Temporary path
        data_file = "PhysTwin/generated_data/sampled_with_12_edges.h5"
            
        # Load first frame of object point cloud as target
        with h5py.File(data_file, 'r') as f:
            pcd = torch.tensor(f['episode_000000/object'][0], dtype=torch.float32, device=self.device) # [n_particles, 3]
        
        # Apply translation to target if specified
        target_translation = torch.tensor([-0.1, 0.0, 0.0], dtype=torch.float32, device=self.device)  # Default translation
        target = pcd + target_translation

        self.target = target.to(self.device)


    def __call__(self, state_seqs, action_seqs):
        """
        Goals:
            - Minimize chamfer distance between object and target throughout the trajectory
            - Minimize object speed in the last frame 
            - Minimize robot speed throughout the trajectory

        Args: 
            state_seqs: [n_sample, n_look_ahead, n_particles * 3] torch tensor
            action_seqs: [n_sample, n_look_ahead, 2] torch tensor - robot velocity

        Returns:
            Dict containing:
            - reward_seqs: [n_sample] torch tensor - total rewards
        """
        # Extract shape of state_seqs and action_seqs
        n_sample, n_look_ahead, _ = action_seqs.shape

        # Reshape to point clouds
        states_seqs = state_seqs.reshape(n_sample, n_look_ahead, -1, 3)  # [n_sample, n_look_ahead, n_particles, 3]
            
        target = self.target.unsqueeze(0).repeat(n_sample, 1, 1) # [n_sample, n_particles, 3]
        
        # Compute Chamfer distance penalty in batch (only between objects and target)
        chamfer_penalties = torch.zeros(n_sample, device=self.device)
        for i in range(n_look_ahead):
            chamfer_penalties += chamfer_distance(states_seqs[:, i, ~self.robot_mask, :], target) * i * i

        final_delta = states_seqs[:, -1, ~self.robot_mask, :] - states_seqs[:, -2, ~self.robot_mask, :]
        final_speed = torch.norm(final_delta, dim=2) # [n_sample, n_particles] 
        final_speed_penalty = torch.mean(final_speed, dim=1) * n_look_ahead * self.fsp_weight # [n_sample]
        
        # Energy = \int v \cdot dv
        acceleration = action_seqs[:, 1:] - action_seqs[:, :-1]
        energy = torch.sum(acceleration * action_seqs[:, :-1], dim=2)  # [n_sample, n_look_ahead]
        energy = torch.sum(torch.abs(energy), dim=1)
        action_penalties = energy * self.ap_weight  # [n_sample]

        print(f"chamfer_penalties: {chamfer_penalties}")
        print(f"final_speed_penalty: {final_speed_penalty}")
        print(f"action_penalties: {action_penalties}")
        return {
            'reward_seqs': -(chamfer_penalties + final_speed_penalty + action_penalties)
        }
        
