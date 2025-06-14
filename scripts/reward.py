import torch
import h5py
from .utils import chamfer_distance

class RewardFn:
    def __init__(self, n_sample, ap_weight):
        """
        load the target point cloud and other configs
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.ap_weight = ap_weight  # action penalty weight
        self.n_sample = n_sample 
        
        # Temporary path
        data_file = "PhysTwin/data/different_types/single_push_rope/data.h5"
            
        # Load first frame of object point cloud as target
        with h5py.File(data_file, 'r') as f:
            pcd = torch.tensor(f['episode_000000/object'][0], dtype=torch.float32, device=self.device) # [n_particles, 3]
        
        # Apply translation to target if specified
        target_translation = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32, device=self.device)  # Default translation
        target = pcd + target_translation

        # Move to device and shape to [n_sample, n_particles, 3]
        target = target.to(self.device)
        self.target = target.unsqueeze(0).repeat(self.n_sample, 1, 1)


    def __call__(self, state_seqs, action_seqs):
        """
        reward = -chamfer_distance - action_penalty

        Args: 
            state_seqs: [n_sample, n_look_ahead, n_particles * 3] torch tensor
            action_seqs: [n_sample, n_look_ahead, n_robot * 3] torch tensor

        Returns:
            Dict containing:
            - reward_seqs: [n_sample] torch tensor - total rewards
        """
        # Extract shape of state_seqs and action_seqs
        n_sample, n_look_ahead, action_dim = action_seqs.shape
        assert n_sample == self.n_sample
        n_robot = action_dim // 3

        # Extract final states and reshape to point clouds
        assert state_seqs.shape[0] == self.n_sample
        final_states = state_seqs[:, -1, :].to(self.device)  # [n_sample, n_particles * 3]
        final_pcd = final_states.reshape(self.n_sample, -1, 3)  # [n_sample, n_particles, 3]
        
        # Compute Chamfer distance penalty in batch
        chamfer_penalties = chamfer_distance(final_pcd, self.target)
        
        # Compute action magnitude penalty
        action_seqs = action_seqs.reshape(n_sample, n_look_ahead, n_robot, 3)
        
        # Compute speed (magnitude) of each robot point
        speeds = torch.norm(action_seqs, dim=3)  # [n_sample, n_look_ahead, n_robot]
        
        # For each timestep, take mean speed across robot points and sum across timesteps
        action_penalties = torch.mean(speeds, dim=2).sum(dim=1) * self.ap_weight


        print(f"chamfer_penalties: {chamfer_penalties}")
        print(f"action_penalties: {action_penalties}")
        return {
            'reward_seqs': -(chamfer_penalties + action_penalties)
        }
        
