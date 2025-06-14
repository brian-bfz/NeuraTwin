import os
import torch
import argparse
from typing import Dict, Any

from .model.rollout import Rollout
from .model.gnn_dyn import PropNetDiffDenModel
from .utils import load_yaml
from .paths import get_model_paths
from scripts.planner import Planner
from scripts.reward import RewardFunction
import h5py


class ModelRolloutFn:
    def __init__(self, model, config, robot_mask, n_sample, topological_edges, first_states):
        """
        Initialize the model rollout function.
        
        Args:
            model: trained GNN model
            config: dict - training configuration
            robot_mask: [n_particles] - boolean tensor for robot particles
            n_sample: int - number of samples the planner generates
            topological_edges: [n_particles, n_particles] tensor - topological edges adjacency matrix
            first_states: [n_particles, 3] - first frame particle positions
        """
        self.model = model
        self.config = config
        self.robot_mask = robot_mask
        self.n_history = config['train']['n_history']
        self.n_robots = robot_mask.sum().item()
        self.n_particles = len(robot_mask)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.topological_edges = topological_edges.to(self.device).unsqueeze(0).repeat(n_sample)
        self.first_states = first_states.to(self.device).unsqueeze(0).repeat(n_sample)

    def __call__(self, state_cur, action_seqs):
        """
        Evaluate multiple action sequences using batched Rollout operations.
        
        Args:
            state_cur: [n_history, state_dim] - current state history 
            action_seqs: [n_sample, n_look_ahead, action_dim] - robot action sequences
            
        Returns:
            dict containing:
                state_seqs: [n_sample, n_look_ahead, state_dim] - predicted trajectories
        """
        n_sample, n_look_ahead, _ = action_seqs.shape

        # Reshape states: [n_history, n_particles*3]] -> [n_sample, n_history, n_particles, 3]
        states = state_cur.view(self.n_history, self.n_particles, 3)
        states = states.unsqueeze(0).repeat(n_sample, 1, 1, 1)

        # Calculate delta history
        states_delta = states[1:] - states[:-1]

        # Create attributes (0=object, 1=robot) [n_sample, n_particles]
        attrs = self.robot_mask.clone().float().unsqueeze(0).repeat(n_sample, 1)
        
        # Reshape actions: [n_sample, n_look_ahead, n_robots*3] -> [n_sample, n_look_ahead, n_robots, 3]
        robot_actions = action_seqs.view(n_sample, n_look_ahead, self.n_robots, 3)
        
        particle_nums = torch.tensor([self.n_particles] * n_sample, device=self.device)

        rollout = Rollout(
            self.model, 
            self.config, 
            states, 
            states_delta, 
            attrs, 
            particle_nums,
            self.topological_edges, 
            self.first_states, 
        )
        
        state_seqs = torch.zeros(n_sample, n_look_ahead, self.n_particles * 3, device=action_seqs.device)
        for step_idx in range(n_look_ahead):
            # Create delta tensor for all samples: [n_sample, particles, 3]
            next_delta = torch.zeros(n_sample, self.n_particles, 3, device=action_seqs.device)
            next_delta[:, self.robot_mask, :] = robot_actions[:, step_idx, :, :]
            
            # Predict next states for all samples simultaneously
            predicted_states = rollout.forward(next_delta)  # [n_sample, particles, 3]
            
            # Flatten states for planner: [n_sample, particles * 3]
            state_seqs[:, step_idx, :] = predicted_states.flatten(start_dim=1)           
        
        return {'state_seqs': state_seqs}
    

class PlannerWrapper: 
    def __init__():
        """
        """

    def load_data(self, episode_idx):
        """
        Load data from the specified episode.    

        Args: 
            episode_idx: int - episode to load

        Returns:
            first_states: [n_particles, 3] - first frame particle positions
            topological_edges: [n_particles, n_particles] - topological edges adjacency matrix
            robot_mask: [n_particles] - boolean tensor for robot particles
        """
        with h5py.File(self.data_file, 'r') as f:
            episode_group = f[f'episode_{episode_idx:06d}']
            object_data = episode_group['object'][0]
            robot_data = episode_group['robot'][0]
            n_object = object_data.shape[0]
            n_robot = robot_data.shape[0]
            first_states = torch.cat([object_data[0], robot_data[0]], dim=0)
            topological_edges = episode_group['topological_edges']
            robot_mask = torch.cat([torch.zeros(n_object), torch.ones(n_robot)], dim=0)
            return first_states, topological_edges, robot_mask

    def plan_action(self, episode_idx, RewardFn):
        """
        Plan action sequence to go from the initial state to the goal state. 

        Args:
            episode_idx: int - episode containing the initial state
            RewardFn: function - reward function describing the goal state

        Returns:
            action_seqs: [n_sample, n_look_ahead, n_robots*3] - action sequence
        """
        first_states, topological_edges, robot_mask = self.load_data(episode_idx)
        n_robots = robot_mask.sum().item()
        action_dim = n_robots * 3

        planner_config = {
            'action_dim': action_dim,
            'n_sample': self.mpc_config['n_sample'],
        }


def main():
    """
    Demonstrate GNN-based model predictive control.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True,
                       help="Model name (e.g., 2025-05-31-21-01-09-427982)")
    parser.add_argument("--episode", type=int, default=0,
                       help="Episode index to load")
    parser.add_argument("--config_path", type=str, default=None)
    parser.add_argument("--data_file", type=str, default=None)
    args = parser.parse_args()
    
    # Setup paths
    model_paths = get_model_paths(args.model)
    model_path = str(model_paths['net_best'])
    if args.config_path is None:
        config_path = str(model_paths['config'])
    else:
        config_path = args.config_path
        
    # Initialize controller
    controller = GNNController(model_path, config_path, args.data_file)
    
    # Load episode
    controller.load_episode(args.episode)
    
    # Setup planner with MPC config
    mpc_config = controller.config.get('mpc', {})
    n_robots = controller.robot_mask.sum().item()
    action_dim = n_robots * 3
    
    planner_config = {
        'action_dim': action_dim,
        'n_sample': mpc_config['n_sample'],
        'n_look_ahead': mpc_config['n_look_ahead'],
        'n_update_iter': mpc_config['n_update_iter'],
        'reward_weight': mpc_config['reward_weight'],
        'action_lower_lim': torch.tensor([-0.1] * action_dim, device=controller.device),
        'action_upper_lim': torch.tensor([0.1] * action_dim, device=controller.device),
        'planner_type': mpc_config['planner_type'],
        'noise_level': mpc_config['noise_level'],
        'verbose': True
    }
    
    controller.setup_planner(planner_config)
    
    # Demonstrate planning
    print("="*60)
    print("GNN-BASED MODEL PREDICTIVE CONTROL DEMO")
    print("="*60)
    
    # Plan action sequence
    optimal_actions = controller.plan_action()
    print(f"Planned action sequence shape: {optimal_actions.shape}")
    print("Planning demonstration complete!")


if __name__ == "__main__":
    main() 