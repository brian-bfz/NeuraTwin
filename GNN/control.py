import torch
import argparse
import h5py

from .model.rollout import Rollout
from .model.gnn_dyn import PropNetDiffDenModel
from .utils import load_yaml
from .paths import get_model_paths
from scripts.planner import Planner
from scripts.reward import RewardFunction

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

        # Calculate delta history: [n_sample, n_history-1, n_particles, 3]
        states_delta = states[:, 1:] - states[:, :-1]

        # Create attributes (0=object, 1=robot) [n_sample, n_particles]
        attrs = self.robot_mask.clone().float().unsqueeze(0).repeat(n_sample, 1)
        
        # Reshape actions: [n_sample, n_look_ahead, n_robots*3] -> [n_sample, n_look_ahead, n_robots, 3]
        robot_actions = action_seqs.view(n_sample, n_look_ahead, self.n_robots, 3)
        
        particle_nums = torch.tensor([self.n_particles] * n_sample, device=self.device)

        rollout = Rollout(
            self.model, 
            self.config, 
            states[:, -1],          # [n_sample, n_particles, 3] - current state (last frame)
            states_delta,           # [n_sample, n_history-1, n_particles, 3] - delta history
            attrs,                  # [n_sample, n_particles] - attributes
            particle_nums,          # [n_sample] - particle counts
            self.topological_edges, # [n_sample, n_particles, n_particles] - topological edges
            self.first_states,      # [n_sample, n_particles, 3] - first frame states
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
    def __init__(self, model_path: str, train_config_path: str, mpc_config_path: str):
        """
        Initialize the configs, GNN model, and whatever else is needed for the planner.
        
        Args:
            model_path: str - path to trained GNN model checkpoint
            train_config_path: str - path to training configuration file
            mpc_config_path: str - path to MPC configuration file
        """
        # Load configurations
        self.train_config = load_yaml(train_config_path)
        self.mpc_config = load_yaml(mpc_config_path)
        
        # Setup device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load trained model
        self.model = PropNetDiffDenModel(self.train_config, torch.cuda.is_available())
        self.model.load_state_dict(torch.load(model_path, weights_only=True, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        
        # Extract episode-independent parameters
        self.n_history = self.train_config['train']['n_history'] # mpc and model n_history must agree
        self.data_file = self.mpc_config['data_file'] # mpc may use different data file from training
        self.n_sample = self.mpc_config['n_sample']
        self.n_look_ahead = self.mpc_config['n_look_ahead']
        self.n_update_iter = self.mpc_config['n_update_iter']
        self.reward_weight = self.mpc_config['reward_weight']
        self.planner_type = self.mpc_config['planner_type']
        self.noise_level = self.mpc_config['noise_level']
        self.action_lower_bound = self.mpc_config['action_lower_bound']
        self.action_upper_bound = self.mpc_config['action_upper_bound']
        self.verbose = self.mpc_config.get('verbose', False)
        
        print(f"Loaded model from: {model_path}")
        print(f"Using device: {self.device}")
        print(f"History length: {self.n_history}")

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
            # Read the first frame
            episode_group = f[f'episode_{episode_idx:06d}']
            object_data = episode_group['object'][0]
            robot_data = episode_group['robot'][0]
            object_edges = episode_group['object_edges']
            n_object = object_data.shape[0]
            n_robot = robot_data.shape[0]

            # Convert to tensors
            object_data = torch.tensor(object_data, dtype=torch.float32, device=self.device)
            robot_data = torch.tensor(robot_data, dtype=torch.float32, device=self.device)
            first_states = torch.cat([object_data, robot_data], dim=0)
            
            robot_mask = torch.cat([torch.zeros(n_object, dtype=torch.bool, device=self.device), torch.ones(n_robot, dtype=torch.bool, device=self.device)], dim=0)

            # Create topological edges matrix (objects only, robots don't have topological edges)
            n_particles = n_object + n_robot
            topological_edges = torch.zeros(n_particles, n_particles, dtype=torch.float32, device=self.device)
            topological_edges[:n_object, :n_object] = object_edges

            return first_states, topological_edges, robot_mask

    def plan_action(self, episode_idx):
        """
        Plan action sequence to go from the initial state to the goal state, defined by the reward function. 

        Args:
            episode_idx: int - episode containing the initial state

        Returns:
            action_seqs: [n_sample, n_look_ahead, n_robots*3] - action sequence
        """
        # Prepare data
        first_states, topological_edges, robot_mask = self.load_data(episode_idx)
        state_cur = first_states.clone().unsqueeze(0).repeat(self.n_history, 1, 1) # [n_history, n_particles, 3]
        state_cur = state_cur.flatten(start_dim=1) # [n_history, n_particles*3]

        # Set up the model rollout function
        model_rollout_fn = ModelRolloutFn(self.model, self.train_config, robot_mask, self.n_sample, topological_edges, first_states)
        n_robots = model_rollout_fn.n_robots
        action_dim = n_robots * 3
        initial_action_seq = torch.zeros(self.n_look_ahead, action_dim, device=self.device)

        # Set up the reward function (blackbox for now)
        reward_fn = RewardFunction({})

        # Set up the planner
        planner = Planner({
            'action_dim': action_dim,
            'model_rollout_fn': model_rollout_fn,
            'evaluate_traj_fn': reward_fn,
            'n_sample': self.n_sample,
            'n_look_ahead': self.n_look_ahead,
            'n_update_iter': self.n_update_iter,
            'reward_weight': self.reward_weight,
            'action_lower_lim': torch.full((action_dim,), self.action_lower_bound, device=self.device),
            'action_upper_lim': torch.full((action_dim,), self.action_upper_bound, device=self.device),
            'planner_type': self.planner_type,
            'noise_level': self.noise_level,
            'verbose': self.verbose,
            'device': self.device
        })

        # Plan action sequence
        result = planner.trajectory_optimization(state_cur, initial_action_seq)
        return result['act_seq']

    def visualize_action(self, episode_idx, action_seqs):
        """
        Use ModelRolloutFn to infer the state sequence from initial state and action seqeuence
        Use Open3D to visualize the state and action sequences, and target of reward function
        Save the video to ./tasks as an mp4 file with avc1 codec

        Args:
            episode_idx: int - episode containing the initial state
            action_seqs: [n_sample, n_look_ahead, n_robots*3] torch tensor 
        """
        
        

if __name__ == "__main__":
    """
    Demonstrate GNN-based model predictive control.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True,
                       help="Model name (e.g., 2025-05-31-21-01-09-427982)")
    parser.add_argument("--episode", type=int, required=True,
                       help="Episode index to load")
    parser.add_argument("--train_config", type=str, default=None,
                       help="Path to model config file")
    parser.add_argument("--mpc_config", type=str, default="GNN/config/mpc/config.yaml",
                       help="Path to MPC config file")
    args = parser.parse_args()
    
    # Setup paths
    model_paths = get_model_paths(args.model)
    model_path = str(model_paths['net_best'])
    
    # Use training config from model directory if not specified
    if args.train_config is None:
        train_config_path = str(model_paths['config'])
    else:
        train_config_path = args.train_config
        
    # Initialize planner wrapper
    planner_wrapper = PlannerWrapper(model_path, train_config_path, args.mpc_config)
    
    # Demonstrate planning
    print("="*60)
    print("GNN-BASED MODEL PREDICTIVE CONTROL DEMO")
    print("="*60)
    
    # Plan action sequence for the specified episode
    optimal_actions = planner_wrapper.plan_action(args.episode)
    print(f"Planned action sequence shape: {optimal_actions.shape}")
    print("Planning demonstration complete!")
