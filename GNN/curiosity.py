import torch
from .utils import fps_rad_tensor, construct_edges_from_tensor
from .control import GNNPlannerWrapper, PhysTwinInGNN
from scripts.reward import RewardFn
from scripts.planner import Planner
import h5py
import os


class VarianceRewardFn:
    """
    Variance-based reward function for curiosity-driven exploration.
    Returns the variance across ensemble predictions as the reward signal.
    Higher variance indicates higher uncertainty, which is rewarded for exploration.
    """
    
    def __init__(self):
        """Initialize variance reward function."""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def __call__(self, state_seqs, action_seqs):
        """
        Calculate variance across ensemble predictions.
        
        Args:
            state_seqs: [n_ensemble, n_sample, n_look_ahead, state_dim] - ensemble predictions
            action_seqs: [n_sample, n_look_ahead, action_dim] - action sequences (unused)
            
        Returns:
            dict containing:
                reward_seqs: [n_sample] - variance-based rewards (higher variance = higher reward)
        """
        # Calculate variance across ensemble dimension
        # state_seqs: [n_ensemble, n_sample, n_look_ahead, state_dim]
        prediction_variance = torch.var(state_seqs, dim=0)  # [n_sample, n_look_ahead, state_dim]
        
        # Sum variance across time and state dimensions to get total uncertainty per sample
        total_variance = torch.sum(prediction_variance, dim=(1, 2))  # [n_sample]
        
        return {'reward_seqs': total_variance}


class EnsembleModelRolloutFn:
    """
    Wrapper around GNNModelRolloutFn that runs multiple forward passes with dropout
    to generate ensemble predictions for uncertainty estimation.
    """
    
    def __init__(self, base_rollout_fn, n_ensemble):
        """
        Initialize ensemble wrapper.
        
        Args:
            base_rollout_fn: GNNModelRolloutFn instance
            n_ensemble: number of ensemble forward passes
        """
        self.base_rollout_fn = base_rollout_fn
        self.n_ensemble = n_ensemble
        
    def __call__(self, state_cur, action_seqs):
        """
        Run ensemble predictions using MC dropout.
        
        Args:
            state_cur: [n_history, state_dim] - current state history
            action_seqs: [n_sample, n_look_ahead, action_dim] - action sequences
            
        Returns:
            dict containing:
                state_seqs: [n_ensemble, n_sample, n_look_ahead, state_dim] - ensemble predictions
        """
        # Store ensemble predictions
        all_predictions = []
        
        # Run multiple forward passes with different dropout masks
        for _ in range(self.n_ensemble):
            # Each call to base_rollout_fn will use a different dropout mask
            prediction = self.base_rollout_fn(state_cur, action_seqs)
            all_predictions.append(prediction['state_seqs'])  # [n_sample, n_look_ahead, state_dim]
        
        # Stack predictions: [n_ensemble, n_sample, n_look_ahead, state_dim]
        ensemble_predictions = torch.stack(all_predictions, dim=0)
                
        return {'state_seqs': ensemble_predictions}


class CuriosityPlanner(GNNPlannerWrapper):
    """
    Curiosity-driven planner that extends GNNPlannerWrapper to use MC dropout 
    for identifying actions with highest prediction uncertainty for exploration.
    Uses variance across ensemble predictions as the exploration signal.
    """
    
    def __init__(self, model_path, train_config_path, mpc_config_path, case_name, mode = "curiosity"):
        """
        Initialize the curiosity planner.
        
        Args:
            model_path: str - path to trained GNN model checkpoint
            train_config_path: str - path to training configuration file
            mpc_config_path: str - path to MPC configuration file
            phystwin_states: [n_particles, 3] - particle positions from PhysTwin
            phystwin_robot_mask: [n_particles] - boolean mask for robot particles from PhysTwin
            case_name: str - case name for PhysTwin initialization
        """
        # Initialize GNNPlannerWrapper with case_name
        super().__init__(model_path, train_config_path, mpc_config_path, case_name)

        # adding mpc mode because Yixuan made a great point that uncertainty is hard to measure
        self.mode = mode
        self.padding = self.mpc_config['padding']
        assert self.mode in ["curiosity", "mpc"], f"Invalid mode: {self.mode}"
        if self.mode == "curiosity":
            # Store exploration parameters
            self.n_ensemble = self.mpc_config['n_ensemble']
            print(f"Curiosity planner initialized with n_ensemble={self.n_ensemble}")

        self.phystwin = PhysTwinInGNN(case_name, self.downsample_rate, device=self.device)        
        
    def load_phystwin_data(self, phystwin_states, phystwin_robot_mask):
        self.phystwin_states = phystwin_states.to(self.device)
        self.phystwin_robot_mask = phystwin_robot_mask.to(self.device)
        
        # Convert PhysTwin data to GNN format
        self._convert_phystwin_to_gnn()
        
        print(f"Converted {len(self.phystwin_states)} PhysTwin particles to {len(self.object_indices) + len(self.robot_indices)} GNN particles")

    def _initialize_model(self, model_path):
        """Override to keep model in training mode for MC dropout."""
        super()._initialize_model(model_path)
        self.model.train()

    def _convert_phystwin_to_gnn(self):
        """
        Convert PhysTwin data format to GNN format using downsampling and edge construction.
        Separates object and robot particles, applies FPS downsampling, constructs topological edges.
        """
        # Extract parameters from train config
        fps_radius = self.train_config['train']['particle']['fps_radius']
        adj_thresh = self.train_config['train']['edges']['topological']['adj_thresh']
        topk = self.train_config['train']['edges']['topological']['topk']
        
        # Separate object and robot particles
        object_states = self.phystwin_states[~self.phystwin_robot_mask]  # [n_obj, 3]
        robot_states = self.phystwin_states[self.phystwin_robot_mask]  # [n_robot, 3]
        
        # Downsample using FPS
        self.object_indices = fps_rad_tensor(object_states, fps_radius)
        sampled_object_states = object_states[self.object_indices]  # [n_obj_sampled, 3]
            
        self.robot_indices = fps_rad_tensor(robot_states, fps_radius)
        self.robot_indices = self.robot_indices + object_states.shape[0]
        # sampled_robot_states = robot_states[robot_indices]  # [n_robot_sampled, 3]
        
        # Construct topological edges for objects only
        object_edges = construct_edges_from_tensor(
            sampled_object_states, adj_thresh, topk
        )
        
        # Concatenate object and robot states
        # self.first_states = torch.cat([sampled_object_states, sampled_robot_states], dim=0)  # [n_particles, 3]
        n_obj_sampled = len(self.object_indices)
        n_robot_sampled = len(self.robot_indices)
        n_particles = n_obj_sampled + n_robot_sampled
        
        # Create robot mask for concatenated states
        self.robot_mask = torch.zeros(n_particles, dtype=torch.bool, device=self.device)
        self.robot_mask[n_obj_sampled:] = True  # Mark robot particles
        
        # Create full topological edges matrix (only objects have edges)
        self.topological_edges = torch.zeros(n_particles, n_particles, dtype=torch.float32, device=self.device)
        if n_obj_sampled > 0:
            self.topological_edges[:n_obj_sampled, :n_obj_sampled] = object_edges

    def _create_ensemble_model_rollout_fn(self, robot_mask, **kwargs):
        """Create ensemble model rollout function wrapped around base GNNModelRolloutFn."""
        base_rollout_fn = super()._create_model_rollout_fn(robot_mask, **kwargs)
        return EnsembleModelRolloutFn(base_rollout_fn, self.n_ensemble)

    def explore(self, data_file, phystwin_states = None, target = None):
        """
        Find action sequence with highest prediction uncertainty using curiosity-driven exploration.
        Use PhysTwin to simulate the action sequence and save the results to the data_file.
        
        Args:
            data_file: str - file where the results will be saved

        Returns:
            next_states: [n_particles, 3] - states of object and robot after the action sequence
        """
        if phystwin_states is not None: # allow high-level code to provide final states from previous exploration
            assert phystwin_states.shape == self.phystwin_states.shape, "New PhysTwin states must have the same shape as the initial PhysTwin states"
            self.phystwin_states = phystwin_states.to(self.device)
        first_states = torch.cat([self.phystwin_states[self.object_indices], self.phystwin_states[self.robot_indices]], dim=0)

        # Get action sequence with highest uncertainty (curiosity) / highest reward (mpc)
        if self.mode == "curiosity":
            result = self.plan_action(first_states)
        elif self.mode == "mpc":
            assert target is not None, "Target must be provided for MPC mode"
            result = self.plan_action(first_states, target)
        act_seq = result['act_seq']
        
        # Append 5 frames with 0 robot movement, so the object comes to rest
        act_seq = torch.cat([act_seq, torch.zeros(self.padding, self.action_dim, device=self.device)], dim=0)
        print(act_seq)
        
        print(f"Exploring with action sequence shape: {act_seq.shape}")
        print(f"Best variance: {result['best_eval_output']['reward_seqs'].item():.6f}")
        
        # Use PhysTwin to simulate the action sequence
        print("Running PhysTwin simulation...")
        full_trajectory, _ = self.phystwin.compute_deformation(
            action_seq=act_seq,
            phystwin_states=self.phystwin_states,
            phystwin_robot_mask=self.phystwin_robot_mask
        )

        # Split into object and robot states (full_trajectory now includes both)
        object_states = full_trajectory[:, self.object_indices, :].cpu().numpy()
        robot_states = full_trajectory[:, self.robot_indices, :].cpu().numpy()
        
        # Get object edges from downsampled data
        n_obj_sampled = len(self.object_indices)
        object_edges = self.topological_edges[:n_obj_sampled, :n_obj_sampled].cpu().numpy()
        
        # Save the results to the data_file
        self._save_results(data_file, object_states, robot_states, object_edges)
        
        print(f"Exploration complete! Results saved to: {data_file}")

        return full_trajectory[-1]

    def _save_results(self, data_file, object_states, robot_states, object_edges):
        """
        Data format: h5py file with the following structure:
        - episode_{idx:06d}
            - object: [n_frames, n_obj_particles, 3]
            - robot: [n_frames, n_bot_particles, 3]
            - object_edges: [n_obj, n_obj]
            - attrs
                - n_frames: int
                - n_obj_particles: int
                - n_bot_particles: int
        - attrs
            - unused_idx: int - lowest unused index 
        
        If data_file exists, append to it and update unused_idx. Otherwise, create a new file.
        
        Args:
            data_file: str - path to h5py file
            object_states: [n_frames, n_obj_particles, 3] - object trajectory
            robot_states: [n_frames, n_robot_particles, 3] - robot trajectory
            object_edges: [n_obj_particles, n_obj_particles] - object adjacency matrix
        """
        n_frames, n_obj_particles, _ = object_states.shape
        n_robot_particles = robot_states.shape[1]
        
        # Determine next episode index
        if os.path.exists(data_file):
            with h5py.File(data_file, 'r') as f:
                if 'attrs' in f and 'unused_idx' in f.attrs:
                    next_idx = f.attrs['unused_idx']
                else:
                    # Find highest episode index
                    episode_keys = [k for k in f.keys() if k.startswith('episode_')]
                    if episode_keys:
                        indices = [int(k.split('_')[1]) for k in episode_keys]
                        next_idx = max(indices) + 1
                    else:
                        next_idx = 0
        else:
            next_idx = 0
        
        # Write/append to file
        with h5py.File(data_file, 'a') as f:
            episode_name = f'episode_{next_idx:06d}'
            
            # Create episode group
            if episode_name in f:
                del f[episode_name]  # Remove if exists
            episode_group = f.create_group(episode_name)
            
            # Save data
            episode_group.create_dataset('object', data=object_states)
            episode_group.create_dataset('robot', data=robot_states)
            episode_group.create_dataset('object_edges', data=object_edges)
            
            # Save episode attributes
            episode_group.attrs['n_frames'] = n_frames
            episode_group.attrs['n_obj_particles'] = n_obj_particles
            episode_group.attrs['n_bot_particles'] = n_robot_particles
            
            # Update global unused_idx
            f.attrs['unused_idx'] = next_idx + 1
        
        print(f"Saved episode {next_idx} to {data_file}")
        print(f"  Frames: {n_frames}")
        print(f"  Object particles: {n_obj_particles}")
        print(f"  Robot particles: {n_robot_particles}")


    def plan_action(self, first_states, target = None):
        """
        Override plan_action to use variance-based reward function instead of target-based reward.
                    
        Returns:
            dict containing planning results with highest uncertainty action sequence
        """
        # Prepare initial state for model
        state_cur = self._prepare_initial_state(first_states, self.robot_mask)
        
        # Set up model rollout function
        if self.mode == "curiosity":
            model_rollout_fn = self._create_ensemble_model_rollout_fn(self.robot_mask, first_states=first_states, topological_edges=self.topological_edges)
        elif self.mode == "mpc":
            model_rollout_fn = self._create_model_rollout_fn(self.robot_mask, first_states=first_states, topological_edges=self.topological_edges)
        
        initial_action_seq = torch.zeros(self.n_look_ahead, self.action_dim, device=self.device)
        
        if self.mode == "curiosity":
            reward_fn = VarianceRewardFn()
        elif self.mode == "mpc":
            reward_fn = RewardFn(self.action_weight, self.fsp_weight, self.robot_mask, target)
        
        # Set up planner
        planner = Planner({
            'action_dim': self.action_dim,
            'model_rollout_fn': model_rollout_fn,
            'evaluate_traj_fn': reward_fn,
            'n_sample': self.n_sample,
            'n_look_ahead': self.n_look_ahead,
            'n_update_iter': self.n_update_iter,
            'reward_weight': self.reward_weight,
            'action_lower_lim': self.action_lower_bound,
            'action_upper_lim': self.action_upper_bound,
            'planner_type': self.planner_type,
            'noise_level': self.noise_level,
            'beta': self.beta,
            'verbose': self.verbose,
            'device': self.device
        })
        
        # Plan action sequence
        import time
        start_time = time.time()
        result = planner.trajectory_optimization(state_cur, initial_action_seq)
        end_time = time.time()
        print(f"Time taken for curiosity-driven planning: {end_time - start_time:.6f} seconds")
        
        return result



