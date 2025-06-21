import numpy as np
import torch
import torch.nn.functional as F
import time
import os
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from .reward import RewardFn

torch.autograd.set_detect_anomaly(True)
from GNN.utils import fps_rad, load_yaml


class PlannerWrapper(ABC):
    """
    Abstract base class for planner wrappers that provides common functionality.
    Contains shared features between different planner implementations.
    """
    
    def __init__(self, mpc_config_path: str):
        """
        Initialize common planner wrapper functionality.
        
        Args:
            mpc_config_path: str - path to MPC configuration file
        """
        
        # Load MPC configuration
        self.mpc_config = load_yaml(mpc_config_path)
        
        # Setup device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Extract common MPC parameters
        self.action_dim = self.mpc_config['action_dim']
        self.n_look_ahead = self.mpc_config['n_look_ahead']
        self.n_sample = self.mpc_config['n_sample']
        self.n_update_iter = self.mpc_config['n_update_iter']
        self.reward_weight = self.mpc_config['reward_weight']
        self.action_lower_bound = torch.full((self.action_dim,), self.mpc_config['action_lower_bound'], device=self.device)
        self.action_upper_bound = torch.full((self.action_dim,), self.mpc_config['action_upper_bound'], device=self.device)
        self.planner_type = self.mpc_config['planner_type']
        self.noise_level = self.mpc_config['noise_level']
        self.verbose = self.mpc_config.get('verbose', False)
        
        self.action_weight = self.mpc_config['action_weight']
        self.fsp_weight = self.mpc_config['fsp_weight']
        self.timestamp = time.strftime('%Y-%m-%d_%H-%M-%S')
        
        print(f"Using device: {self.device}")
    
    @abstractmethod
    def _initialize_model(self):
        """Initialize the specific model (GNN or PhysTwin)."""
        pass
    
    @abstractmethod
    def _create_model_rollout_fn(self, robot_mask, **kwargs):
        """Create model rollout function specific to the implementation."""
        pass
    
    def plan_action(self, target_pcd, first_states, robot_mask, topological_edges=None, episode_idx=None, save_dir=None):
        """
        Plan action sequence to go from the initial state to the goal state.
        
        Args:
            target_pcd: torch.Tensor - target point cloud for reward function
            first_states: torch.Tensor - initial states
            robot_mask: torch.Tensor - robot mask
            topological_edges: torch.Tensor - topological edges (GNN only)
            episode_idx: int - episode containing the initial state (optional)
            save_dir: str - directory to save results (optional)
            
        Returns:
            dict containing:
                act_seq: [n_look_ahead, action_dim] - action sequence
                eval_outputs: list of dicts containing evaluation outputs for each iteration
                best_model_output: dict containing best model output
        """
        # Prepare initial state for model
        state_cur = self._prepare_initial_state(first_states, robot_mask)
        
        # Set up model rollout function
        model_rollout_fn = self._create_model_rollout_fn(robot_mask, first_states=first_states, topological_edges=topological_edges)
        initial_action_seq = torch.zeros(self.n_look_ahead, self.action_dim, device=self.device)
        
        # Set up reward function with provided target
        reward_fn = RewardFn(self.action_weight, self.fsp_weight, robot_mask, target_pcd)
        
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
            'verbose': self.verbose,
            'device': self.device
        })
        
        # Plan action sequence
        start_time = time.time()
        result = planner.trajectory_optimization(state_cur, initial_action_seq)
        end_time = time.time()
        print(f"Time taken for planning: {end_time - start_time:.2f} seconds")
        
        self._save_planning_results(result, episode_idx, save_dir)
        
        # Plot rewards if verbose
        if self.verbose:
            self._plot_rewards(result['eval_outputs'], save_dir)
        
        # Print best evaluation output
        if 'best_eval_output' in result and result['best_eval_output'] is not None:
            print("Best evaluation output:")
            for key, value in result['best_eval_output'].items():
                if torch.is_tensor(value):
                    if value.numel() == 1:
                        print(f"  {key}: {value.item()}")
                    else:
                        print(f"  {key}: shape {value.shape}, mean={value.mean().item():.6f}")
                else:
                    print(f"  {key}: {value}")
        
        return result
    
    @abstractmethod
    def _prepare_initial_state(self, first_states, robot_mask):
        """Prepare initial state format specific to the model."""
        pass
    
    def _save_planning_results(self, result, episode_idx, save_dir):
        """Save planning results to specified directory."""
        os.makedirs(save_dir, exist_ok=True)
        
        # Save action sequence
        act_seq_path = os.path.join(save_dir, f"act_seq_episode_{episode_idx:06d}_{self.timestamp}.npz")
        np.savez(act_seq_path, act_seq=result['act_seq'].cpu().numpy())
        
        # Save best model output
        if 'best_model_output' in result:
            bmo_path = os.path.join(save_dir, f"bmo_episode_{episode_idx:06d}_{self.timestamp}.npz")
            bmo_data = {k: v.cpu().numpy() if torch.is_tensor(v) else v 
                       for k, v in result['best_model_output'].items()}
            np.savez(bmo_path, **bmo_data)
        
        print(f"Planning results saved to: {save_dir}")
    
    def _plot_rewards(self, eval_outputs, save_dir=None):
        """Plot reward progression over iterations."""
        rewards_per_iter = [torch.max(out['reward_seqs']).item() for out in eval_outputs]
        plt.figure()
        plt.plot(rewards_per_iter)
        plt.xlabel("Iteration")
        plt.ylabel("Max Reward in Batch")
        plt.title("Planner Reward vs. Iteration")
        
        if save_dir is not None:
            plot_path = os.path.join(save_dir, f"reward_{self.timestamp}.png")
            plt.savefig(plot_path)
            print(f"Reward plot saved to: {plot_path}")
        else:
            plt.show()
        plt.close()


# Tips to tune MPC:
# - When sampling actions, noise_level should be large enough to have enough coverage, but not too large to cause instability
# - Larger n_sample should lead to better performance, but it will also increase the computation cost
# - Properly tune reward_weight, higher reward_weight encourages to 'exploit' the current best action sequence, while lower reward_weight encourages to 'explore' more action sequences
# - Plot reward vs. iteration to see the convergence of the planner


class Planner(object):
    """
    MPPI-based trajectory optimizer for physics simulation and control.
    Samples action sequences, evaluates them using a physics simulator and reward function,
    and optimizes to find the best action sequence for maximizing expected reward.
    Supports MPPI, gradient descent, and hybrid optimization methods.
    """

    def __init__(self, config):
        """
        config contains following keys:
        - action_dim: the dimension of the action
        - model_rollout_fn:
          - description: the function to rollout the model
          - input:
            - state_cur (shape: [n_his, state_dim] torch tensor)
            - action_seqs (shape: [n_sample, n_look_ahead, action_dim] torch tensor)
          - output: a dict containing the following keys:
            - state_seqs: the sequence of the state, shape: [n_sample, n_look_ahead, state_dim] torch tensor
            - any other keys that you want to return
        - evaluate_traj_fn:
          - description: the function to evaluate the trajectory
          - input:
            - state_seqs (shape: [n_sample, n_look_ahead, state_dim] torch tensor)
            - action_seqs (shape: [n_sample, n_look_ahead, action_dim] torch tensor)
          - output: a dict containing the following keys:
            - reward_seqs (shape: [n_sample] torch tensor)
            - any other keys that you want to return
        - n_sample: the number of action trajectories to sample
        - n_look_ahead: the number of steps to look ahead
        - n_update_iter: the number of iterations to update the action sequence
        - reward_weight: the weight of the reward to aggregate action sequences
        - action_lower_lim:
          - description: the lower limit of the action
          - shape: [action_dim]
          - type: torch tensor
        - action_upper_lim: the upper limit of the action
          - description: the upper limit of the action
          - shape: [action_dim]
          - type: torch tensor
        - planner_type: the type of the planner (options: 'GD', 'MPPI', 'MPPI_GD')
        """
        self.config = config
        self.action_dim = config['action_dim']
        self.model_rollout = config['model_rollout_fn']
        self.evaluate_traj = config['evaluate_traj_fn']
        self.n_sample = config['n_sample']
        self.beta = config['beta']
        self.n_look_ahead = config['n_look_ahead']
        self.n_update_iter = config['n_update_iter']
        self.reward_weight = config['reward_weight']
        self.action_lower_lim = config['action_lower_lim']
        self.action_upper_lim = config['action_upper_lim']
        self.planner_type = config['planner_type']
        assert self.planner_type in ['GD', 'MPPI', 'MPPI_GD']
        assert self.action_lower_lim.shape == (self.action_dim,)
        assert self.action_upper_lim.shape == (self.action_dim,)
        assert type(self.action_lower_lim) == torch.Tensor
        assert type(self.action_upper_lim) == torch.Tensor
        
        # OPTIONAL
        # - device: 'cpu' or 'cuda', default: 'cuda'
        # - verbose: True or False, default: False
        # - sampling_action_seq_fn:
        #   - description: the function to sample the action sequence
        #   - input: init_act_seq (shape: [n_look_ahead, action_dim] torch tensor)
        #   - output: act_seqs (shape: [n_sample, n_look_ahead, action_dim] torch tensor)
        #   - default: sample action sequences from a normal distribution
        # - noise_type: the type of the noise (options: 'normal'), default: 'normal'
        # - noise_level: the level of the noise, default: 0.1
        # - n_his: the number of history states to use, default: 1
        # - rollout_best: whether rollout the best act_seq and get model prediction and reward. True or False, default: True
        # - lr: the learning rate of the optimizer, default: 1e-3
        self.device = config['device'] if 'device' in config else 'cuda'
        self.verbose = config['verbose'] if 'verbose' in config else False
        self.sample_action_sequences = config['sampling_action_seq_fn'] if 'sampling_action_seq_fn' in config else self.sample_action_sequences_default
        self.noise_type = config['noise_type'] if 'noise_type' in config else 'normal'
        assert self.noise_type in ['normal', 'fps']
        self.noise_level = config['noise_level'] if 'noise_level' in config else 0.1
        self.n_his = config['n_his'] if 'n_his' in config else 1
        self.rollout_best = config['rollout_best'] if 'rollout_best' in config else True
        self.lr= config['lr'] if 'lr' in config else 1e-3

    def sample_action_sequences_default(self, act_seq):
        """
        Sample action sequences with noise perturbation for trajectory optimization.
        Applies temporal filtering and clipping to generate diverse action candidates.
        
        Args:
            act_seq: [n_look_ahead, action_dim] - initial action sequence
            
        Returns:
            act_seqs: [n_sample, n_look_ahead, action_dim] - perturbed action sequences
        """
        assert act_seq.shape == (self.n_look_ahead, self.action_dim)
        assert type(act_seq) == torch.Tensor
        
        if self.noise_type == "fps":
            action_lower_lim_np = self.action_lower_lim.cpu().numpy()
            action_upper_lim_np = self.action_upper_lim.cpu().numpy()
            grid_size = 0.02
            grid_axis = []
            for i in range(self.action_dim):
                grid_axis.append(np.arange(action_lower_lim_np[i], action_upper_lim_np[i], grid_size))
            grids = np.meshgrid(*grid_axis)
            grids = np.stack(grids, axis=-1).reshape(-1, self.action_dim)
            act_seqs = fps_rad(grids, self.n_sample) # (n_sample, action_dim)
            act_seqs = torch.from_numpy(act_seqs).to(self.device).float()
            act_seqs = act_seqs.unsqueeze(1).repeat(1, self.n_look_ahead, 1)
            return act_seqs

        beta_filter = self.beta

        # [n_sample, n_look_ahead, action_dim]
        act_seqs = torch.stack([act_seq.clone()] * self.n_sample)

        # [n_sample, action_dim]
        act_residual = torch.zeros((self.n_sample, self.action_dim), dtype=act_seqs.dtype, device=self.device)

        # actions that go as input to the dynamics network
        for i in range(self.n_look_ahead):
            if self.noise_type == "normal":
                noise_sample = torch.normal(0, self.noise_level, (self.n_sample, self.action_dim), device=self.device)
            else:
                raise ValueError("unknown noise type: %s" %(self.noise_type))

            act_residual = beta_filter * noise_sample + act_residual * (1. - beta_filter)

            # add the perturbation to the action sequence
            act_seqs[:, i] += act_residual

            # clip to range
            act_seqs[:, i] = torch.clamp(act_seqs[:, i],
                                         self.action_lower_lim,
                                         self.action_upper_lim)

        assert act_seqs.shape == (self.n_sample, self.n_look_ahead, self.action_dim)
        assert type(act_seqs) == torch.Tensor
        return act_seqs

    def optimize_action(self, act_seqs, reward_seqs, optimizer=None):
        """
        Optimize action sequences based on reward feedback using selected planner type.
        Routes to appropriate optimization method (MPPI, GD, or hybrid).
        
        Args:
            act_seqs: [n_sample, n_look_ahead, action_dim] - candidate action sequences
            reward_seqs: [n_sample] - reward for each action sequence
            optimizer: optimizer for gradient descent (optional)
            
        Returns:
            optimized action sequence based on planner type
        """
        assert act_seqs.shape == (self.n_sample, self.n_look_ahead, self.action_dim)
        assert reward_seqs.shape == (self.n_sample,)
        assert type(act_seqs) == torch.Tensor
        assert type(reward_seqs) == torch.Tensor

        if self.planner_type == 'MPPI':
            return self.optimize_action_mppi(act_seqs, reward_seqs)
        elif self.planner_type == 'GD':
            return self.optimize_action_gd(act_seqs, reward_seqs, optimizer)
        elif self.planner_type == 'MPPI_GD':
            raise NotImplementedError
        else:
            raise ValueError("unknown planner type: %s" %(self.planner_type))

    def trajectory_optimization(self, state_cur, act_seq):
        """
        Main trajectory optimization interface using specified planner type.
        Iteratively samples, evaluates, and optimizes action sequences for optimal control.
        
        Args:
            state_cur: [n_his, state_dim] - current state history
            act_seq: [n_look_ahead, action_dim] - initial action sequence guess
            
        Returns:
            dict containing:
                act_seq: [n_look_ahead, action_dim] - optimized action sequence
                model_outputs: list of model rollout results (if verbose)
                eval_outputs: list of reward evaluation results (if verbose)  
                best_model_output: final model rollout (if rollout_best)
                best_eval_output: final reward evaluation (if rollout_best)
        """
        assert type(state_cur) == torch.Tensor
        assert act_seq.shape == (self.n_look_ahead, self.action_dim)
        assert type(act_seq) == torch.Tensor
        if self.planner_type == 'MPPI':
            return self.trajectory_optimization_mppi(state_cur, act_seq)
        elif self.planner_type == 'GD':
            return self.trajectory_optimization_gd(state_cur, act_seq)
        elif self.planner_type == 'MPPI_GD':
            raise NotImplementedError
        else:
            raise ValueError("unknown planner type: %s" %(self.planner_type))
    
    def optimize_action_mppi(self, act_seqs, reward_seqs):
        """
        Model Predictive Path Integral (MPPI) optimization step.
        Computes weighted average of action sequences based on softmax of rewards.
        
        Args:
            act_seqs: [n_sample, n_look_ahead, action_dim] - candidate action sequences
            reward_seqs: [n_sample] - reward for each sequence
            
        Returns:
            act_seq: [n_look_ahead, action_dim] - optimized action sequence
        """
        act_seq = torch.sum(act_seqs * F.softmax(reward_seqs * self.reward_weight, dim=0).unsqueeze(-1).unsqueeze(-1), dim=0)
        return self.clip_actions(act_seq)
    
    def optimize_action_gd(self, act_seqs, reward_seqs, optimizer):
        """
        Gradient descent optimization step for action sequences.
        Minimizes negative reward (maximizes reward) through backpropagation.
        
        Args:
            act_seqs: [n_sample, n_look_ahead, action_dim] - action sequences with gradients
            reward_seqs: [n_sample] - reward for each sequence
            optimizer: PyTorch optimizer for gradient updates
        """
        loss = -torch.mean(reward_seqs)
        
        optimizer.zero_grad()
        loss.backward()
        try:
            assert torch.isnan(act_seqs.grad).sum() == 0
        except:
            print('act_seqs:', act_seqs)
            print('act_seqs.grad:', act_seqs.grad)
            exit()
        optimizer.step()
    
    def optimize_action_mppi_gd(self, act_seqs, reward_seqs):
        pass

    def clip_actions(self, act_seqs):
        """
        Clip action sequences to specified bounds.
        Ensures all actions remain within valid control limits.
        
        Args:
            act_seqs: [**dim, action_dim] - action sequences to clip
            
        Returns:
            act_seqs: [**dim, action_dim] - clipped action sequences
        """
        act_seqs.data.clamp_(self.action_lower_lim, self.action_upper_lim)
        return act_seqs
    
    def trajectory_optimization_mppi(self, state_cur, act_seq):
        """
        MPPI trajectory optimization implementation.
        Iteratively samples action sequences, evaluates via physics simulation,
        and updates using weighted averaging based on trajectory rewards.
        
        Args:
            state_cur: [n_his, state_dim] - current state history
            act_seq: [n_look_ahead, action_dim] - initial action sequence
            
        Returns:
            dict with optimized action sequence and optional debug outputs
        """
        if self.verbose:
            model_outputs = []
            eval_outputs = []
        for i in range(self.n_update_iter):
            with torch.no_grad():
                act_seqs = self.sample_action_sequences(act_seq)
                assert act_seqs.shape == (self.n_sample, self.n_look_ahead, self.action_dim)
                assert type(act_seqs) == torch.Tensor
                model_out = self.model_rollout(state_cur, act_seqs)
                state_seqs = model_out['state_seqs']
                assert type(state_seqs) == torch.Tensor
                eval_out = self.evaluate_traj(state_seqs, act_seqs)
                reward_seqs = eval_out['reward_seqs']
                act_seq = self.optimize_action(act_seqs, reward_seqs)
                if self.verbose:
                    model_outputs.append(model_out)
                    eval_outputs.append(eval_out)

        if self.rollout_best:
            best_model_out = self.model_rollout(state_cur, act_seq.unsqueeze(0))
            best_eval_out = self.evaluate_traj(best_model_out['state_seqs'], act_seq.unsqueeze(0))
                
        return {'act_seq': act_seq,
                'model_outputs': model_outputs if self.verbose else None,
                'eval_outputs': eval_outputs if self.verbose else None,
                'best_model_output': best_model_out if self.rollout_best else None,
                'best_eval_output': best_eval_out if self.rollout_best else None}
    
    def trajectory_optimization_gd(self, state_cur, act_seq):
        """
        Gradient descent trajectory optimization implementation.
        Uses differentiable physics simulation to optimize action sequences
        through backpropagation and gradient-based updates.
        
        Args:
            state_cur: [n_his, state_dim] - current state history
            act_seq: [n_look_ahead, action_dim] - initial action sequence
            
        Returns:
            dict with optimized action sequence and optional debug outputs
        """
        act_seqs = self.sample_action_sequences(act_seq).requires_grad_() # (n_sample, n_look_ahead, action_dim)
        optimizer = torch.optim.Adam([act_seqs], lr=self.lr, betas=(0.9, 0.999))
        if self.verbose:
            model_outputs = []
            eval_outputs = []
        for i in range(self.n_update_iter):
            assert act_seqs.shape == (self.n_sample, self.n_look_ahead, self.action_dim)
            assert type(act_seqs) == torch.Tensor
            model_out = self.model_rollout(state_cur, act_seqs)
            state_seqs = model_out['state_seqs']
            assert type(state_seqs) == torch.Tensor
            eval_out = self.evaluate_traj(state_seqs, act_seqs)
            reward_seqs = eval_out['reward_seqs'] # (n_sample)
            self.optimize_action(act_seqs, reward_seqs, optimizer)
            self.clip_actions(act_seqs)
            if self.verbose:
                model_outputs.append(model_out)
                eval_outputs.append(eval_out)
        act_seq = act_seqs[torch.argmax(reward_seqs)]
        
        if self.rollout_best:
            best_model_out = self.model_rollout(state_cur, act_seq.unsqueeze(0))
            best_eval_out = self.evaluate_traj(best_model_out['state_seqs'], act_seq.unsqueeze(0))
                
        return {'act_seq': act_seq,
                'model_outputs': model_outputs if self.verbose else None,
                'eval_outputs': eval_outputs if self.verbose else None,
                'best_model_output': best_model_out if self.rollout_best else None,
                'best_eval_output': best_eval_out if self.rollout_best else None}
    
    def trajectory_optimization_mppi_gd(self, state_cur, act_seq = None):
        pass
