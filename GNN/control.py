import torch
import argparse
import numpy as np
import os

from .model.rollout import Rollout
from .model.gnn_dyn import PropNetDiffDenModel
from .utils import load_yaml, Visualizer
from .paths import get_model_paths
from scripts.planner import PlannerWrapper
from scripts.utils import load_mpc_data, setup_task_directory
from scripts.reward import RewardFn

# PhysTwin imports for actual deformation simulation
from PhysTwin.qqtt import InvPhyTrainerWarp
from PhysTwin.qqtt.utils import cfg
from PhysTwin.config_manager import PhysTwinConfig
from PhysTwin.control import PhysTwinModelRolloutFn
from PhysTwin.paths import URDF_XARM7
from PhysTwin.robot import RobotLoader


class GNNModelRolloutFn:
    def __init__(self, model, config, robot_mask, topological_edges, first_states):
        """
        Initialize the model rollout function.
        
        Args:
            model: trained GNN model
            config: dict - training configuration
            robot_mask: [n_particles] - boolean tensor for robot particles
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
        self.topological_edges = topological_edges.to(self.device)
        self.first_states = first_states.to(self.device)

    def __call__(self, state_cur, action_seqs):
        """
        Evaluate multiple action sequences using batched Rollout operations.
        
        Args:
            state_cur: [n_history, state_dim] - current state history 
            action_seqs: [n_sample, n_look_ahead, 3] - robot velocity sequences (broadcast to all robot particles)
            
        Returns:
            dict containing:
                state_seqs: [n_sample, n_look_ahead, state_dim] - predicted trajectories
        """
        n_sample, n_look_ahead, _ = action_seqs.shape
        filler = torch.zeros((n_sample, n_look_ahead, 1), device=action_seqs.device)
        action_seqs = torch.cat([action_seqs, filler], dim=2)

        # Reshape states: [n_history, n_particles*3]] -> [n_sample, n_history, n_particles, 3]
        states = state_cur.view(self.n_history, self.n_particles, 3)
        states = states.unsqueeze(0).repeat(n_sample, 1, 1, 1)

        # Calculate delta history: [n_sample, n_history-1, n_particles, 3]
        states_delta = states[:, 1:] - states[:, :-1]

        # Create attributes (0=object, 1=robot) [n_sample, n_particles]
        attrs = self.robot_mask.clone().float().unsqueeze(0).repeat(n_sample, 1)
        
        # Broadcast robot actions to all robot particles: [n_sample, n_look_ahead, 3] -> [n_sample, n_look_ahead, n_robots, 3]
        robot_actions = action_seqs.unsqueeze(2).repeat(1, 1, self.n_robots, 1)
        
        particle_nums = torch.tensor([self.n_particles] * n_sample, device=self.device)

        # Adjust topological_edges and first_states to match current batch size
        topological_edges = self.topological_edges.unsqueeze(0).repeat(n_sample, 1, 1)
        first_states = self.first_states.unsqueeze(0).repeat(n_sample, 1, 1)

        rollout = Rollout(
            self.model, 
            self.config, 
            states[:, -1],          # [n_sample, n_particles, 3] - current state (last frame)
            states_delta,           # [n_sample, n_history-1, n_particles, 3] - delta history
            attrs,                  # [n_sample, n_particles] - attributes
            particle_nums,          # [n_sample] - particle counts
            topological_edges,      # [n_sample, n_particles, n_particles] - topological edges
            first_states,           # [n_sample, n_particles, 3] - first frame states
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


class PhysTwinInGNN:
    """
    Compute ground truth deformation using PhysTwin.
    """
    
    def __init__(self, case_name, downsample_rate):
        """
        Initialize PhysTwin simulator.
        
        Args:
            case_name: str - case name for PhysTwin configuration
            downsample_rate: int - downsampling rate for configuration adjustment
        """
        self.case_name = case_name
        self.downsample_rate = downsample_rate
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                
        # Create PhysTwin configuration
        phystwin_config = PhysTwinConfig(case_name=self.case_name)
        
        # Adjust PhysTwin configuration for GNN downsampling
        cfg.num_substeps = int(cfg.num_substeps * self.downsample_rate)
        cfg.FPS = int(cfg.FPS / self.downsample_rate)
        print(f"Adjusted PhysTwin config: substeps={cfg.num_substeps}, FPS={cfg.FPS}")
        
        self.trainer = InvPhyTrainerWarp(
            data_path=phystwin_config.get_data_path(),
            base_dir=str(phystwin_config.case_paths['base_dir']),
            pure_inference_mode=True,
            static_meshes=[],
            robot_controller=phystwin_config.get_robot_controller("default"),
        )
        
        # Initialize simulator with trained model
        best_model_path = phystwin_config.get_best_model_path()
        self.trainer.initialize_simulator(best_model_path)
        
        print("PhysTwin initialization completed!")
        
    def compute_deformation(self, action_seq, phystwin_states, phystwin_robot_mask, 
                          action_weight=None, fsp_weight=None, target_pcd=None):
        """
        Compute actual object deformation using PhysTwin simulation.
        Compute reward if target_pcd is provided.
        
        Args:
            action_seq: [n_look_ahead, action_dim] - action sequence
            phystwin_states: [n_particles, 3] - initial particle states
            phystwin_robot_mask: [n_particles] - robot mask
            action_weight: float - action penalty weight (for reward calculation) (optional)
            fsp_weight: float - final speed penalty weight (for reward calculation) (optional)
            target_pcd: torch.Tensor - target for reward calculation (optional)
            
        Returns:
            actual_trajectory: [n_look_ahead+1, n_particles, 3] - actual deformation trajectory
        """
        # Create PhysTwin model rollout function with PhysTwin robot mask
        phystwin_rollout_fn = PhysTwinModelRolloutFn(self.trainer, phystwin_robot_mask, self.device)
        
        # Extract initial object and robot states
        initial_object_state = phystwin_states[~phystwin_robot_mask]  # [n_object_particles, 3]
        initial_robot_state = phystwin_states[phystwin_robot_mask]  # [n_robot_particles, 3]
            
        # Convert action sequence: [n_look_ahead, action_dim] -> [n_look_ahead, 2] (only x,y)
        action_seq_2d = action_seq[:, :2]  # [n_look_ahead, 2]
            
        # Get actual deformation from PhysTwin using rollout_single_sequence
        print("Computing actual deformation with PhysTwin...")
        predicted_states = phystwin_rollout_fn.rollout_single_sequence(
            initial_object_state, initial_robot_state, action_seq_2d
        )  # [n_look_ahead, n_particles, 3]
            
        actual_initial_state = torch.cat([initial_object_state, initial_robot_state], dim=0).unsqueeze(0)  # [1, n_particles, 3]
        actual_trajectory = torch.cat([actual_initial_state, predicted_states], dim=0)  # [n_look_ahead+1, n_particles, 3]
                        
        # Calculate reward with actual trajectory (optional)
        if target_pcd is not None and action_weight is not None and fsp_weight is not None:
            reward_fn = RewardFn(action_weight, fsp_weight, phystwin_robot_mask, target_pcd)
            # Convert predicted_states to format expected by reward function: [1, n_look_ahead, n_particles*3]
            reward_states = predicted_states.flatten(start_dim=1).unsqueeze(0)  # [1, n_look_ahead, n_particles*3]
            action_seqs = action_seq_2d.unsqueeze(0)  # [1, n_look_ahead, 2]
            actual_reward = reward_fn(reward_states, action_seqs)['reward_seqs'][0].item()
            print(f"Actual reward (PhysTwin deformation): {actual_reward:.6f}")

        return actual_trajectory
            

class GNNPlannerWrapper(PlannerWrapper): 
    def __init__(self, model_path: str, train_config_path: str, mpc_config_path: str, case_name: str):
        """
        Initialize the GNN planner wrapper.
        
        Args:
            model_path: str - path to trained GNN model checkpoint
            train_config_path: str - path to training configuration file
            mpc_config_path: str - path to MPC configuration file
            case_name: str - case name for PhysTwin simulation
        """
        super().__init__(mpc_config_path)
        
        # Load training configuration
        self.train_config = load_yaml(train_config_path)
        self.case_name = case_name
        
        # Initialize GNN model
        self._initialize_model(model_path)
        
        print(f"Loaded GNN model from: {model_path}")
        print(f"History length: {self.n_history}")

    def _initialize_model(self, model_path):
        """Initialize the GNN model."""
        # Load trained model
        self.model = PropNetDiffDenModel(self.train_config, torch.cuda.is_available())
        self.model.load_state_dict(torch.load(model_path, weights_only=True, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        self.downsample_rate = self.train_config['dataset']['downsample_rate']
        self.n_history = self.train_config['train']['n_history'] # mpc and model n_history must agree

    def _prepare_initial_state(self, first_states, robot_mask):
        """Prepare initial state for GNN model."""
        state_cur = first_states.clone().unsqueeze(0).repeat(self.n_history, 1, 1) # [n_history, n_particles, 3]
        state_cur = state_cur.flatten(start_dim=1) # [n_history, n_particles*3]
        return state_cur

    def _create_model_rollout_fn(self, robot_mask, **kwargs):
        """Create GNN model rollout function."""
        topological_edges = kwargs.get('topological_edges')
        first_states = kwargs.get('first_states')
        return GNNModelRolloutFn(self.model, self.train_config, robot_mask, topological_edges, first_states)


def visualize_action_gnn(save_dir, file_name):
    """
    Visualize GNN action sequence using saved results and optionally PhysTwin actual deformation.
    
    Args:
        save_dir: str - directory containing saved results
        file_name: str - filename in format episode_{idx}_{timestamp}
        
    Returns:
        str - path to saved video
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load config file from save_dir
    config_path = os.path.join(save_dir, "mpc_config.yaml")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    config_data = load_yaml(config_path)
    
    # Load target
    target_path = os.path.join(save_dir, "target.npz")
    if not os.path.exists(target_path):
        raise FileNotFoundError(f"Target file not found: {target_path}")
    target_data = np.load(target_path)
    target_pcd = torch.tensor(target_data['target'], dtype=torch.float32, device=device)
    
    # Load predicted states from bmo file
    bmo_path = os.path.join(save_dir, f"bmo_{file_name}.npz")
    if not os.path.exists(bmo_path):
        raise FileNotFoundError(f"BMO file not found: {bmo_path}")
    bmo_data = np.load(bmo_path)
    predicted_states = torch.tensor(bmo_data['state_seqs'], device=device)
    
    # Load action sequence from act_seq file
    act_seq_path = os.path.join(save_dir, f"act_seq_{file_name}.npz")
    if not os.path.exists(act_seq_path):
        raise FileNotFoundError(f"Action sequence file not found: {act_seq_path}")
    act_seq_data = np.load(act_seq_path)
    action_seq = torch.tensor(act_seq_data['act_seq'], device=device)
    
    # Extract episode index from filename
    episode_idx = int(file_name.split('_')[1])
    
    # Load data using config paths
    first_states, robot_mask, topological_edges = load_mpc_data(episode_idx, config_data['data_file'], device)
    
    # Reshape predicted states to [n_look_ahead, n_particles, 3]
    n_particles = robot_mask.shape[0]
    predicted_states = predicted_states.reshape(-1, n_particles, 3)
    
    # Add initial state to create full trajectory
    initial_state = first_states.unsqueeze(0)  # [1, n_particles, 3]
    full_trajectory = torch.cat([initial_state, predicted_states], dim=0)  # [n_look_ahead+1, n_particles, 3]
    
    # Initialize PhysTwin simulator for actual deformation calculation
    downsample_rate = 3
    phystwin_simulator = PhysTwinInGNN(
        case_name="single_push_rope",
        downsample_rate=downsample_rate
    )
    
    # Load PhysTwin data for this specific episode
    phystwin_data_file = config_data['phystwin_data_file']
    phystwin_first_states, phystwin_robot_mask, _ = load_mpc_data(episode_idx, phystwin_data_file, device)
        
    # Compute actual deformation
    actual_trajectory = phystwin_simulator.compute_deformation(
        action_seq=action_seq,
        phystwin_states=phystwin_first_states,
        phystwin_robot_mask=phystwin_robot_mask,
        action_weight=config_data['action_weight'],
        fsp_weight=config_data['fsp_weight'],
        target_pcd=target_pcd
    )
    
    # Filter out robot particles for visualization (only show object particles)
    # actual_trajectory = actual_trajectory[:, ~phystwin_robot_mask, :]
    print("PhysTwin actual deformation computed successfully!")
    
    # Set up visualization paths
    save_path = os.path.join(save_dir, f"{file_name}.mp4")
    
    # Initialize visualizer with camera calibration
    camera_calib_path = "PhysTwin/data/different_types/single_push_rope"
    visualizer = Visualizer(camera_calib_path, downsample_rate)
    
    # Create target trajectory (static target repeated for all timesteps)
    n_frames = len(full_trajectory)
    target_trajectory = target_pcd.unsqueeze(0).repeat(n_frames, 1, 1)  # [n_frames, n_target_particles, 3]
    
    # Use the visualizer to create the video
    video_path = visualizer.visualize_object_motion(
        predicted_states=full_trajectory,
        tool_mask=robot_mask,
        actual_objects=actual_trajectory,
        save_path=save_path,
        topological_edges=topological_edges,
        target=target_pcd.cpu().numpy()
    )
    
    print(f"GNN MPC visualization saved to: {video_path}")
    return video_path


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
    parser.add_argument("--case_name", type=str, default="single_push_rope",
                       help="Case name to use for PhysTwin")
    parser.add_argument("--dir_name", type=str, required=True,
                       help="Directory name within tasks folder to store results")
    args = parser.parse_args()
    
    # Setup paths
    model_paths = get_model_paths(args.model)
    model_path = str(model_paths['net_best'])
    
    # Use training config from model directory if not specified
    if args.train_config is None:
        train_config_path = str(model_paths['config'])
    else:
        train_config_path = args.train_config
        
    with torch.no_grad():
        # Initialize planner wrapper
        planner_wrapper = GNNPlannerWrapper(model_path, train_config_path, args.mpc_config, args.case_name)
    
        # Load initial data to get robot mask for target creation
        first_states, robot_mask, topological_edges = load_mpc_data(args.episode, planner_wrapper.mpc_config['data_file'], planner_wrapper.device)
        
        # Setup task directory and get target
        save_dir, target_pcd = setup_task_directory(args.dir_name, args.mpc_config, planner_wrapper.device, model_type="GNN")
    
        # Demonstrate planning
        print("="*60)
        print("GNN-BASED MODEL PREDICTIVE CONTROL DEMO")
        print("="*60)
    
        # Plan action sequence for the specified episode
        planning_result = planner_wrapper.plan_action(args.episode, target_pcd, save_dir, first_states, robot_mask, topological_edges)
        optimal_actions = planning_result['act_seq']
        print(f"Planned action sequence shape: {optimal_actions.shape}")
        print("Planning demonstration complete!")

        # Visualize action sequence if requested
        if 'best_model_output' in planning_result:
            # Generate filename from episode and timestamp
            timestamp = planner_wrapper.timestamp
            file_name = f"episode_{args.episode:06d}_{timestamp}"
            
            visualize_action_gnn(save_dir, file_name)
