import torch
import argparse
import h5py
import numpy as np
import os
import matplotlib.pyplot as plt
import glob
import warp as wp
import open3d as o3d
import cv2
import pickle
import json

from .qqtt import InvPhyTrainerWarp
from .qqtt.utils import logger, cfg
from .qqtt.engine.robot_movement_controller import RobotMovementController
from .config_manager import PhysTwinConfig, create_common_parser
from .paths import get_case_paths, URDF_XARM7
from scripts.planner import Planner
from scripts.reward import RewardFn
from GNN.utils import load_yaml


class PhysTwinModelRolloutFn:
    def __init__(self, trainer, robot_mask, device):
        """
        Initialize the PhysTwin model rollout function.
        
        Args:
            trainer: InvPhyTrainerWarp instance with loaded model
            robot_mask: [n_particles] - boolean tensor for robot particles
            device: torch device
        """
        self.trainer = trainer
        self.robot_mask = robot_mask
        self.device = device
        self.n_particles = len(robot_mask)
        self.n_object_particles = self.trainer.num_all_points
        self.n_robot_particles = (~robot_mask[:self.n_object_particles]).sum().item() + robot_mask[self.n_object_particles:].sum().item()
        
        # Calculate frame duration for velocity conversion
        self.frame_duration = cfg.dt * cfg.num_substeps
        
        # Create robot movement controller
        self.robot_controller = RobotMovementController(
            robot=self.trainer.robot,
            n_ctrl_parts=1,
            device=device
        )
        
    def __call__(self, state_cur, action_seqs):
        """
        Evaluate multiple action sequences using sequential PhysTwin rollouts.
        
        Args:
            state_cur: [1, state_dim] - current state (object + robot particles)
            action_seqs: [n_sample, n_look_ahead, 3] - robot translation sequences
            
        Returns:
            dict containing:
                state_seqs: [n_sample, n_look_ahead, state_dim] - predicted trajectories
        """
        n_sample, n_look_ahead, _ = action_seqs.shape
        
        # Extract initial state
        initial_state = state_cur[-1].view(self.n_particles, 3)  # [n_particles, 3]
        initial_object_state = initial_state[:self.n_object_particles]  # [n_object_particles, 3]
        initial_robot_state = initial_state[self.n_object_particles:]  # [n_robot_particles, 3]
        
        # Sequential rollout for each action sequence sample
        state_seqs = torch.zeros(n_sample, n_look_ahead, state_cur.shape[1], device=self.device)
        
        for sample_idx in range(n_sample):
            predicted_states = self._rollout_single_sequence(
                initial_object_state,
                initial_robot_state, 
                action_seqs[sample_idx]  # [n_look_ahead, 3]
            )
            state_seqs[sample_idx] = predicted_states.flatten(start_dim=1)  # [n_look_ahead, n_particles * 3]
            
        return {'state_seqs': state_seqs}
    
    def _rollout_single_sequence(self, initial_object_state, initial_robot_state, action_seq):
        """
        Run PhysTwin simulation for single action sequence.
        
        Args:
            initial_object_state: [n_object_particles, 3]
            initial_robot_state: [n_robot_particles, 3] 
            action_seq: [n_look_ahead, 3] - robot translation sequence
            
        Returns:
            predicted_states: [n_look_ahead, n_particles, 3] - combined object + robot states
        """
        # Reset simulator to initial object state
        initial_state_warp = wp.from_torch(initial_object_state.contiguous(), dtype=wp.vec3)
        self.trainer.simulator.set_init_state(
            initial_state_warp,
            self.trainer.simulator.wp_init_velocities  # Reset velocities to zero
        )
        
        # Reset robot to initial position (reconstruct from initial_robot_state)
        self._reset_robot_to_state(initial_robot_state)
        
        predicted_states = []
        
        for step_idx in range(len(action_seq)):
            # Apply robot translation using controller
            robot_translation = action_seq[step_idx].cpu().numpy()
            
            # Update robot movement using the controller
            movement_result = self.robot_controller.update_robot_movement(
                target_change=np.array([robot_translation]),  # Shape: [1, 3] for n_ctrl_parts=1
                finger_change=0.0,  # Fixed gripper opening
                rot_change=None
            )
            
            # Update simulator with proper robot movement
            self.trainer.simulator.set_mesh_interactive(
                movement_result['interpolated_dynamic_points'],
                movement_result['interpolated_center'],
                movement_result['dynamic_velocity'],
                movement_result['dynamic_omega'],
            )
            
            # Run physics step with collision detection
            if self.trainer.simulator.object_collision_flag:
                self.trainer.simulator.update_collision_graph()
            wp.capture_launch(self.trainer.simulator.forward_graph)
            
            # Get new object state
            x = wp.to_torch(self.trainer.simulator.wp_states[-1].wp_x, requires_grad=False)
            object_state = x[:self.n_object_particles]  # Only object particles
            
            # Combine object and robot states (use final robot position)
            combined_state = torch.cat([object_state, movement_result['current_trans_dynamic_points']], dim=0)
            predicted_states.append(combined_state)
            
            # Update simulator state for next step
            self.trainer.simulator.set_init_state(
                self.trainer.simulator.wp_states[-1].wp_x,
                self.trainer.simulator.wp_states[-1].wp_v,
            )
        
        return torch.stack(predicted_states)  # [n_look_ahead, n_particles, 3]
    
    def _reset_robot_to_state(self, robot_state):
        """Reset robot to match the given robot state."""
        # For simplicity, reset to initial robot configuration
        # In a full implementation, this would reconstruct robot pose from robot_state
        start_position = np.mean(robot_state.cpu().numpy(), axis=0)
        current_position = np.mean(self.robot_controller.current_trans_dynamic_points.cpu().numpy(), axis=0)
        translation = start_position - current_position
        self.trainer.robot.change_init_pose(translation)
        self.trainer.reset_robot()
        # Reset the robot controller to match the new robot position
        self.robot_controller.reset()


class PhysTwinPlannerWrapper:
    """Wrapper for PhysTwin-based model predictive control"""
    
    def __init__(self, mpc_config_path: str, case_name: str, 
                 base_path: str = None, bg_img_path: str = None, gaussian_path: str = None):
        """
        Initialize the PhysTwin planner wrapper.
        
        Args:
            mpc_config_path: Path to MPC configuration file
            case_name: Name of the case/experiment
            base_path: Base path for data (optional)
            bg_img_path: Path to background image (optional)
            gaussian_path: Path to gaussian output directory (optional)
        """
        # Initialize configuration
        self.config = PhysTwinConfig(
            case_name=case_name,
            base_path=base_path,
            bg_img_path=bg_img_path,
            gaussian_path=gaussian_path
        )
        
        # Load MPC configuration
        self.mpc_config = load_yaml(mpc_config_path)   
             
        # Setup device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Extract MPC parameters
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
        
        # Initialize trainer
        self.trainer = self._initialize_trainer()
        
        print(f"PhysTwin MPC initialized for case: {case_name}")
        print(f"Using device: {self.device}")
        print(f"History length: {self.n_history}")

    def _initialize_trainer(self):
        """Initialize PhysTwin trainer with loaded model."""
        # Create robot
        sample_robot = self.config.create_robot("default")
        
        # Create trainer
        trainer = InvPhyTrainerWarp(
            data_path=self.config.get_data_path(),
            base_dir=str(self.config.case_paths['base_dir']),
            pure_inference_mode=True,
            static_meshes=[],
            robot=sample_robot,
        )
        
        # Initialize simulator with trained model
        best_model_path = self.config.get_best_model_path()
        trainer.initialize_simulator(best_model_path)
        
        return trainer
        
    def load_data(self, episode_idx):
        """
        Load data from the specified episode.    

        Args: 
            episode_idx: int - episode to load

        Returns:
            first_states: [n_particles, 3] - combined object and robot particle positions
            robot_mask: [n_particles] - boolean tensor for robot particles
        """
        with h5py.File(self.mpc_config['data_file'], 'r') as f:
            # Read the first frame
            episode_group = f[f'episode_{episode_idx:06d}']
            object_data = episode_group['object'][0]
            robot_data = episode_group['robot'][0] 
            n_object = object_data.shape[0]
            n_robot = robot_data.shape[0]

            # Convert to tensors
            object_data = torch.tensor(object_data, dtype=torch.float32, device=self.device)
            robot_data = torch.tensor(robot_data, dtype=torch.float32, device=self.device)
            
            # Combine object and robot data
            first_states = torch.cat([object_data, robot_data], dim=0)
            
            # Create robot mask
            robot_mask = torch.cat([
                torch.zeros(n_object, dtype=torch.bool, device=self.device), 
                torch.ones(n_robot, dtype=torch.bool, device=self.device)
            ], dim=0)

            return first_states, robot_mask

    def plan_action(self, episode_idx):
        """
        Plan action sequence to go from the initial state to the goal state.

        Args:
            episode_idx: int - episode containing the initial state

        Returns:
            dict containing:
                act_seq: [n_look_ahead, 3] - action sequence
                eval_outputs: list of dicts containing evaluation outputs for each iteration
                best_model_output: dict containing best model output
        """
        # Prepare data
        first_states, robot_mask = self.load_data(episode_idx)
        
        # PhysTwin doesn't need history - just current state
        state_cur = first_states.clone().unsqueeze(0)  # [1, n_particles, 3]
        state_cur = state_cur.flatten(start_dim=1)  # [1, n_particles * 3]

        # Set up the model rollout function
        model_rollout_fn = PhysTwinModelRolloutFn(self.trainer, robot_mask, self.device)
        initial_action_seq = torch.zeros(self.n_look_ahead, self.action_dim, device=self.device)

        # Set up the reward function
        reward_fn = RewardFn(self.action_weight, robot_mask)

        # Set up the planner
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
        result = planner.trajectory_optimization(state_cur, initial_action_seq)
        
        # Plot rewards if verbose
        if self.verbose and 'eval_outputs' in result and result['eval_outputs'] is not None:
            rewards_per_iter = [torch.max(out['reward_seqs']).item() for out in result['eval_outputs']]
            plt.figure()
            plt.plot(rewards_per_iter)
            plt.xlabel("Iteration")
            plt.ylabel("Max Reward in Batch")
            plt.title("PhysTwin Planner Reward vs. Iteration")
            plot_path = os.path.join("PhysTwin", "tasks", "reward_plot.png")
            os.makedirs(os.path.dirname(plot_path), exist_ok=True)
            plt.savefig(plot_path)
            print(f"Reward plot saved to: {plot_path}")

        return result

    def visualize_action(self, episode_idx, action_seq, predicted_states):
        """
        Visualize the planned action sequence and predicted states.

        Args:
            episode_idx: int - episode containing the initial state
            action_seq: [n_look_ahead, 3] torch tensor 
            predicted_states: [n_look_ahead, n_particles*3] predicted states
        """
                
        # Get data for visualization
        first_states, robot_mask = self.load_data(episode_idx)
        n_particles = len(robot_mask)
        
        # Reshape predicted states: [n_look_ahead, n_particles*3] -> [n_look_ahead, n_particles, 3]
        predicted_states = predicted_states.reshape(-1, n_particles, 3)
        
        # Add initial state to create full trajectory
        initial_state = first_states.unsqueeze(0)  # [1, n_particles, 3]
        full_trajectory = torch.cat([initial_state, predicted_states], dim=0)  # [n_look_ahead+1, n_particles, 3]
        
        # Get target from reward function
        reward_fn = RewardFn(self.action_weight, robot_mask)
        target_pcd = reward_fn.target.cpu().numpy()  # [n_target_particles, 3]
        
        # Load camera parameters from config files (same as v_from_d.py)
        case_paths = get_case_paths(self.case_name)
        data_path = case_paths['data_dir'] 
        
        # Load camera calibration
        with open(f"{data_path}/calibrate.pkl", "rb") as f:
            c2ws = pickle.load(f)
        w2cs = [np.linalg.inv(c2w) for c2w in c2ws]
        
        # Load metadata for intrinsics and window size
        with open(f"{data_path}/metadata.json", "r") as f:
            data = json.load(f)
        intrinsics = np.array(data["intrinsics"])
        width, height = data["WH"]
        
        # Use first camera view
        vis_cam_idx = 0
        intrinsic = intrinsics[vis_cam_idx]
        w2c = w2cs[vis_cam_idx]
        
        # Setup visualization with proper camera parameters
        vis = o3d.visualization.Visualizer()
        vis.create_window(visible=False, width=width, height=height)
        render_option = vis.get_render_option()
        render_option.point_size = 10.0
        
        # Create point clouds
        object_pcd = o3d.geometry.PointCloud()
        robot_pcd = o3d.geometry.PointCloud()
        target_pcd_vis = o3d.geometry.PointCloud()
        
        # Set target point cloud (static, blue)
        target_pcd_vis.points = o3d.utility.Vector3dVector(target_pcd)
        target_pcd_vis.paint_uniform_color([0, 0, 1])  # Blue
        vis.add_geometry(target_pcd_vis)
        
        # Initialize with first frame
        initial_frame = full_trajectory[0].cpu().numpy()
        object_points = initial_frame[~robot_mask.cpu()]
        robot_points = initial_frame[robot_mask.cpu()]
        
        object_pcd.points = o3d.utility.Vector3dVector(object_points)
        object_pcd.paint_uniform_color([0, 1, 0])  # Green for predicted object states
        vis.add_geometry(object_pcd)
        
        robot_pcd.points = o3d.utility.Vector3dVector(robot_points)
        robot_pcd.paint_uniform_color([1, 0, 0])  # Red for robot
        vis.add_geometry(robot_pcd)
        
        # Setup camera view
        view_control = vis.get_view_control()
        camera_params = o3d.camera.PinholeCameraParameters()
        intrinsic_parameter = o3d.camera.PinholeCameraIntrinsic(
            width, height, intrinsic
        )
        camera_params.intrinsic = intrinsic_parameter
        camera_params.extrinsic = w2c
        view_control.convert_from_pinhole_camera_parameters(
            camera_params, allow_arbitrary=True
        )
        
        # Create output directory and video writer
        output_dir = "PhysTwin/tasks"
        os.makedirs(output_dir, exist_ok=True)
        video_path = os.path.join(output_dir, f"mpc_episode_{episode_idx}.mp4")
        
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        fps = cfg.FPS  # Use FPS from config
        out = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
        
        # Animate through trajectory
        for frame_idx in range(len(full_trajectory)):
            current_frame = full_trajectory[frame_idx].cpu().numpy()
            object_points = current_frame[~robot_mask.cpu()]
            robot_points = current_frame[robot_mask.cpu()]
            
            # Update point clouds
            object_pcd.points = o3d.utility.Vector3dVector(object_points)
            robot_pcd.points = o3d.utility.Vector3dVector(robot_points)
            
            vis.update_geometry(object_pcd)
            vis.update_geometry(robot_pcd)
            vis.poll_events()
            vis.update_renderer()
            
            # Capture frame
            image = np.asarray(vis.capture_screen_float_buffer(do_render=True))
            image = (image * 255).astype(np.uint8)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            out.write(image)
            
            # Show progress
            if frame_idx % 5 == 0:
                print(f"Rendering frame {frame_idx}/{len(full_trajectory)-1}")
        
        # Clean up
        out.release()
        vis.destroy_window()
        
        print(f"MPC visualization saved to: {video_path}")
        return video_path


if __name__ == "__main__":
    """
    Demonstrate PhysTwin-based model predictive control.
    """
    # Create parser with common arguments
    parser = create_common_parser()
    
    # Add script-specific arguments
    parser.add_argument("--episode", type=int, required=True,
                       help="Episode index to load")
    parser.add_argument("--mpc_config", type=str, default="PhysTwin/configs/mpc_config.yaml",
                       help="Path to MPC config file")
    parser.add_argument("--data_file", type=str, default=None,
                       help="Path to episode data file")
    args = parser.parse_args()
    
    with torch.no_grad():
        # Initialize planner wrapper
        planner_wrapper = PhysTwinPlannerWrapper(
            mpc_config_path=args.mpc_config,
            case_name=args.case_name,
            base_path=args.base_path,
            bg_img_path=args.bg_img_path,
            gaussian_path=args.gaussian_path
        )
    
        # Demonstrate planning
        print("="*60)
        print("PHYSTWIN-BASED MODEL PREDICTIVE CONTROL DEMO")
        print("="*60)
    
        # Plan action sequence for the specified episode
        planning_result = planner_wrapper.plan_action(args.episode)
        optimal_actions = planning_result['act_seq']
        print(f"Planned action sequence shape: {optimal_actions.shape}")
        print("Planning demonstration complete!")

        # Visualize action sequence
        if 'best_model_output' in planning_result:
            predicted_states = planning_result['best_model_output']['state_seqs']
            planner_wrapper.visualize_action(args.episode, optimal_actions, predicted_states, args.data_file) 