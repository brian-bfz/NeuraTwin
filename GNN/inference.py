import os
import numpy as np
import torch
import cv2
import open3d as o3d
import pickle
import json
from .dataset.dataset_gnn_dyn import ParticleDataset
from .model.gnn_dyn import PropNetDiffDenModel
from .model.rollout import Rollout
from .utils import load_yaml, visualize_edges
from .paths import *
import argparse
from scripts.utils import parse_episodes


class Visualizer:
    """
    GNN-based object motion visualizer for complete episode analysis.
    Handles loading trained models, camera calibration, and generating predictions
    with visualization capabilities using the Rollout class.
    """
    
    def __init__(self, model_path, config_path, camera_calib_path, data_file):
        """
        Initialize the visualizer with trained model and data pipeline.
        
        Args:
            model_path: str - path to trained model checkpoint
            config_path: str - path to training configuration file
            camera_calib_path: str - path to camera calibration data
            data_file: str - path to HDF5 dataset file
        """
        # Load configuration and setup device
        self.config = load_yaml(config_path)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load camera to world transforms
        with open(os.path.join(camera_calib_path, "calibrate.pkl"), "rb") as f:
            self.c2ws = pickle.load(f)
        self.w2cs = [np.linalg.inv(c2w) for c2w in self.c2ws]
            
        with open(os.path.join(camera_calib_path, "metadata.json"), "r") as f:
            data = json.load(f)
        self.intrinsics = np.array(data["intrinsics"])
        self.WH = data["WH"]
        self.FPS = data["fps"] / self.config['dataset']['downsample_rate']
        
        # Initialize dataset for consistent data loading
        if data_file is None:
            data_file = self.config['dataset']['file']
        self.dataset = ParticleDataset(data_file, self.config, 'train')  # Use train to access all episodes
        
        # Extract model parameters
        self.n_history = self.config['train']['n_history']
        self.fps_radius = self.config['train']['particle']['fps_radius']
        self.adj_thresh = self.config['train']['edges']['collision']['adj_thresh']
        self.topk = self.config['train']['edges']['collision']['topk']
        
        # Load trained model
        self.model = PropNetDiffDenModel(self.config, torch.cuda.is_available())
        self.model.load_state_dict(torch.load(model_path, weights_only=True, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        
        print(f"Loaded model from: {model_path}")
        print(f"Using device: {self.device}")
        print(f"Dataset initialized with {self.dataset.n_episode} episodes")
        print(f"History length: {self.n_history}")

    def predict_episode_rollout(self, states, states_delta, attrs, particle_num, topological_edges, first_states):
        """
        Predict object motion for entire episode using autoregressive rollout.
        
        Args:
            states: [timesteps, particles, 3] - full episode states with history padding
            states_delta: [timesteps-1, particles, 3] - particle displacements
                For t < n_history-1: consecutive frame differences
                For t = n_history-1: 0 for objects, actual motion for robots
            attrs: [timesteps, particles] - particle attributes (0=object, 1=robot)
            particle_num: int - total number of particles
            topological_edges: [particles, particles] - adjacency matrix for topological edges
            first_states: [particles, 3] - first frame states for topological edge computations
            
        Returns:
            predicted_states: [timesteps, particles, 3] - complete predicted trajectory
        """
        # Rollout to end of episode
        n_rollout = states.shape[0] - self.n_history
        
        print(f"Running rollout prediction:")
        print(f"  Total timesteps: {states.shape[0]}")
        print(f"  History length: {self.n_history}")
        print(f"  Rollout steps: {n_rollout}")
        print(f"  Particles: {particle_num}")
        
        # Initialize rollout with history data (add batch dimension)
        predictor = Rollout(
            self.model, 
            self.config,
            states[self.n_history - 1].unsqueeze(0),      # [1, particles, 3]
            states_delta[:self.n_history - 1].unsqueeze(0), # [1, n_history - 1, particles, 3] 
            attrs[self.n_history - 1].unsqueeze(0),       # [1, particles]
            torch.tensor([particle_num], device=self.device),  # [1]
            topological_edges.unsqueeze(0),  # [1, particles, particles]
            first_states.unsqueeze(0)  # [1, particles, 3]
        )
        
        # Initialize prediction storage with history
        predicted_states = [states[i].clone() for i in range(self.n_history)]
        
        # Run autoregressive rollout
        for i in range(n_rollout):
            step_idx = i + self.n_history
            
            # Get next frame data and add batch dimension
            next_delta = states_delta[step_idx - 1].unsqueeze(0)     # [1, particles, 3]
            
            # Predict next state and remove batch dimension
            predicted_state = predictor.forward(next_delta)[0]  # [particles, 3]
            predicted_states.append(predicted_state)
        
        # Stack predictions and return [timesteps, particles, 3]
        return torch.stack(predicted_states)

    def calculate_prediction_error(self, predicted_states, actual_states):
        """
        Calculate MSE errors between predicted and ground truth trajectories.
        
        Args:
            predicted_states: [timesteps, particles, 3] - predicted trajectory
            actual_states: [timesteps, particles, 3] - ground truth trajectory
            
        Returns:
            errors: list[float] - MSE error for each timestep
        """
        errors = []
        n_timesteps = min(len(predicted_states), len(actual_states))
        
        for t in range(n_timesteps):
            error = torch.nn.functional.mse_loss(predicted_states[t], actual_states[t]).item()
            errors.append(error)
            
        return errors
        
    def visualize_object_motion(self, predicted_states, tool_mask, actual_objects, 
                               episode_num, save_path, topological_edges=None):
        """
        Create 3D visualization comparing predicted vs actual object motion.
        Renders particles as colored point clouds with connectivity edges.
        
        Args:
            predicted_states: [timesteps, total_particles, 3] tensor - predicted trajectory (objects + robots)
            tool_mask: [total_particles] tensor - boolean mask (False=object, True=robot)
            actual_objects: [timesteps, n_obj, 3] tensor - ground truth object trajectory  
            episode_num: int - episode number for labeling
            save_path: str - output video file path
            topological_edges: [N, N] tensor - topological edges for object and robot particles
            
        Returns:
            save_path: str - path where video was saved
        """
        print(f"Creating visualization for episode {episode_num}...")
        
        # Video parameters
        width, height = self.WH
        fps = self.FPS
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        out = cv2.VideoWriter(save_path, fourcc, fps, (width, height))
        
        # Create Open3D visualizer
        vis = o3d.visualization.Visualizer()
        vis.create_window(width=width, height=height, visible=False)
        
        render_option = vis.get_render_option()
        render_option.point_size = 10.0

        # Move tensors to CPU
        predicted_states = predicted_states.cpu()
        tool_mask = tool_mask.cpu()
        topological_edges = topological_edges.cpu()
        
        # Split predicted states into objects and robots using tool_mask. Convert to numpy for o3d. Keep predicted_states as tensor. 
        actual_objects = actual_objects.cpu().numpy()
        pred_objects = predicted_states[:, ~tool_mask, :].numpy()
        pred_robots = predicted_states[:, tool_mask, :].numpy()
        
        # Create point clouds            
        actual_pcd = o3d.geometry.PointCloud()
        actual_pcd.points = o3d.utility.Vector3dVector(actual_objects[0])
        actual_pcd.paint_uniform_color([0.0, 0.0, 1.0])  # Blue for actual
            
        robot_pcd = o3d.geometry.PointCloud()
        robot_pcd.points = o3d.utility.Vector3dVector(pred_robots[0])
        robot_pcd.paint_uniform_color([1.0, 0.0, 0.0])  # Red for robot
        
        pred_pcd = o3d.geometry.PointCloud()
        pred_pcd.points = o3d.utility.Vector3dVector(pred_objects[0])
        pred_pcd.paint_uniform_color([0.0, 1.0, 0.0])  # Green for predicted

        # Add geometries to visualizer
        vis.add_geometry(actual_pcd)
        vis.add_geometry(robot_pcd)
        vis.add_geometry(pred_pcd)

        n_frames = min(len(predicted_states), len(actual_objects))
        print(f"Rendering {n_frames} frames...")
        
        # Set up camera parameters if available
        if self.w2cs is not None:
            view_control = vis.get_view_control()
            camera_params = o3d.camera.PinholeCameraParameters()
            intrinsic_parameter = o3d.camera.PinholeCameraIntrinsic(
                width, height, self.intrinsics[0]  # Using first camera
            )
            camera_params.intrinsic = intrinsic_parameter
            camera_params.extrinsic = self.w2cs[0]
            view_control.convert_from_pinhole_camera_parameters(
                camera_params, allow_arbitrary=True
            )

        # Initialize edge sets
        pred_line_sets = []

        # Render each frame
        for frame_idx in range(n_frames):
            # Update particle positions
            pred_obj_pos = pred_objects[frame_idx]
            actual_obj_pos = actual_objects[frame_idx]
            robot_pos = pred_robots[frame_idx]
            
            if frame_idx == 0:
                print(f"Sampled object particles: {pred_obj_pos.shape[0]}")
                print(f"Robot particles: {robot_pos.shape[0]}")
            
            # Update point cloud positions
            pred_pcd.points = o3d.utility.Vector3dVector(pred_obj_pos)
            robot_pcd.points = o3d.utility.Vector3dVector(robot_pos)
            actual_pcd.points = o3d.utility.Vector3dVector(actual_obj_pos)

            vis.update_geometry(pred_pcd)
            vis.update_geometry(actual_pcd)
            vis.update_geometry(robot_pcd)

            # Remove old edges
            for line_set in pred_line_sets:
                if line_set is not None:
                    vis.remove_geometry(line_set, reset_bounding_box=False)
            
            pred_line_sets = visualize_edges(
                predicted_states[frame_idx], topological_edges, tool_mask, 
                self.adj_thresh, self.topk, False, 
                [[1.0, 0.6, 0.2], [0.3, 0.6, 0.3]]  # light orange, light green
            )
            for line_set in pred_line_sets:
                if line_set is not None:
                    vis.add_geometry(line_set, reset_bounding_box=False)
                        
            # Render frame
            vis.poll_events()
            vis.update_renderer()
            
            static_image = np.asarray(
                vis.capture_screen_float_buffer(do_render=True)
            )
            static_image = (static_image * 255).astype(np.uint8)
            static_image_bgr = cv2.cvtColor(static_image, cv2.COLOR_RGB2BGR)
            out.write(static_image_bgr)
                        
            if frame_idx % 10 == 0:
                print(f"  Rendered frame {frame_idx}/{n_frames}")
        
        # Cleanup
        out.release()
        vis.destroy_window()
        
        print(f"Video saved to: {save_path}")
        return save_path

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def main():
    """
    Main function to test object motion prediction on specified episodes.
    Loads trained model, runs rollout predictions, and optionally generates visualizations.
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True,
                       help="Model name (e.g., 2025-05-31-21-01-09-427982 or custom_model_name)")
    parser.add_argument("--camera_calib_path", type=str, default="PhysTwin/data/different_types/single_push_rope")
    parser.add_argument("--episodes", nargs='+', type=str, default=["0-4"],
                       help="Episodes to test. Format: space-separated list (0 1 2 3 4) or range (0-4)")
    parser.add_argument("--data_file", type=str, default=None)
    parser.add_argument("--video", action='store_true',
                       help="Generate visualization videos (optional)")
    parser.add_argument("--config_path", type=str, default=None)
    args = parser.parse_args()
    
    # Parse episode specification
    try:
        test_episodes = parse_episodes(args.episodes)
    except ValueError as e:
        print(f"Error parsing episodes: {e}")
        print("Examples:")
        print("  Space-separated: --episodes 0 1 2 3 4")
        print("  Range format: --episodes 0-4")
        return
    
    # Setup file paths
    model_paths = get_model_paths(args.model)
    model_path = str(model_paths['net_best'])
    if args.config_path is None:
        config_path = str(model_paths['config'])
    else:
        config_path = args.config_path
        
    data_file = args.data_file 
    camera_calib_path = args.camera_calib_path
    
    # Initialize visualizer
    visualizer = Visualizer(model_path, config_path, camera_calib_path, data_file)
    
    print("="*60)
    print("OBJECT MOTION PREDICTION TEST")
    print("="*60)
    print(f"Testing episodes: {test_episodes}")
    print(f"Video generation: {'Enabled' if args.video else 'Disabled'}")
    
    all_errors = []
    
    # Test each episode
    for episode_num in test_episodes:
        print(f"\n{'='*40}")
        print(f"TESTING EPISODE {episode_num}")
        print(f"{'='*40}")
        
        with torch.no_grad():
            # Load episode data
            states, states_delta, attrs, particle_num, topological_edges, first_states = visualizer.dataset.load_full_episode(episode_num)
            states = states.to(visualizer.device)
            states_delta = states_delta.to(visualizer.device)
            attrs = attrs.to(visualizer.device)
            topological_edges = topological_edges.to(visualizer.device)
            first_states = first_states.to(visualizer.device)

            # Determine particle counts from attributes (0=object, 1=robot)
            n_obj = (attrs[0] == 0).sum().item()
            n_robot = (attrs[0] == 1).sum().item()
            
            print(f"Loaded episode with {n_obj} object particles, {n_robot} robot particles")
            print(f"Topological edges loaded: {topological_edges.sum():.0f} edges")
                        
            translation = torch.ones_like(states) * 1000
            states = states + translation

            # Run autoregressive rollout prediction
            predicted_states = visualizer.predict_episode_rollout(states, states_delta, attrs, particle_num, topological_edges, first_states)
            
            states = states - translation
            predicted_states = predicted_states - translation

            # Split into object and robot components for evaluation
            predicted_objects = predicted_states[:, :n_obj, :]
            actual_objects = states[:, :n_obj, :]
            
            # Calculate prediction errors (evaluate object motion only)
            errors = visualizer.calculate_prediction_error(predicted_objects, actual_objects)
            all_errors.extend(errors)
            
            avg_error = np.mean(errors)
            print(f"RMSE: {np.sqrt(avg_error):.6f} | Timesteps predicted: {len(errors)}")
            
            # Generate video only if --video flag is provided
            if args.video:
                tool_mask = (attrs[0] > 0.5).bool()  # attrs is 0 for objects, 1 for robots; use first timestep only
                video_path = str(model_paths['video_dir'] / f"prediction_{episode_num}.mp4")
                visualizer.visualize_object_motion(
                    predicted_states, tool_mask, actual_objects, 
                    episode_num, video_path, topological_edges
                )
                
    # Overall results
    if all_errors:
        print(f"\n{'='*60}")
        print("OVERALL RESULTS")
        print(f"{'='*60}")
        print(f"Episodes tested: {len(test_episodes)}")
        print(f"Total timesteps: {len(all_errors)}")
        print(f"Average MSE: {np.mean(all_errors):.6f}")
        print(f"RMSE: {np.sqrt(np.mean(all_errors)):.6f}")
        print(f"Std Dev: {np.std(all_errors):.6f}")
        
        # Provide performance interpretation
        rmse = np.sqrt(np.mean(all_errors))
        if rmse < 0.003:
            print("🎉 Excellent prediction accuracy!")
        elif rmse < 0.01:
            print("✅ Good prediction accuracy!")
        elif rmse < 0.05:
            print("⚠️  Moderate prediction accuracy")
        else:
            print("❌ Poor prediction accuracy - model needs improvement")
        
        if args.video:
            print(f"\nVideos saved in '{model_paths['video_dir']}' directory")
        else:
            print(f"\nTo generate videos, rerun with --video flag")

if __name__ == "__main__":
    main()
