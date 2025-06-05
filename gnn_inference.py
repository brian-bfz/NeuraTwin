import os
import numpy as np
import torch
import cv2
import open3d as o3d
import pickle
import json
from dataset.dataset_gnn_dyn import ParticleDataset
from model.gnn_dyn import PropNetDiffDenModel
from utils import load_yaml, fps_rad_tensor
import argparse

class ObjectMotionPredictor:
    def __init__(self, model_path, config_path, camera_calib_path, data_file):
        """Initialize the object motion predictor"""
        self.config = load_yaml(config_path)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load camera calibration
        with open(os.path.join(camera_calib_path, "calibrate.pkl"), "rb") as f:
            self.c2ws = pickle.load(f)
        self.w2cs = [np.linalg.inv(c2w) for c2w in self.c2ws]
            
        with open(os.path.join(camera_calib_path, "metadata.json"), "r") as f:
            data = json.load(f)
        self.intrinsics = np.array(data["intrinsics"])
        self.WH = data["WH"]
        self.FPS = data["fps"] / self.config['dataset']['downsample_rate']
        
        # Initialize dataset for consistent data loading
        self.dataset = ParticleDataset(data_file, self.config, 'train')  # Use train to access all episodes
        
        # Get training parameters for consistency
        self.n_history = self.config['train']['n_history']
        self.fps_radius = self.config['train']['fps_radius']
        self.adj_thresh = self.config['train']['particle']['adj_thresh']
        
        # Load trained model
        self.model = PropNetDiffDenModel(self.config, torch.cuda.is_available())
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        
        print(f"Loaded model from: {model_path}")
        print(f"Using device: {self.device}")
        print(f"Dataset initialized with {self.dataset.n_episode} episodes")
        print(f"History length: {self.n_history}")
    
    def predict_episode_rollout(self, states, states_delta, attrs, particle_num):
        """
        Predict object motion for entire episode using rollout approach from training
        
        Args:
            states: [timesteps, particles, 3] - full episode states with history padding
            states_delta: [timesteps-1, particles, 3] - state deltas
            attrs: [timesteps, particles] - particle attributes
            particle_num: total number of particles
            
        Returns:
            predicted_states: [timesteps, particles, 3] - predicted trajectory
        """
        
        # Move data to device
        states = states.to(self.device)
        states_delta = states_delta.to(self.device) 
        attrs = attrs.to(self.device)
        
        # Calculate n_rollout from episode length
        n_rollout = states.shape[0] - self.n_history
        
        print(f"Running rollout prediction:")
        print(f"  Total timesteps: {states.shape[0]}")
        print(f"  History length: {self.n_history}")
        print(f"  Rollout steps: {n_rollout}")
        print(f"  Particles: {particle_num}")
        
        # Add batch dimension for compatibility with model
        states = states.unsqueeze(0)  # [1, timesteps, particles, 3]
        states_delta = states_delta.unsqueeze(0)  # [1, timesteps-1, particles, 3]
        attrs = attrs.unsqueeze(0)  # [1, timesteps, particles]
        particle_nums = torch.tensor([particle_num], dtype=torch.int32)
        
        # Initialize prediction storage
        predicted_states = [states[0, i].clone() for i in range(self.n_history)]  # Start with history
        
        # Initialize history buffer for rollout (same as training)
        history_buffer_states = states[:, :self.n_history, :, :].clone()  # [1, n_history, particles, 3]
        history_buffer_delta = states_delta[:, :self.n_history, :, :].clone()  # [1, n_history, particles, 3]
        
        with torch.no_grad():
            for idx_step in range(n_rollout):
                # Extract current history window (same as training)
                a_hist = attrs[:, idx_step:idx_step + self.n_history, :]  # [1, n_history, particles]
                s_hist = history_buffer_states  # [1, n_history, particles, 3]
                s_delta_hist = history_buffer_delta  # [1, n_history, particles, 3]
                
                # Predict next state using model
                s_pred = self.model.predict_one_step(a_hist, s_hist, s_delta_hist, particle_nums)  # [1, particles, 3]
                
                # Store prediction
                predicted_states.append(s_pred[0].clone().cpu())
                
                # Update history buffer for next rollout step (same as training)
                if idx_step < n_rollout - 1:  # Don't update on last step
                    # Update delta buffer with predicted particle deltas
                    history_buffer_delta[:, -1, :, :] = s_pred - history_buffer_states[:, -1, :, :]
                    
                    # Remove oldest frame, add new frame's robot deltas
                    history_buffer_delta = torch.cat([
                        history_buffer_delta[:, 1:, :, :],  # Remove first frame
                        states_delta[:, idx_step + self.n_history, :, :].unsqueeze(1)  # Add new delta
                    ], dim=1)
                    
                    # Remove oldest frame, add new prediction
                    history_buffer_states = torch.cat([
                        history_buffer_states[:, 1:, :, :],  # Remove first frame  
                        s_pred.unsqueeze(1)  # Add prediction as new frame
                    ], dim=1)
        
        # Stack predictions and return [timesteps, particles, 3]
        return torch.stack(predicted_states)
    
    def calculate_prediction_error(self, predicted_states, actual_states):
        """Calculate prediction errors between predicted and actual trajectories"""
        errors = []
        n_timesteps = min(len(predicted_states), len(actual_states))
        
        for t in range(n_timesteps):
            error = torch.nn.functional.mse_loss(predicted_states[t], actual_states[t]).item()
            errors.append(error)
            
        return errors
    
    def split_object_robot_states(self, states, n_obj_particles):
        """Split combined states into object and robot trajectories"""
        object_states = states[:, :n_obj_particles, :]  # [timesteps, n_obj, 3]
        robot_states = states[:, n_obj_particles:, :]   # [timesteps, n_robot, 3]
        return object_states, robot_states
    
    def visualize_object_motion(self, predicted_objects, actual_objects, robot_trajectory, 
                               episode_num, save_path):
        """
        Create visualization comparing predicted vs actual object motion
        Uses the existing visualizer pattern
        
        Args:
            predicted_objects: [timesteps, n_obj, 3] predicted object trajectory
            actual_objects: [timesteps, n_obj, 3] actual object trajectory  
            robot_trajectory: [timesteps, n_robot, 3] robot trajectory
            episode_num: episode number for labeling
            save_path: path to save video
        """
        
        print(f"Creating visualization for episode {episode_num}...")
        
        # Video parameters
        width, height = self.WH
        fps = self.FPS
        fourcc = cv2.VideoWriter_fourcc(*'avc1')  # Same as v_from_d.py
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        out = cv2.VideoWriter(save_path, fourcc, fps, (width, height))
        
        # Create Open3D visualizer
        vis = o3d.visualization.Visualizer()
        vis.create_window(width=width, height=height, visible=False)
        
        # Set render options
        render_option = vis.get_render_option()
        render_option.point_size = 10.0
        
        # Create point clouds            
        actual_pcd = o3d.geometry.PointCloud()
        actual_pcd.points = o3d.utility.Vector3dVector(actual_objects[0].numpy())
        actual_pcd.paint_uniform_color([0.0, 0.0, 1.0])  # Red for actual
            
        robot_pcd = o3d.geometry.PointCloud()
        robot_pcd.points = o3d.utility.Vector3dVector(robot_trajectory[0].numpy())
        robot_pcd.paint_uniform_color([0.0, 1.0, 0.0])  # Green for robot
        
        pred_pcd = o3d.geometry.PointCloud()
        pred_pcd.points = o3d.utility.Vector3dVector(predicted_objects[0].numpy())
        pred_pcd.paint_uniform_color([1.0, 0.0, 0.0])  # Blue for predicted

        # Add geometries to visualizer
        vis.add_geometry(actual_pcd)
        vis.add_geometry(robot_pcd)
        vis.add_geometry(pred_pcd)

        n_frames = min(len(predicted_objects), len(actual_objects), len(robot_trajectory))
        print(f"Rendering {n_frames} frames...")
        
        def create_edges_for_points(positions, distance_threshold):
            """Create edges between points within distance threshold"""
            edges = []
            n_points = positions.shape[0]
            
            for i in range(n_points):
                for j in range(i + 1, n_points):
                    distance = np.linalg.norm(positions[i] - positions[j])
                    if distance <= distance_threshold:
                        edges.append([i, j])
            
            return np.array(edges) if edges else np.empty((0, 2), dtype=int)
        
        # Set up camera parameters if available
        if self.w2cs is not None:
            view_control = vis.get_view_control()
            camera_params = o3d.camera.PinholeCameraParameters()
            intrinsic_parameter = o3d.camera.PinholeCameraIntrinsic(
                width, height, self.intrinsics[0]  # Using first camera
            )
            camera_params.intrinsic = intrinsic_parameter
            camera_params.extrinsic = self.w2cs[0]  # Using first camera
            view_control.convert_from_pinhole_camera_parameters(
                camera_params, allow_arbitrary=True
            )
        # Print camera parameters for debugging
        print("[gnn_inference] Camera intrinsic:")
        print(self.intrinsics[0])
        print("[gnn_inference] Camera extrinsic (w2c):")
        print(self.w2cs[0])

        pred_line_set = None
        actual_line_set = None

        
        for frame_idx in range(n_frames):
            # Get positions for this frame
            pred_obj_pos = predicted_objects[frame_idx].numpy()
            actual_obj_pos = actual_objects[frame_idx].numpy()
            robot_pos = robot_trajectory[frame_idx].numpy()
            
            if frame_idx == 0:
                print(f"Sampled object particles: {actual_obj_pos.shape[0]}")
                print(f"Robot particles: {robot_pos.shape[0]}")
            
            pred_pcd.points = o3d.utility.Vector3dVector(pred_obj_pos)
            robot_pcd.points = o3d.utility.Vector3dVector(robot_pos)
            actual_pcd.points = o3d.utility.Vector3dVector(actual_obj_pos)

            vis.update_geometry(pred_pcd)
            vis.update_geometry(actual_pcd)
            vis.update_geometry(robot_pcd)

            # Remove old edges
            if pred_line_set is not None:
                vis.remove_geometry(pred_line_set, reset_bounding_box=False)
            if actual_line_set is not None:
                vis.remove_geometry(actual_line_set, reset_bounding_box=False)
    
            # Create new edges
            pred_edges = create_edges_for_points(pred_obj_pos, self.adj_thresh)
            if len(pred_edges) > 0:
                pred_line_set = o3d.geometry.LineSet()
                pred_line_set.points = o3d.utility.Vector3dVector(pred_obj_pos)
                pred_line_set.lines = o3d.utility.Vector2iVector(pred_edges)
                pred_line_colors = np.tile([1.0, 0.0, 0.0], (len(pred_edges), 1)) # Blue for predicted
                pred_line_set.colors = o3d.utility.Vector3dVector(pred_line_colors)
                vis.add_geometry(pred_line_set, reset_bounding_box=False)   
            else:
                pred_line_set = None            

            actual_edges = create_edges_for_points(actual_obj_pos, self.adj_thresh)
            if len(actual_edges) > 0:
                actual_line_set = o3d.geometry.LineSet()
                actual_line_set.points = o3d.utility.Vector3dVector(actual_obj_pos)
                actual_line_set.lines = o3d.utility.Vector2iVector(actual_edges)
                actual_line_colors = np.tile([0.0, 0.0, 1.0], (len(actual_edges), 1)) # Red for actual
                actual_line_set.colors = o3d.utility.Vector3dVector(actual_line_colors)
                vis.add_geometry(actual_line_set, reset_bounding_box=False)
            else:
                actual_line_set = None
                        
            # Render frame
            vis.poll_events()
            vis.update_renderer()
            
            static_image = np.asarray(
                vis.capture_screen_float_buffer(do_render=True)
            )
            static_image = (static_image * 255).astype(np.uint8)

            out.write(static_image)
                        
            if frame_idx % 10 == 0:
                print(f"  Rendered frame {frame_idx}/{n_frames}")
        
        out.release()
        vis.destroy_window()
        
        print(f"Video saved to: {save_path}")
        return save_path

def parse_episodes(episodes_arg):
    """
    Parse episode specification that supports two formats:
    1. Space-separated list: ['0', '1', '2', '3', '4'] -> [0, 1, 2, 3, 4]
    2. Range format: ['0-4'] -> [0, 1, 2, 3, 4] (inclusive)
    
    Args:
        episodes_arg: List of strings from argparse
    
    Returns:
        List of episode numbers
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

def main():
    """Main function to test object motion prediction"""
    
    # Configuration
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="2025-05-31-21-01-09-427982",
                       help="Model name (e.g., 2025-05-31-21-01-09-427982 or custom_model_name)")
    parser.add_argument("--camera_calib_path", type=str, default="data/single_push_rope")
    parser.add_argument("--episodes", nargs='+', type=str, default=["0-4"],
                       help="Episodes to test. Format: space-separated list (0 1 2 3 4) or range (0-4)")
    parser.add_argument("--data_file", type=str, default="../PhysTwin/generated_data/less_empty_data.h5")
    parser.add_argument("--video", action='store_true',
                       help="Generate visualization videos (optional)")
    parser.add_argument("--config_path", type=str, default=None)
    args = parser.parse_args()
    
    # Parse episodes from the flexible input format
    try:
        test_episodes = parse_episodes(args.episodes)
    except ValueError as e:
        print(f"Error parsing episodes: {e}")
        print("Examples:")
        print("  Space-separated: --episodes 0 1 2 3 4")
        print("  Range format: --episodes 0-4")
        return
    
    model_path = f"data/gnn_dyn_model/{args.model}/net_best.pth"
    if args.config_path is None:
        config_path = f"data/gnn_dyn_model/{args.model}/config.yaml"
    else:
        config_path = args.config_path
    data_file = args.data_file 
    camera_calib_path = args.camera_calib_path
    
    # Initialize predictor with camera calibration
    predictor = ObjectMotionPredictor(model_path, config_path, camera_calib_path, data_file)
    
    print("="*60)
    print("OBJECT MOTION PREDICTION TEST")
    print("="*60)
    print(f"Testing episodes: {test_episodes}")
    print(f"Video generation: {'Enabled' if args.video else 'Disabled'}")
    
    all_errors = []
    
    for episode_num in test_episodes:
        print(f"\n{'='*40}")
        print(f"TESTING EPISODE {episode_num}")
        print(f"{'='*40}")
        
        try:
            # Load episode data in the new format
            states, states_delta, attrs, particle_num = predictor.dataset.load_full_episode(episode_num)
            
            # Extract number of object particles from attrs (0=object, 1=robot)
            n_obj_particles = (attrs[0] == 0).sum().item()
            n_robot_particles = (attrs[0] == 1).sum().item()
            
            print(f"Loaded episode with {n_obj_particles} object particles, {n_robot_particles} robot particles")
            
            # Run rollout prediction
            predicted_states = predictor.predict_episode_rollout(states, states_delta, attrs, particle_num)
            
            # Split predictions and ground truth into object/robot components
            predicted_objects, predicted_robots = predictor.split_object_robot_states(predicted_states, n_obj_particles)
            actual_objects, actual_robots = predictor.split_object_robot_states(states, n_obj_particles)
            
            # Calculate prediction errors (only for object particles)
            errors = predictor.calculate_prediction_error(predicted_objects, actual_objects)
            all_errors.extend(errors)
            
            avg_error = np.mean(errors)
            print(f"Average MSE: {avg_error:.6f} | RMSE: {np.sqrt(avg_error):.6f} | Timesteps predicted: {len(errors)}")
            
            # Generate video only if --video flag is provided
            if args.video:
                video_path = f"data/video/{args.model}/prediction_{episode_num}.mp4"
                predictor.visualize_object_motion(
                    predicted_objects, actual_objects, actual_robots, 
                    episode_num, video_path
                )
            
        except Exception as e:
            print(f"Error processing episode {episode_num}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
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
        
        # Interpretation
        rmse = np.sqrt(np.mean(all_errors))
        if rmse < 0.01:
            print("🎉 Excellent prediction accuracy!")
        elif rmse < 0.05:
            print("✅ Good prediction accuracy!")
        elif rmse < 0.1:
            print("⚠️  Moderate prediction accuracy")
        else:
            print("❌ Poor prediction accuracy - model needs improvement")
        
        if args.video:
            print(f"\nVideos saved in 'data/video/{args.model}' directory")
        else:
            print(f"\nTo generate videos, rerun with --video flag")

if __name__ == "__main__":
    main()
