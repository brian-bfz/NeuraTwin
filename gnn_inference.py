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
    def __init__(self, model_path, config_path, camera_calib_path, data_root):
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
        self.FPS = data["fps"]
        
        # Initialize dataset for consistent data loading
        self.dataset = ParticleDataset(data_root, self.config, 'train')  # Use train to access all episodes
        
        # CRITICAL: Use same parameters as training!
        self.fps_radius = self.config['train']['fps_radius']  # 0.03
        self.adj_thresh = self.config['train']['particle']['adj_thresh']
        
        # Load trained model
        self.model = PropNetDiffDenModel(self.config, torch.cuda.is_available())
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        
        print(f"Loaded model from: {model_path}")
        print(f"Using device: {self.device}")
        print(f"Dataset initialized with {self.dataset.n_episode} episodes")
        
    def predict_object_response(self, current_object_pos, current_robot_pos, robot_delta):
        """
        Predict how objects respond to robot motion
        
        Args:
        - current_object_pos: [N_obj, 3] current object positions
        - current_robot_pos: [N_robot, 3] current robot positions  
        - robot_delta: [N_robot, 3] robot motion for this timestep
        
        Returns:
        - next_object_pos: [N_obj, 3] predicted next object positions
        """
        
        N_obj = current_object_pos.shape[0]
        N_robot = current_robot_pos.shape[0]
        N_total = N_obj + N_robot
        
        # Combine object and robot particles (model expects both)
        s_cur = torch.cat([current_object_pos, current_robot_pos], dim=0).unsqueeze(0)  # [1, N_total, 3]
        
        # Create attributes: 0 for objects, 1 for robot
        a_cur = torch.zeros(1, N_total, device=self.device)
        a_cur[0, N_obj:] = 1.0  # Robot particles have attr=1
        
        # Create motion deltas: 0 for objects, actual motion for robot
        s_delta = torch.zeros(1, N_total, 3, device=self.device)
        s_delta[0, N_obj:] = robot_delta.unsqueeze(0)  # Only robot moves
        
        # Move to device
        s_cur = s_cur.to(self.device)
        
        # Predict next positions
        with torch.no_grad():
            s_next_pred = self.model.predict_one_step(a_cur, s_cur, s_delta)
        
        # Extract only object predictions
        next_object_pos = s_next_pred[0, :N_obj, :].cpu()  # [N_obj, 3]
        
        return next_object_pos
    
    def predict_object_motion(self, initial_object_pos, robot_trajectory):
        """
        Given initial object positions and robot trajectory, predict object motion over time
        
        Args:
        - initial_object_pos: [N_obj, 3] starting object positions
        - robot_trajectory: [timesteps, N_robot, 3] robot motion sequence
        
        Returns:
        - object_predictions: [timesteps, N_obj, 3] predicted object motion
        """
        
        object_predictions = [initial_object_pos.clone()]
        current_object_pos = initial_object_pos.clone()
        
        print(f"Predicting object motion for {robot_trajectory.shape[0]-1} timesteps...")
        
        for t in range(robot_trajectory.shape[0] - 1):
            # Robot motion between timesteps
            current_robot_pos = robot_trajectory[t]
            next_robot_pos = robot_trajectory[t + 1]
            robot_delta = next_robot_pos - current_robot_pos
            
            # Predict how objects respond to this robot motion
            next_object_pos = self.predict_object_response(
                current_object_pos, 
                current_robot_pos, 
                robot_delta
            )
            
            object_predictions.append(next_object_pos)
            current_object_pos = next_object_pos
                    
        return torch.stack(object_predictions)
    
    def calculate_prediction_error(self, predicted_objects, actual_objects):
        """Calculate prediction errors"""
        errors = []
        for t in range(min(len(predicted_objects), len(actual_objects))):
            error = torch.nn.functional.mse_loss(predicted_objects[t], actual_objects[t]).item()
            errors.append(error)
        return errors
    
    def visualize_object_motion(self, predicted_objects, actual_objects, robot_trajectory, 
                               full_objects, episode_num, save_path):
        """
        Create visualization comparing predicted vs actual object motion
        Uses the existing visualizer pattern
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

        # Full ground truth object point cloud (yellow in BGR: [0, 1, 1])
        # full_gt_pcd = o3d.geometry.PointCloud()
        # full_gt_pcd.points = o3d.utility.Vector3dVector(full_objects[0].numpy())
        # full_gt_pcd.paint_uniform_color([0.0, 1.0, 1.0])  # Yellow for full GT (BGR)

        # Add full GT first so it appears at the bottom
        # vis.add_geometry(full_gt_pcd)
        vis.add_geometry(actual_pcd)
        vis.add_geometry(robot_pcd)
        vis.add_geometry(pred_pcd)

        n_frames = min(len(predicted_objects), len(actual_objects), len(full_objects))
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
            # full_gt_pos = full_objects[frame_idx].numpy()
            
            if frame_idx == 0:
                print(f"Sampled object particles: {actual_obj_pos.shape[0]}")
                # print(f"Full GT object particles: {full_gt_pos.shape[0]}")
                print(f"Robot particles: {robot_pos.shape[0]}")
            
            pred_pcd.points = o3d.utility.Vector3dVector(pred_obj_pos)
            robot_pcd.points = o3d.utility.Vector3dVector(robot_pos)
            actual_pcd.points = o3d.utility.Vector3dVector(actual_obj_pos)
            # full_gt_pcd.points = o3d.utility.Vector3dVector(full_gt_pos)

            vis.update_geometry(pred_pcd)
            vis.update_geometry(actual_pcd)
            vis.update_geometry(robot_pcd)
            # vis.update_geometry(full_gt_pcd)

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
            
            # # Add text overlay with sampling info
            # text1 = f'Frame {frame_idx} | Red=Predicted Objects | Blue=Actual Objects | Green=Robot'
            # text2 = f'Sampled particles (FPS radius={self.fps_radius}) | Edge threshold={self.adj_thresh}'
            # image_bgr = cv2.putText(image_bgr, text1, (10, 30), 
            #                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            # image_bgr = cv2.putText(image_bgr, text2, (10, 60), 
            #                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                        
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
    parser.add_argument("--name", type=str, default="2025-05-31-21-01-09-427982",
                       help="Model name (e.g., 2025-05-31-21-01-09-427982 or custom_model_name)")
    parser.add_argument("--camera_calib_path", type=str, default="data/single_push_rope")
    parser.add_argument("--episodes", nargs='+', type=str, default=["0-4"],
                       help="Episodes to test. Format: space-separated list (0 1 2 3 4) or range (0-4)")
    parser.add_argument("--data_root", type=str, default="../test/PhysTwin/generated_data")
    parser.add_argument("--video", action='store_true',
                       help="Generate visualization videos (optional)")
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
    
    model_path = f"data/gnn_dyn_model/{args.name}/net_best.pth"
    config_path = "config/train/gnn_dyn.yaml"
    data_root = args.data_root
    camera_calib_path = args.camera_calib_path
    
    # Initialize predictor with camera calibration
    predictor = ObjectMotionPredictor(model_path, config_path, camera_calib_path, data_root)
    
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
            # Load episode data with FPS sampling
            actual_objects, robot_trajectory, full_object_trajectory, obj_indices, robot_indices = predictor.dataset.load_full_episode(episode_num)
            
            # Predict object motion starting from initial positions
            initial_object_pos = actual_objects[0]
            predicted_objects = predictor.predict_object_motion(initial_object_pos, robot_trajectory)
            
            # Calculate prediction errors
            errors = predictor.calculate_prediction_error(predicted_objects, actual_objects)
            all_errors.extend(errors)
            
            avg_error = np.mean(errors)
            print(f"Average MSE: {avg_error:.6f} | RMSE: {np.sqrt(avg_error):.6f} | Timesteps predicted: {len(errors)}")
            
            # Generate video only if --video flag is provided
            if args.video:
                video_path = f"data/video/{args.name}/prediction_{episode_num}.mp4"
                predictor.visualize_object_motion(
                    predicted_objects, actual_objects, robot_trajectory, 
                    full_object_trajectory, episode_num, video_path
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
            print(f"\nVideos saved in 'data/video/{args.name}' directory")
        else:
            print(f"\nTo generate videos, rerun with --video flag")

if __name__ == "__main__":
    main()
