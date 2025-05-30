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

class ObjectMotionPredictor:
    def __init__(self, model_path, config_path, camera_calib_path):
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
        
        # CRITICAL: Use same FPS radius as training!
        self.fps_radius = self.config['train']['fps_radius']  # 0.03
        self.adj_thresh = self.config['train']['particle']['adj_thresh']
        
        # Load trained model
        self.model = PropNetDiffDenModel(self.config, torch.cuda.is_available())
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        
        print(f"Loaded model from: {model_path}")
        print(f"Using device: {self.device}")
    
    def load_episode_data(self, data_dir, episode_num):
        """
        Load raw episode data, apply FPS sampling, and separate object/robot particles
        
        Returns:
        - sampled_object_trajectory: [timesteps, N_sampled_obj, 3] - actual object motion (sampled)
        - sampled_robot_trajectory: [timesteps, N_sampled_robot, 3] - robot motion (sampled)
        - object_sample_indices: indices of sampled object particles
        - robot_sample_indices: indices of sampled robot particles
        """
        
        # Find number of timesteps for this episode
        timestep = 0
        while os.path.exists(f'{data_dir}/{episode_num}/object/x_{timestep}.pt'):
            timestep += 1
        n_timesteps = timestep
        
        print(f"Episode {episode_num}: Found {n_timesteps} timesteps")
        
        # Load first timestep to determine sampling indices
        first_object = torch.load(f'{data_dir}/{episode_num}/object/x_0.pt')
        first_robot = torch.load(f'{data_dir}/{episode_num}/robot/x_0.pt')
        
        print(f"Raw particles - Object: {first_object.shape[0]}, Robot: {first_robot.shape[0]}")
        
        # CRITICAL FIX: Apply FPS sampling (same as training!)
        object_sample_indices = fps_rad_tensor(first_object, self.fps_radius)
        robot_sample_indices = fps_rad_tensor(first_robot, self.fps_radius)
        
        n_sampled_obj = object_sample_indices.shape[0]
        n_sampled_robot = robot_sample_indices.shape[0]
        
        print(f"Sampled particles - Object: {n_sampled_obj}, Robot: {n_sampled_robot}")
        print(f"Sampling ratio - Object: {n_sampled_obj/first_object.shape[0]:.3f}, Robot: {n_sampled_robot/first_robot.shape[0]:.3f}")
        
        # Pre-allocate arrays for sampled particles
        sampled_object_trajectory = torch.zeros(n_timesteps, n_sampled_obj, 3)
        sampled_robot_trajectory = torch.zeros(n_timesteps, n_sampled_robot, 3)
        
        # Load and sample all timesteps
        for t in range(n_timesteps):
            # Load full data
            object_full = torch.load(f'{data_dir}/{episode_num}/object/x_{t}.pt')
            robot_full = torch.load(f'{data_dir}/{episode_num}/robot/x_{t}.pt')
            
            # Apply consistent sampling (same indices for all timesteps)
            sampled_object_trajectory[t] = object_full[object_sample_indices]
            sampled_robot_trajectory[t] = robot_full[robot_sample_indices]
        
        return sampled_object_trajectory, sampled_robot_trajectory, object_sample_indices, robot_sample_indices
    
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
            
            if t % 10 == 0:
                print(f"  Predicted timestep {t+1}/{robot_trajectory.shape[0]-1}")
        
        return torch.stack(object_predictions)
    
    def calculate_prediction_error(self, predicted_objects, actual_objects):
        """Calculate prediction errors"""
        errors = []
        for t in range(min(len(predicted_objects), len(actual_objects))):
            error = torch.nn.functional.mse_loss(predicted_objects[t], actual_objects[t]).item()
            errors.append(error)
        return errors
    
    def visualize_object_motion(self, predicted_objects, actual_objects, robot_trajectory, 
                               episode_num, save_path):
        """
        Create visualization comparing predicted vs actual object motion
        Uses the existing visualizer pattern
        """
        
        print(f"Creating visualization for episode {episode_num}...")
        
        # Video parameters
        width, height = self.WH
        fps = self.FPS
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        out = cv2.VideoWriter(save_path, fourcc, fps, (width, height))
        
        # Create Open3D visualizer
        vis = o3d.visualization.Visualizer()
        vis.create_window(width=width, height=height, visible=False)
        
        # Set render options
        render_option = vis.get_render_option()
        render_option.point_size = 10.0
        
        # Create point clouds
        pred_pcd = o3d.geometry.PointCloud()
        pred_pcd.points = o3d.utility.Vector3dVector(predicted_objects[0].numpy())
        pred_pcd.paint_uniform_color([1.0, 0.2, 0.2])  # Red for predicted
            
        actual_pcd = o3d.geometry.PointCloud()
        actual_pcd.points = o3d.utility.Vector3dVector(actual_objects[0].numpy())
        actual_pcd.paint_uniform_color([0.2, 0.6, 1.0])  # Blue for actual
            
        robot_pcd = o3d.geometry.PointCloud()
        robot_pcd.points = o3d.utility.Vector3dVector(robot_trajectory[0].numpy())
        robot_pcd.paint_uniform_color([0.2, 1.0, 0.2])  # Green for robot

        vis.add_geometry(pred_pcd)
        vis.add_geometry(actual_pcd)
        vis.add_geometry(robot_pcd)

        n_frames = min(len(predicted_objects), len(actual_objects))
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

        
        for frame_idx in range(n_frames):
            # Get positions for this frame
            pred_obj_pos = predicted_objects[frame_idx].numpy()
            actual_obj_pos = actual_objects[frame_idx].numpy()
            robot_pos = robot_trajectory[frame_idx].numpy()
            
            if frame_idx == 0:
                print(robot_pos)
            
            pred_pcd.points = o3d.utility.Vector3dVector(pred_obj_pos)
            robot_pcd.points = o3d.utility.Vector3dVector(robot_pos)
            actual_pcd.points = o3d.utility.Vector3dVector(actual_obj_pos)

            vis.update_geometry(pred_pcd)
            vis.update_geometry(actual_pcd)
            vis.update_geometry(robot_pcd)

            # # Create edges for predicted objects
            # pred_edges = create_edges_for_points(pred_obj_pos, self.adj_thresh)
            # pred_line_set = o3d.geometry.LineSet()
            # if len(pred_edges) > 0:
            #     pred_line_set.points = o3d.utility.Vector3dVector(pred_obj_pos)
            #     pred_line_set.lines = o3d.utility.Vector2iVector(pred_edges)
            #     pred_line_colors = np.tile([1.0, 0.2, 0.2], (len(pred_edges), 1))
            #     pred_line_set.colors = o3d.utility.Vector3dVector(pred_line_colors)
            
            # # Create edges for actual objects (blue)
            # actual_edges = create_edges_for_points(actual_obj_pos, self.adj_thresh)
            # actual_line_set = o3d.geometry.LineSet()
            # if len(actual_edges) > 0:
            #     actual_line_set.points = o3d.utility.Vector3dVector(actual_obj_pos)
            #     actual_line_set.lines = o3d.utility.Vector2iVector(actual_edges)
            #     actual_line_colors = np.tile([0.2, 0.6, 1.0], (len(actual_edges), 1))
            #     actual_line_set.colors = o3d.utility.Vector3dVector(actual_line_colors)
                        
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

def main():
    """Main function to test object motion prediction"""
    
    # Configuration
    model_path = "data/gnn_dyn_model/2025-05-29-14-52-41-654478/net_best.pth"
    config_path = "config/train/gnn_dyn.yaml"
    data_dir = "../test/PhysTwin/generated_data"
    camera_calib_path = "data/single_push_rope"  # Path to camera calibration data
    
    # Test episodes 
    test_episodes = [0, 1, 2]
    
    # Initialize predictor with camera calibration
    predictor = ObjectMotionPredictor(model_path, config_path, camera_calib_path)
    
    print("="*60)
    print("OBJECT MOTION PREDICTION TEST")
    print("="*60)
    
    all_errors = []
    
    for episode_num in test_episodes:
        print(f"\n{'='*40}")
        print(f"TESTING EPISODE {episode_num}")
        print(f"{'='*40}")
        
        try:
            # Load episode data with FPS sampling
            actual_objects, robot_trajectory, obj_indices, robot_indices = predictor.load_episode_data(data_dir, episode_num)
            
            # Predict object motion starting from initial positions
            initial_object_pos = actual_objects[0]
            predicted_objects = predictor.predict_object_motion(initial_object_pos, robot_trajectory)
            
            # Calculate prediction errors
            errors = predictor.calculate_prediction_error(predicted_objects, actual_objects)
            all_errors.extend(errors)
            
            avg_error = np.mean(errors)
            print(f"\nEpisode {episode_num} Results:")
            print(f"  Average MSE: {avg_error:.6f}")
            print(f"  RMSE: {np.sqrt(avg_error):.6f}")
            print(f"  Timesteps predicted: {len(errors)}")
            
            # Create visualization
            video_path = f"data/video/prediction_{episode_num}.mp4"
            predictor.visualize_object_motion(
                predicted_objects, actual_objects, robot_trajectory, 
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
        
        print(f"\nVideos saved in 'data/video/' directory")

if __name__ == "__main__":
    main()
