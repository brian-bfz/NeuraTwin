import os
import numpy as np
import torch
import open3d as o3d
import cv2
from dataset.dataset_gnn_dyn import ParticleDataset
import yaml

def load_yaml(path):
    with open(path, 'r') as file:
        return yaml.safe_load(file)

def visualize_particle_dataset(data_dir, config_path, episode_idx=0, output_path="data/video/particles_visualization.mp4"):
    """
    Visualize particle dataset with different colors for object vs robot particles
    
    Args:
        data_dir: Path to dataset directory
        config_path: Path to config yaml file
        episode_idx: Which episode to visualize
        output_path: Output video file path
    """
    
    # Load config and create dataset
    config = load_yaml(config_path)
    dataset = ParticleDataset(data_dir, config, phase='train')
    
    # Get first sample to understand dimensions
    sample = dataset[episode_idx]
    states, states_delta, attrs, particle_nums = sample
    
    print(f"States shape: {states.shape}")
    print(f"Attrs shape: {attrs.shape}")
    print(f"Particle nums: {particle_nums}")
    
    # Analyze particle distribution to set up camera properly
    all_positions = states.view(-1, 3).numpy()
    center = np.mean(all_positions, axis=0)
    extent = np.max(all_positions, axis=0) - np.min(all_positions, axis=0)
    
    print(f"Particle center: {center}")
    print(f"Particle extent: {extent}")
    
    # Video parameters
    fps = 10
    width, height = 1280, 720
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    
    # Create video writer
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Create Open3D visualizer
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=width, height=height, visible=False)
    
    # Set up camera parameters
    view_control = vis.get_view_control()
    
    n_frames = states.shape[0]
    print(f"Rendering {n_frames} frames...")
    
    for frame_idx in range(n_frames):
        # Get current frame positions and attributes
        positions = states[frame_idx].numpy()  # [N, 3]
        frame_attrs = attrs[frame_idx].numpy()  # [N]
        
        print(positions)  # Debug print
        
        # Create point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(positions)
        
        # Color particles based on attributes
        colors = np.zeros((len(positions), 3))
        
        # Object particles (attr = 0): Blue
        object_mask = frame_attrs == 0
        colors[object_mask] = [0.2, 0.6, 1.0]  # Light blue
        
        # Robot particles (attr = 1): Red
        robot_mask = frame_attrs == 1
        colors[robot_mask] = [1.0, 0.2, 0.2]  # Red
        
        pcd.colors = o3d.utility.Vector3dVector(colors)
        
        # Clear previous frame and add new point cloud
        vis.clear_geometries()
        vis.add_geometry(pcd)
        
        # Set camera view consistently for all frames
        view_control.set_lookat(center)
        view_control.set_up([0, 1, 0])  # Z is up
        view_control.set_front([0, 0, 1])  # Look along Y axis (rotated 90 degrees)
        
        # Set zoom based on particle extent
        zoom_factor = 1.0 / (max(extent) * 2)
        view_control.set_zoom(zoom_factor)
        
        # Update visualization
        vis.poll_events()
        vis.update_renderer()
        
        # Capture frame
        image = vis.capture_screen_float_buffer(do_render=True)
        image_np = np.asarray(image)
        
        # Convert from RGB float [0,1] to BGR uint8 [0,255] for cv2
        image_bgr = (image_np * 255).astype(np.uint8)
        image_bgr = cv2.cvtColor(image_bgr, cv2.COLOR_RGB2BGR)
        
        # Write frame to video
        video_writer.write(image_bgr)
        
        if frame_idx % 10 == 0:
            print(f"Rendered frame {frame_idx}/{n_frames}")
    
    # Clean up
    video_writer.release()
    vis.destroy_window()
    
    print(f"Video saved to: {output_path}")
    print(f"Object particles: Blue (attr=0)")
    print(f"Robot particles: Red (attr=1)")

def visualize_multiple_episodes(data_dir, config_path, num_episodes=3):
    """Visualize multiple episodes"""
    for i in range(num_episodes):
        output_path = f"data/video/particles_episode_{i}.mp4"
        print(f"\n=== Visualizing Episode {i} ===")
        try:
            visualize_particle_dataset(data_dir, config_path, episode_idx=i, output_path=output_path)
        except Exception as e:
            print(f"Error visualizing episode {i}: {e}")

if __name__ == "__main__":
    # Example usage
    data_dir = "data/gnn_dyn_data"  # Update with your actual data directory
    config_path = "config/train/gnn_dyn.yaml"
    
    # Visualize single episode
    visualize_particle_dataset(data_dir, config_path, episode_idx=40)
    
    # Optionally visualize multiple episodes
    # visualize_multiple_episodes(data_dir, config_path, num_episodes=3)