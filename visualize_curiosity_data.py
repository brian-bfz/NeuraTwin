#!/usr/bin/env python3
"""
Visualizer for curiosity-generated training data.
Uses the existing visualize_object_motion function to create videos of the generated episodes.
"""

import os
import sys
import argparse
import h5py
import torch
import numpy as np
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

from GNN.utils.visualization import Visualizer
from GNN.utils import load_yaml


def load_episode_data(data_file, episode_idx):
    """
    Load a specific episode from the curiosity-generated data file.
    
    Args:
        data_file: str - path to HDF5 data file
        episode_idx: int - episode index to load
        
    Returns:
        tuple: (object_states, robot_states, object_edges) where:
            object_states: [n_frames, n_obj_particles, 3] - object trajectory
            robot_states: [n_frames, n_robot_particles, 3] - robot trajectory  
            object_edges: [n_obj_particles, n_obj_particles] - topological edges
    """
    with h5py.File(data_file, 'r') as f:
        episode_name = f'episode_{episode_idx:06d}'
        if episode_name not in f:
            available_episodes = [k for k in f.keys() if k.startswith('episode_')]
            raise ValueError(f"Episode {episode_idx} not found. Available episodes: {len(available_episodes)}")
        
        episode_group = f[episode_name]
        
        object_states = torch.tensor(episode_group['object'][:], dtype=torch.float32)
        robot_states = torch.tensor(episode_group['robot'][:], dtype=torch.float32)
        object_edges = torch.tensor(episode_group['object_edges'][:], dtype=torch.float32)
        
        n_frames = episode_group.attrs['n_frames']
        n_obj_particles = episode_group.attrs['n_obj_particles']
        n_robot_particles = episode_group.attrs['n_bot_particles']
        
        print(f"Loaded episode {episode_idx}:")
        print(f"  Frames: {n_frames}")
        print(f"  Object particles: {n_obj_particles}")
        print(f"  Robot particles: {n_robot_particles}")
        
        return object_states, robot_states, object_edges


def create_combined_trajectory(object_states, robot_states):
    """
    Combine object and robot trajectories into a single trajectory with proper masking.
    
    Args:
        object_states: [n_frames, n_obj, 3] - object trajectory
        robot_states: [n_frames, n_robot, 3] - robot trajectory
        
    Returns:
        tuple: (combined_states, tool_mask) where:
            combined_states: [n_frames, n_particles, 3] - combined trajectory
            tool_mask: [n_particles] - boolean mask (False=object, True=robot)
    """
    n_frames, n_obj, _ = object_states.shape
    n_robot = robot_states.shape[1]
    n_particles = n_obj + n_robot
    
    # Create combined trajectory
    combined_states = torch.zeros(n_frames, n_particles, 3)
    combined_states[:, :n_obj, :] = object_states
    combined_states[:, n_obj:, :] = robot_states
    
    # Create tool mask (False for object particles, True for robot particles)
    tool_mask = torch.zeros(n_particles, dtype=torch.bool)
    tool_mask[n_obj:] = True
    
    return combined_states, tool_mask


def create_topological_edges_matrix(object_edges, n_robot_particles):
    """
    Create full topological edges matrix for combined object+robot particles.
    
    Args:
        object_edges: [n_obj, n_obj] - object-only topological edges
        n_robot_particles: int - number of robot particles
        
    Returns:
        topological_edges: [n_particles, n_particles] - full edges matrix
    """
    n_obj = object_edges.shape[0]
    n_particles = n_obj + n_robot_particles
    
    # Create full matrix (only objects have topological edges)
    topological_edges = torch.zeros(n_particles, n_particles, dtype=torch.float32)
    topological_edges[:n_obj, :n_obj] = object_edges
    
    return topological_edges


def get_camera_calibration_path(case_name):
    """
    Get camera calibration path for the given case.
    
    Args:
        case_name: str - case name (e.g., "single_push_rope")
        
    Returns:
        str - path to camera calibration directory
    """
    # Map case names to their data directories
    case_mapping = {
        "single_push_rope": "PhysTwin/data/single_push_rope",
        "rope": "PhysTwin/data/rope", 
        "cloth": "PhysTwin/data/cloth"
    }
    
    if case_name in case_mapping:
        return case_mapping[case_name]
    else:
        # Default fallback
        return f"PhysTwin/data/{case_name}"


def visualize_episode(data_file, episode_idx, output_dir, case_name="single_push_rope", target_pcd=None):
    """
    Visualize a single episode from curiosity-generated data.
    
    Args:
        data_file: str - path to HDF5 data file
        episode_idx: int - episode index to visualize
        output_dir: str - directory to save output video
        case_name: str - case name for camera calibration
        target_pcd: tensor or None - target point cloud for visualization
        
    Returns:
        str - path to output video file
    """
    # Load episode data
    object_states, robot_states, object_edges = load_episode_data(data_file, episode_idx)
    
    # Create combined trajectory and masks
    combined_states, tool_mask = create_combined_trajectory(object_states, robot_states)
    
    # Create topological edges matrix
    topological_edges = create_topological_edges_matrix(object_edges, robot_states.shape[1])
    
    # Get camera calibration path
    camera_calib_path = get_camera_calibration_path(case_name)
    if not os.path.exists(camera_calib_path):
        print(f"Warning: Camera calibration path not found: {camera_calib_path}")
        print("Using default camera settings")
        camera_calib_path = None
    
    # Initialize visualizer
    if camera_calib_path:
        visualizer = Visualizer(camera_calib_path)
    else:
        # Create a mock visualizer with default settings
        class MockVisualizer(Visualizer):
            def __init__(self):
                self.WH = [640, 480]  # Default resolution
                self.FPS = 30.0  # Default FPS
                self.w2cs = None
                self.intrinsics = None
                self.c2ws = None
                
        visualizer = MockVisualizer()
    
    # Create default target if none provided
    if target_pcd is None:
        # Create a simple target at the final object position with some offset
        final_obj_pos = object_states[-1].mean(dim=0)  # Center of final object positions
        target_pcd = final_obj_pos.unsqueeze(0) + torch.tensor([[0.1, 0.0, 0.0]])  # Slight offset
        target_pcd = target_pcd.numpy()
    
    # Setup output path
    os.makedirs(output_dir, exist_ok=True)
    output_video = os.path.join(output_dir, f"curiosity_episode_{episode_idx:06d}.mp4")
    
    print(f"Visualizing episode {episode_idx}...")
    print(f"Combined trajectory shape: {combined_states.shape}")
    print(f"Tool mask: {tool_mask.sum().item()} robot particles, {(~tool_mask).sum().item()} object particles")
    
    # Create visualization
    # Note: For curiosity data, we use the same trajectory for both predicted and actual
    # since this is showing the generated data, not comparing predictions
    saved_path = visualizer.visualize_object_motion(
        predicted_states=combined_states,
        tool_mask=tool_mask,
        actual_objects=None,  # Use object states as "ground truth"
        save_path=output_video,
        topological_edges=topological_edges,
        target=None
    )
    
    return saved_path


def list_available_episodes(data_file):
    """
    List all available episodes in the data file.
    
    Args:
        data_file: str - path to HDF5 data file
        
    Returns:
        list - episode indices
    """
    with h5py.File(data_file, 'r') as f:
        episode_names = [k for k in f.keys() if k.startswith('episode_')]
        episode_indices = sorted([int(name.split('_')[1]) for name in episode_names])
        
        print(f"Found {len(episode_indices)} episodes in {data_file}")
        if episode_indices:
            print(f"Episode range: {min(episode_indices)} to {max(episode_indices)}")
            
        return episode_indices


def main():
    """Main visualization function."""
    parser = argparse.ArgumentParser(description='Visualize curiosity-generated training data')
    parser.add_argument('--data_file', type=str, required=True,
                       help='Path to HDF5 data file with curiosity-generated episodes')
    parser.add_argument('--episode', type=int, default=None,
                       help='Specific episode to visualize (default: visualize all)')
    parser.add_argument('--output_dir', type=str, default='curiosity_visualizations',
                       help='Output directory for videos')
    parser.add_argument('--case_name', type=str, default='single_push_rope',
                       help='Case name for camera calibration')
    parser.add_argument('--list_episodes', action='store_true',
                       help='List available episodes and exit')
    parser.add_argument('--max_episodes', type=int, default=5,
                       help='Maximum number of episodes to visualize (when visualizing all)')
    
    args = parser.parse_args()
    
    # Check if data file exists
    if not os.path.exists(args.data_file):
        print(f"ERROR: Data file not found: {args.data_file}")
        return
    
    # List episodes if requested
    if args.list_episodes:
        list_available_episodes(args.data_file)
        return
    
    # Get available episodes
    available_episodes = list_available_episodes(args.data_file)
    if not available_episodes:
        print("No episodes found in data file")
        return
    
    # Determine which episodes to visualize
    if args.episode is not None:
        if args.episode not in available_episodes:
            print(f"Episode {args.episode} not found. Available episodes: {available_episodes}")
            return
        episodes_to_visualize = [args.episode]
    else:
        # Visualize up to max_episodes
        episodes_to_visualize = available_episodes[:args.max_episodes]
        print(f"Visualizing {len(episodes_to_visualize)} episodes (limited by --max_episodes={args.max_episodes})")
    
    # Visualize each episode
    output_videos = []
    for episode_idx in episodes_to_visualize:
        try:
            print(f"\n" + "="*50)
            print(f"VISUALIZING EPISODE {episode_idx}")
            print("="*50)
            
            output_video = visualize_episode(
                args.data_file, 
                episode_idx, 
                args.output_dir, 
                args.case_name
            )
            output_videos.append(output_video)
            
            print(f"✓ Episode {episode_idx} visualized successfully")
            
        except Exception as e:
            print(f"✗ Failed to visualize episode {episode_idx}: {e}")
            import traceback
            traceback.print_exc()
    
    # Print summary
    print(f"\n" + "="*60)
    print("VISUALIZATION SUMMARY")
    print("="*60)
    print(f"Successfully visualized: {len(output_videos)} episodes")
    print(f"Output directory: {args.output_dir}")
    
    if output_videos:
        print("\nGenerated videos:")
        for video_path in output_videos:
            print(f"  - {video_path}")
        print(f"\n🎉 All visualizations complete! Check the videos in '{args.output_dir}'")
    else:
        print("\n❌ No visualizations were generated successfully")


if __name__ == "__main__":
    # Example usage:
    # python visualize_curiosity_data.py --data_file temp_experiments/single_process_test.h5 --list_episodes
    # python visualize_curiosity_data.py --data_file temp_experiments/single_process_test.h5 --episode 0
    # python visualize_curiosity_data.py --data_file temp_experiments/single_process_test.h5 --max_episodes 3
    main() 