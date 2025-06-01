#!/usr/bin/env python3
"""
Convert existing .pt files to the new shared HDF5 format.

Converts 2000 episodes (0-1999) from individual .pt files to a single data.h5 file.
Each episode has object/ and robot/ folders with 82 .pt files each.
"""

import os
import torch
import h5py
import numpy as np
from datetime import datetime

def convert_pt_to_hdf5(source_dir="generated_data", output_file="generated_data/data.h5", force_overwrite=False):
    """
    Convert existing .pt files to shared HDF5 format.
    
    Args:
        source_dir: Directory containing episode folders (0, 1, 2, ..., 1999)
        output_file: Output HDF5 file path
        force_overwrite: If True, overwrite existing HDF5 file
    """
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Initialize HDF5 file
    if force_overwrite or not os.path.exists(output_file):
        if force_overwrite and os.path.exists(output_file):
            print(f"Force overwriting existing HDF5 file: {output_file}")
        else:
            print(f"Creating new HDF5 file: {output_file}")
        with h5py.File(output_file, 'w') as f:
            # Add global metadata
            f.attrs['created'] = datetime.now().isoformat()
            f.attrs['description'] = 'PhysTwin episode data collection (converted from .pt files)'
            f.attrs['conversion_source'] = source_dir
    else:
        print(f"Appending to existing HDF5 file: {output_file}")
        # Check if file already has global metadata, add if missing
        with h5py.File(output_file, 'a') as f:
            if 'description' not in f.attrs:
                f.attrs['description'] = 'PhysTwin episode data collection'
            if 'conversion_source' not in f.attrs:
                f.attrs['conversion_source'] = source_dir
    
    # Process each episode
    n_episodes = 2000
    
    print(f"Converting {n_episodes} episodes...")
    
    # Check which episodes already exist (only if not force overwriting)
    existing_episodes = set()
    if not force_overwrite:
        with h5py.File(output_file, 'r') as f:
            existing_episodes = {key for key in f.keys() if key.startswith('episode_')}
    
    converted_count = 0
    skipped_count = 0
    
    for episode_id in range(n_episodes):
        episode_key = f'episode_{episode_id:06d}'
        
        # Skip if episode already exists (and not force overwriting)
        if not force_overwrite and episode_key in existing_episodes:
            print(f"Episode {episode_id:3d}: Skipped (already exists)")
            skipped_count += 1
            continue
            
        episode_dir = os.path.join(source_dir, str(episode_id))
        object_dir = os.path.join(episode_dir, "object")
        robot_dir = os.path.join(episode_dir, "robot")
        
        # Check if directories exist
        if not os.path.exists(episode_dir):
            print(f"Episode {episode_id:3d}: Warning - Directory not found: {episode_dir}")
            continue
            
        if not os.path.exists(object_dir):
            print(f"Episode {episode_id:3d}: Warning - Object directory not found")
            continue
            
        if not os.path.exists(robot_dir):
            print(f"Episode {episode_id:3d}: Warning - Robot directory not found")
            continue
        
        try:
            # Dynamically determine number of frames by counting .pt files
            object_files = [f for f in os.listdir(object_dir) if f.startswith('x_') and f.endswith('.pt')]
            robot_files = [f for f in os.listdir(robot_dir) if f.startswith('x_') and f.endswith('.pt')]
            
            if not object_files:
                print(f"Episode {episode_id:3d}: Warning - No object files found")
                continue
                
            if not robot_files:
                print(f"Episode {episode_id:3d}: Warning - No robot files found")
                continue
                        
            n_frames = len(object_files)
            
            # Load first object file to determine particle count
            first_object_file = os.path.join(object_dir, f"x_0.pt")
            first_object_data = torch.load(first_object_file, map_location='cpu')
            n_obj_particles = first_object_data.shape[0]
            
            # Load first robot file to determine particle count
            first_robot_file = os.path.join(robot_dir, f"x_0.pt")
            first_robot_data = torch.load(first_robot_file, map_location='cpu')
            if len(first_robot_data.shape) == 2 and first_robot_data.shape[1] == 3:
                n_bot_particles = first_robot_data.shape[0]
            else:
                print(f"Episode {episode_id:3d}: Warning - Unexpected robot tensor format: {first_robot_data.shape}")
                continue
            
            print(f"Episode {episode_id:3d}: Processing {n_frames} frames, {n_obj_particles} obj particles, {n_bot_particles} robot particles")
            
            # Load object data (n_frames of n_obj_particles x 3 tensors)
            object_frames = []
            for frame_idx in range(n_frames):
                pt_file = os.path.join(object_dir, f"x_{frame_idx}.pt")
                frame_data = torch.load(pt_file, map_location='cpu')
                
                # Verify shape consistency
                if frame_data.shape != (n_obj_particles, 3):
                    print(f"Episode {episode_id:3d}: Warning - Inconsistent object shape in frame {frame_idx}: {frame_data.shape}, expected ({n_obj_particles}, 3)")
                    # Pad or truncate to match expected size
                    if frame_data.shape[0] < n_obj_particles:
                        # Pad with zeros
                        padded_data = np.zeros((n_obj_particles, 3), dtype=np.float32)
                        padded_data[:frame_data.shape[0], :] = frame_data.numpy()
                        object_frames.append(padded_data)
                    else:
                        # Truncate
                        object_frames.append(frame_data[:n_obj_particles, :].numpy())
                else:
                    object_frames.append(frame_data.numpy())
            
            # Stack into (n_frames, n_obj_particles, 3) array
            object_data = np.stack(object_frames, axis=0)
            
            # Load robot data (n_frames of n_bot_particles x 3 tensors)
            robot_frames = []
            for frame_idx in range(n_frames):
                pt_file = os.path.join(robot_dir, f"x_{frame_idx}.pt")
                frame_data = torch.load(pt_file, map_location='cpu')
                
                # Extract robot data (handle potential additional data)
                if len(frame_data.shape) == 2 and frame_data.shape[1] == 3:
                    robot_frame = frame_data.numpy()
                    
                    # Verify shape consistency
                    if robot_frame.shape[0] != n_bot_particles:
                        print(f"Episode {episode_id:3d}: Warning - Inconsistent robot shape in frame {frame_idx}: {robot_frame.shape}, expected ({n_bot_particles}, 3)")
                        # Pad or truncate to match expected size
                        if robot_frame.shape[0] < n_bot_particles:
                            # Pad with zeros
                            padded_data = np.zeros((n_bot_particles, 3), dtype=np.float32)
                            padded_data[:robot_frame.shape[0], :] = robot_frame
                            robot_frames.append(padded_data)
                        else:
                            # Truncate
                            robot_frames.append(robot_frame[:n_bot_particles, :])
                    else:
                        robot_frames.append(robot_frame)
                else:
                    print(f"Episode {episode_id:3d}: Warning - Unexpected robot tensor format in frame {frame_idx}: {frame_data.shape}")
                    robot_frames.append(np.zeros((n_bot_particles, 3), dtype=np.float32))
            
            # Stack into (n_frames, n_bot_particles, 3) array
            robot_data = np.stack(robot_frames, axis=0)
            
            # Save to HDF5
            with h5py.File(output_file, 'a') as f:
                # If force overwriting and episode exists, delete it first
                if force_overwrite and episode_key in f:
                    del f[episode_key]
                    
                # Create episode group
                episode_group = f.create_group(episode_key)
                
                # Create datasets with chunk size equal to the whole dataset
                episode_group.create_dataset(
                    'object', 
                    data=object_data, 
                    compression='gzip', 
                    compression_opts=9,
                    shuffle=True,
                    chunks=object_data.shape
                )
                
                episode_group.create_dataset(
                    'robot', 
                    data=robot_data, 
                    compression='gzip', 
                    compression_opts=9,
                    shuffle=True,
                    chunks=robot_data.shape
                )
                
                # Store metadata as attributes (using actual detected values)
                episode_group.attrs['n_frames'] = n_frames
                episode_group.attrs['n_obj_particles'] = n_obj_particles
                episode_group.attrs['n_bot_particles'] = n_bot_particles
                episode_group.attrs['object_type'] = 'rope'
                episode_group.attrs['motion_type'] = 'single_push'
                episode_group.attrs['episode_id'] = episode_id
                episode_group.attrs['include_gaussian'] = False
                
            converted_count += 1
                
        except Exception as e:
            print(f"Episode {episode_id:3d}: Error - {e}")
            continue
    
    # Print final statistics
    with h5py.File(output_file, 'r') as f:
        episode_groups = [key for key in f.keys() if key.startswith('episode_')]
        file_size_mb = os.path.getsize(output_file) / (1024 * 1024)
        
        print(f"\nConversion complete!")
        print(f"Converted {converted_count} new episodes")
        if not force_overwrite:
            print(f"Skipped {skipped_count} existing episodes")
        print(f"Total episodes in file: {len(episode_groups)}")
        print(f"Output file: {output_file}")
        print(f"File size: {file_size_mb:.2f} MB")

def verify_conversion(hdf5_file="generated_data/data.h5", sample_episodes=[0, 1, 1999]):
    """Verify the conversion by checking a few episodes."""
    print(f"\nVerifying conversion...")
    
    with h5py.File(hdf5_file, 'r') as f:
        for episode_id in sample_episodes:
            episode_key = f'episode_{episode_id:06d}'
            if episode_key in f:
                episode_group = f[episode_key]
                obj_shape = episode_group['object'].shape
                robot_shape = episode_group['robot'].shape
                
                n_frames = episode_group.attrs['n_frames']
                n_obj_particles = episode_group.attrs['n_obj_particles']
                n_bot_particles = episode_group.attrs['n_bot_particles']
                
                print(f"Episode {episode_id}:")
                print(f"  Object shape: {obj_shape} (expected: ({n_frames}, {n_obj_particles}, 3))")
                print(f"  Robot shape: {robot_shape} (expected: ({n_frames}, {n_bot_particles}, 3))")
                print(f"  Metadata: n_frames={n_frames}, "
                      f"n_obj_particles={n_obj_particles}, "
                      f"n_bot_particles={n_bot_particles}")
                
                # Verify shapes match metadata
                obj_shape_correct = obj_shape == (n_frames, n_obj_particles, 3)
                robot_shape_correct = robot_shape == (n_frames, n_bot_particles, 3)
                
                if not obj_shape_correct:
                    print(f"  ERROR: Object shape mismatch!")
                if not robot_shape_correct:
                    print(f"  ERROR: Robot shape mismatch!")
                
                # Check for any NaN or inf values
                obj_data = episode_group['object'][:]
                robot_data = episode_group['robot'][:]
                
                obj_nan = np.isnan(obj_data).any()
                robot_nan = np.isnan(robot_data).any()
                obj_inf = np.isinf(obj_data).any()
                robot_inf = np.isinf(robot_data).any()
                
                if obj_nan or robot_nan or obj_inf or robot_inf:
                    print(f"  Warning: Found NaN/inf values - obj_nan: {obj_nan}, robot_nan: {robot_nan}, obj_inf: {obj_inf}, robot_inf: {robot_inf}")
                else:
                    print(f"  Data quality: OK")
                print()
            else:
                print(f"Episode {episode_id}: Not found")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Convert .pt files to shared HDF5 format")
    parser.add_argument("--source_dir", type=str, default="generated_data", 
                       help="Directory containing episode folders")
    parser.add_argument("--output_file", type=str, default="generated_data/data.h5", 
                       help="Output HDF5 file path")
    parser.add_argument("--force_overwrite", action="store_true", 
                       help="Force overwrite existing HDF5 file")
    parser.add_argument("--verify", action="store_true", 
                       help="Verify the conversion after completion")
    
    args = parser.parse_args()
    
    # Run conversion
    convert_pt_to_hdf5(args.source_dir, args.output_file, args.force_overwrite)
    
    # Verify if requested
    if args.verify:
        verify_conversion(args.output_file)
    
    print("\nYou can now inspect the converted data with:")
    print(f"python inspect_data.py --data_file {args.output_file}")
    print(f"python inspect_data.py --data_file {args.output_file} --episode 0") 