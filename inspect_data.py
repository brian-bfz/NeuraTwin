#!/usr/bin/env python3
"""
Utility script to inspect the shared HDF5 data file structure and contents.
"""

import h5py
import numpy as np
import argparse
import os
from datetime import datetime

def print_group_structure(group, indent=0):
    """Recursively print HDF5 group structure"""
    prefix = "  " * indent
    
    for key in group.keys():
        item = group[key]
        if isinstance(item, h5py.Group):
            print(f"{prefix}{key}/ (group)")
            # Print attributes if any
            if item.attrs:
                for attr_key, attr_value in item.attrs.items():
                    print(f"{prefix}  @{attr_key}: {attr_value}")
            print_group_structure(item, indent + 1)
        elif isinstance(item, h5py.Dataset):
            print(f"{prefix}{key} (dataset): shape={item.shape}, dtype={item.dtype}")

def inspect_data_file(data_file_path):
    """Inspect the contents of the shared HDF5 data file"""
    
    if not os.path.exists(data_file_path):
        print(f"Data file not found: {data_file_path}")
        return
    
    print(f"Inspecting: {data_file_path}")
    file_size_mb = os.path.getsize(data_file_path) / (1024 * 1024)
    print(f"File size: {file_size_mb:.2f} MB")
    print("-" * 50)
    
    with h5py.File(data_file_path, 'r') as f:
        # Print global attributes
        print("Global attributes:")
        for attr_key, attr_value in f.attrs.items():
            print(f"  @{attr_key}: {attr_value}")
        print()
        
        # Print structure
        print("File structure:")
        print_group_structure(f)
        print()
        
        # List episodes
        episode_groups = [key for key in f.keys() if key.startswith('episode_')]
        episode_groups.sort()
        
        print(f"Found {len(episode_groups)} episodes:")
        
        total_frames = 0
        total_object_particles = 0
        total_robot_particles = 0
        
        for episode_key in episode_groups:
            episode_group = f[episode_key]
            
            # Get metadata
            n_frames = episode_group.attrs.get('n_frames', 'unknown')
            n_obj_particles = episode_group.attrs.get('n_obj_particles', 'unknown')
            n_bot_particles = episode_group.attrs.get('n_bot_particles', 'unknown')
            object_type = episode_group.attrs.get('object_type', 'unknown')
            motion_type = episode_group.attrs.get('motion_type', 'unknown')
            timestamp = episode_group.attrs.get('timestamp', 'unknown')
            
            print(f"  {episode_key}: {n_frames} frames, {n_obj_particles} obj particles, {n_bot_particles} robot particles")
            print(f"    Object type: {object_type}, Motion type: {motion_type}")
            print(f"    Timestamp: {timestamp}")
            
            if isinstance(n_frames, int):
                total_frames += n_frames
            if isinstance(n_obj_particles, int):
                total_object_particles = n_obj_particles  # Should be same for all episodes
            if isinstance(n_bot_particles, int):
                total_robot_particles = n_bot_particles   # Should be same for all episodes
            
            # Check data integrity
            if 'object' in episode_group and 'robot' in episode_group:
                obj_shape = episode_group['object'].shape
                robot_shape = episode_group['robot'].shape
                print(f"    Data shapes: object={obj_shape}, robot={robot_shape}")
                
                # Check if gaussians data exists
                if 'gaussians' in episode_group:
                    gaussians_group = episode_group['gaussians']
                    if 'xyz' in gaussians_group:
                        gaussians_shape = gaussians_group['xyz'].shape
                        print(f"    Gaussians shape: {gaussians_shape}")
            print()
        
        print(f"Summary:")
        print(f"  Total episodes: {len(episode_groups)}")
        print(f"  Total frames: {total_frames}")
        print(f"  Object particles per episode: {total_object_particles}")
        print(f"  Robot particles per episode: {total_robot_particles}")

def inspect_episode(data_file_path, episode_id):
    """Inspect a specific episode in detail"""
    
    with h5py.File(data_file_path, 'r') as f:
        episode_key = f'episode_{episode_id:06d}'
        
        if episode_key not in f:
            print(f"Episode {episode_id} not found in data file")
            return
        
        episode_group = f[episode_key]
        
        print(f"Episode {episode_id} details:")
        print("-" * 30)
        
        # Print all attributes
        print("Metadata:")
        for attr_key, attr_value in episode_group.attrs.items():
            print(f"  {attr_key}: {attr_value}")
        print()
        
        # Print dataset details
        for dataset_name in ['object', 'robot']:
            if dataset_name in episode_group:
                dataset = episode_group[dataset_name]
                print(f"{dataset_name.capitalize()} data:")
                print(f"  Shape: {dataset.shape}")
                print(f"  Dtype: {dataset.dtype}")
                print(f"  Chunks: {dataset.chunks}")
                print(f"  Compression: {dataset.compression}")
                
                # Show sample data from first and last frame
                if len(dataset.shape) >= 2:
                    print(f"  First frame sample (first 3 particles):")
                    print(f"    {dataset[0, :min(3, dataset.shape[1]), :]}")
                    if dataset.shape[0] > 1:
                        print(f"  Last frame sample (first 3 particles):")
                        print(f"    {dataset[-1, :min(3, dataset.shape[1]), :]}")
                print()
        
        # Check gaussians data if available
        if 'gaussians' in episode_group:
            gaussians_group = episode_group['gaussians']
            print("Gaussians data:")
            for dataset_name in gaussians_group.keys():
                dataset = gaussians_group[dataset_name]
                print(f"  {dataset_name}: shape={dataset.shape}, dtype={dataset.dtype}")
            print()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect shared HDF5 data file")
    parser.add_argument("--data_file", type=str, default="generated_data/data.h5", 
                       help="Path to the HDF5 data file")
    parser.add_argument("--episode", type=int, help="Inspect specific episode in detail")
    
    args = parser.parse_args()
    
    if args.episode is not None:
        inspect_episode(args.data_file, args.episode)
    else:
        inspect_data_file(args.data_file) 