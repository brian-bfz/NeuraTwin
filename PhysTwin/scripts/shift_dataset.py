#!/usr/bin/env python3
"""
Script to read episodes 0-9 from HDF5 dataset, shift all positions by +100,
and save to a new file called shifted_data.h5
"""

import os
import h5py
import numpy as np
import argparse
from tqdm import tqdm

def shift_dataset(input_hdf5_path, output_hdf5_path, episodes_to_shift, shift_value=100.0):
    """
    Read specified episodes from input HDF5, shift all positions by shift_value,
    and save to output HDF5 file.
    
    Args:
        input_hdf5_path: Path to input HDF5 file (e.g., 'data.h5')
        output_hdf5_path: Path to output HDF5 file (e.g., 'shifted_data.h5')
        episodes_to_shift: List of episode indices to process (e.g., [0, 1, 2, ..., 9])
        shift_value: Value to add to all particle positions (default: 100.0)
    """
    
    print(f"Reading from: {input_hdf5_path}")
    print(f"Writing to: {output_hdf5_path}")
    print(f"Episodes to process: {episodes_to_shift}")
    print(f"Shift value: {shift_value}")
    
    # Open input file in read mode
    with h5py.File(input_hdf5_path, 'r') as input_file:
        # Create output file
        with h5py.File(output_hdf5_path, 'w') as output_file:
            
            print(f"Processing {len(episodes_to_shift)} episodes...")
            
            for episode_idx in tqdm(episodes_to_shift, desc="Shifting episodes"):
                episode_name = f'episode_{episode_idx:06d}'
                
                # Check if episode exists in input file
                if episode_name not in input_file:
                    print(f"Warning: Episode {episode_idx} not found in input file, skipping...")
                    continue
                
                input_episode = input_file[episode_name]
                
                # Create corresponding episode group in output file
                output_episode = output_file.create_group(episode_name)
                
                # Copy metadata attributes
                for attr_name, attr_value in input_episode.attrs.items():
                    output_episode.attrs[attr_name] = attr_value
                
                # Get metadata for info
                n_frames = input_episode.attrs['n_frames']
                n_obj_particles = input_episode.attrs['n_obj_particles'] 
                n_bot_particles = input_episode.attrs['n_bot_particles']
                
                print(f"  Episode {episode_idx}: {n_frames} frames, "
                      f"{n_obj_particles} object particles, {n_bot_particles} robot particles")
                
                # Process object data
                if 'object' in input_episode:
                    object_data = input_episode['object'][:]  # Shape: [time, n_obj, 3]
                    
                    # Shift all object positions by adding shift_value to x, y, z coordinates
                    shifted_object_data = object_data + shift_value
                    
                    # Save shifted object data
                    output_episode.create_dataset('object', data=shifted_object_data, 
                                                compression='gzip', compression_opts=9)
                    
                    print(f"    Object data: {object_data.shape} -> shifted by {shift_value}")
                
                # Process robot data
                if 'robot' in input_episode:
                    robot_data = input_episode['robot'][:]  # Shape: [time, n_bot, 3]
                    
                    # Shift all robot positions by adding shift_value to x, y, z coordinates
                    shifted_robot_data = robot_data + shift_value
                    
                    # Save shifted robot data
                    output_episode.create_dataset('robot', data=shifted_robot_data, 
                                                compression='gzip', compression_opts=9)
                    
                    print(f"    Robot data: {robot_data.shape} -> shifted by {shift_value}")
                
                # Copy any other datasets that might exist (maintaining original structure)
                for dataset_name in input_episode.keys():
                    if dataset_name not in ['object', 'robot']:
                        # Copy other datasets without modification
                        input_data = input_episode[dataset_name][:]
                        output_episode.create_dataset(dataset_name, data=input_data,
                                                    compression='gzip', compression_opts=9)
                        print(f"    Copied dataset '{dataset_name}': {input_data.shape}")
            
            print(f"\nSuccessfully created shifted dataset: {output_hdf5_path}")
            print(f"All particle positions shifted by {shift_value} units in all dimensions")

def main():
    parser = argparse.ArgumentParser(description="Shift particle positions in HDF5 dataset")
    parser.add_argument('--input_dir', type=str, required=True,
                       help='Directory containing the input data.h5 file')
    parser.add_argument('--episodes', type=str, default='0-9',
                       help='Episodes to process (e.g., "0-9", "0,1,2,3" or "all")')
    parser.add_argument('--shift', type=float, default=100.0,
                       help='Value to add to all particle positions (default: 100.0)')
    parser.add_argument('--output_name', type=str, default='shifted_data.h5',
                       help='Output filename (default: shifted_data.h5)')
    
    args = parser.parse_args()
    
    # Construct file paths
    input_hdf5_path = os.path.join(args.input_dir, 'data.h5')
    output_hdf5_path = os.path.join(args.input_dir, args.output_name)
    
    # Check if input file exists
    if not os.path.exists(input_hdf5_path):
        print(f"Error: Input file {input_hdf5_path} not found!")
        return
    
    # Parse episodes
    if args.episodes == 'all':
        # Read all episodes from the file
        with h5py.File(input_hdf5_path, 'r') as f:
            episode_keys = [k for k in f.keys() if k.startswith('episode_')]
            episodes_to_shift = [int(k.split('_')[1]) for k in episode_keys]
        print(f"Found {len(episodes_to_shift)} episodes in file")
    elif '-' in args.episodes:
        # Range format: "0-9"
        start, end = map(int, args.episodes.split('-'))
        episodes_to_shift = list(range(start, end + 1))
    else:
        # Comma-separated format: "0,1,2,3"
        episodes_to_shift = [int(x.strip()) for x in args.episodes.split(',')]
    
    print(f"Episodes to process: {episodes_to_shift}")
    
    # Check if output file already exists
    if os.path.exists(output_hdf5_path):
        response = input(f"Output file {output_hdf5_path} already exists. Overwrite? (y/n): ")
        if response.lower() != 'y':
            print("Aborted.")
            return
    
    # Perform the shifting
    try:
        shift_dataset(input_hdf5_path, output_hdf5_path, episodes_to_shift, args.shift)
        
        # Print summary
        print(f"\n{'='*60}")
        print("SUMMARY")
        print(f"{'='*60}")
        print(f"Input file: {input_hdf5_path}")
        print(f"Output file: {output_hdf5_path}")
        print(f"Episodes processed: {len(episodes_to_shift)}")
        print(f"Shift applied: +{args.shift} to all x, y, z coordinates")
        print(f"File saved successfully!")
        
    except Exception as e:
        print(f"Error during processing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 