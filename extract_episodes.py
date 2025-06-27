#!/usr/bin/env python3
"""
Script to extract specific episodes from HDF5 file and renumber them.
Extracts episodes 0-2699 and 5400-5999, renumbering the second batch as 2700-3299.
"""

import h5py
import numpy as np

def extract_episodes(input_file, output_file):
    """
    Extract episodes 0-2699 and 5400-5999 from input_file and save to output_file.
    Renumber episodes 5400-5999 as 2700-3299 in the output file.
    """
    
    with h5py.File(input_file, 'r') as f_in:
        with h5py.File(output_file, 'w') as f_out:
            
            # Extract episodes 0-2699 (keep original numbering)
            print("Extracting episodes 0-2699...")
            for i in range(2700):
                episode_name = f'episode_{i:06d}'
                if episode_name in f_in:
                    # Copy the entire episode group
                    f_in.copy(f'episode_{i:06d}', f_out, name=f'episode_{i:06d}')
                    print(f"Copied {episode_name}")
                else:
                    print(f"Warning: {episode_name} not found in input file")
            
            # Extract episodes 5400-5999 and renumber as 2700-3299
            print("Extracting episodes 5400-5999 and renumbering as 2700-3299...")
            for i in range(600):  # 600 episodes from 5400-5999
                original_episode = 5400 + i
                new_episode = 2700 + i
                
                original_name = f'episode_{original_episode:06d}'
                new_name = f'episode_{new_episode:06d}'
                
                if original_name in f_in:
                    # Copy the entire episode group with new name
                    f_in.copy(original_name, f_out, name=new_name)
                    print(f"Copied {original_name} -> {new_name}")
                else:
                    print(f"Warning: {original_name} not found in input file")
    
    print(f"Extraction complete! Output saved to {output_file}")

if __name__ == "__main__":
    input_file = "PhysTwin/generated_data/push_sampled_12_edges.h5"
    output_file = "PhysTwin/generated_data/push_sampled_extracted.h5"
    
    print(f"Input file: {input_file}")
    print(f"Output file: {output_file}")
    
    extract_episodes(input_file, output_file) 