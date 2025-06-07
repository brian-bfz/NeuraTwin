#!/usr/bin/env python3
import h5py
import os

def drop_first_frames():
    input_file = "generated_data/data.h5"
    output_file = "generated_data/less_empty_data.h5"
    frames_to_drop = 20
    
    # Remove output file if it exists
    if os.path.exists(output_file):
        os.remove(output_file)
    
    with h5py.File(input_file, 'r') as f_in, h5py.File(output_file, 'w') as f_out:
        
        # Copy global attributes
        for attr_key, attr_value in f_in.attrs.items():
            f_out.attrs[attr_key] = attr_value
        
        # Process each episode
        for episode_key in f_in.keys():
            if not episode_key.startswith('episode_'):
                continue
                
            episode_in = f_in[episode_key]
            original_frames = episode_in.attrs.get('n_frames', 0)
            
            # Skip if not enough frames
            if original_frames <= frames_to_drop:
                continue
            
            # Create new episode
            episode_out = f_out.create_group(episode_key)
            
            # Copy attributes, update frame count
            for attr_key, attr_value in episode_in.attrs.items():
                if attr_key == 'n_frames':
                    episode_out.attrs[attr_key] = original_frames - frames_to_drop
                else:
                    episode_out.attrs[attr_key] = attr_value
            
            # Drop first 25 frames from object and robot datasets
            if 'object' in episode_in:
                episode_out.create_dataset('object', data=episode_in['object'][frames_to_drop:])
            
            if 'robot' in episode_in:
                episode_out.create_dataset('robot', data=episode_in['robot'][frames_to_drop:])

if __name__ == "__main__":
    drop_first_frames()
    print("Done! Saved to generated_data/less_empty_data.h5")