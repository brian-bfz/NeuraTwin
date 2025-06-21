import os
import h5py
import random
import torch
from GNN.curiosity import CuriosityPlanner
import argparse
from GNN.utils import load_yaml
from scripts.utils import load_mpc_data, load_mpc_data2
from GNN.paths import get_model_paths


def generate_data(model_path, train_config_path, input_file, output_file, mode="curiosity"):
    mpc_config_path = "GNN/config/mpc/curiosity.yaml"
    config = load_yaml(mpc_config_path)
    n_episodes = config['n_episodes']
    n_explore_iter = config['n_explore_iter']
    assert n_episodes % n_explore_iter == 0
    
    # Initialize output file as empty if it doesn't exist, create it
    if os.path.exists(output_file):
        raise ValueError(f"Output file already exists: {output_file}")
    else:
        with h5py.File(output_file, 'w') as f:
            f.attrs['description'] = 'GNN curiosity-driven training data'
            f.attrs['unused_idx'] = 0
        print(f"Created empty output file: {output_file}")


    if mode == "curiosity":
        curiosity_planner = CuriosityPlanner(model_path, train_config_path, mpc_config_path, "single_push_rope")
    elif mode == "mpc":
        mpc_planner = CuriosityPlanner(model_path, train_config_path, mpc_config_path, "single_push_rope", mode = "mpc")

    for i in range(n_episodes // n_explore_iter):
        # Extract phystwin_states and phystwin_robot_mask.
        # Just use the first episode for now, since all episodes start with the same initial states.
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if mode == "curiosity":
            phystwin_states, phystwin_robot_mask, _ = load_mpc_data(0, input_file, device)    
            curiosity_planner.load_phystwin_data(phystwin_states, phystwin_robot_mask)
            for j in range(n_explore_iter):
                phystwin_states = curiosity_planner.explore(output_file, phystwin_states) # pass final states from previous exploration as initial states for next exploration
        elif mode == "mpc":
            phystwin_states, phystwin_robot_mask, target = load_mpc_data2(input_file, device)
            mpc_planner.load_phystwin_data(phystwin_states, phystwin_robot_mask)
            for j in range(n_explore_iter):
                phystwin_states = mpc_planner.explore(output_file, phystwin_states, target) # pass final states from previous exploration as initial states for next exploration

import time
if __name__ == "__main__":
    """
    Simple test script for both curiosity and mpc modes.
    """
    # Test configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Paths and files
    input_file = "PhysTwin/generated_data/mixed_push_rope.h5"
    
    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"Input file not found: {input_file}")
        print("Please ensure the input data file exists before running tests.")
        exit(1)
    
    # Get model paths (similar to test_curiosity.py)
    model_name = "new_push_data"
    model_paths = get_model_paths(model_name)
    model_path = str(model_paths['net_best'])
    train_config_path = str(model_paths['config'])
    
    # Check if model files exist
    if not os.path.exists(model_path):
        print(f"Model file not found: {model_path}")
        print("Please ensure the trained model exists before running tests.")
        exit(1)
    
    if not os.path.exists(train_config_path):
        print(f"Train config file not found: {train_config_path}")
        print("Please ensure the training config exists before running tests.")
        exit(1)
    
    print("Testing data generation...")
    
    # Test 1: Curiosity mode
    print("\n=== Testing Curiosity Mode ===")
    curiosity_output_file = "temp_experiments/test_curiosity_data.h5"
    
    # Remove existing output file if it exists
    if os.path.exists(curiosity_output_file):
        os.remove(curiosity_output_file)
        print(f"Removed existing output file: {curiosity_output_file}")
    
    try:
        start_time = time.time()
        generate_data(
            model_path=model_path,
            train_config_path=train_config_path,
            input_file=input_file,
            output_file=curiosity_output_file,
            mode="curiosity"
        )
        print(f"✓ Curiosity mode completed successfully!")
        print(f"  Output saved to: {curiosity_output_file}")
        end_time = time.time()
        print(f"  Time taken: {end_time - start_time} seconds")
        
        # Check output file
        if os.path.exists(curiosity_output_file):
            with h5py.File(curiosity_output_file, 'r') as f:
                episode_keys = [k for k in f.keys() if k.startswith('episode_')]
                print(f"  Generated {len(episode_keys)} episodes")
        
    except Exception as e:
        print(f"✗ Error in curiosity mode: {e}")
    
    # Test 2: MPC mode
    print("\n=== Testing MPC Mode ===")
    mpc_output_file = "temp_experiments/test_mpc_data.h5"
    
    # Remove existing output file if it exists
    if os.path.exists(mpc_output_file):
        os.remove(mpc_output_file)
        print(f"Removed existing output file: {mpc_output_file}")
    
    try:
        start_time = time.time()
        generate_data(
            model_path=model_path,
            train_config_path=train_config_path,
            input_file=input_file,
            output_file=mpc_output_file,
            mode="mpc"
        )
        print(f"✓ MPC mode completed successfully!")
        print(f"  Output saved to: {mpc_output_file}")
        end_time = time.time()
        print(f"  Time taken: {end_time - start_time} seconds")
        
        # Check output file
        if os.path.exists(mpc_output_file):
            with h5py.File(mpc_output_file, 'r') as f:
                episode_keys = [k for k in f.keys() if k.startswith('episode_')]
                print(f"  Generated {len(episode_keys)} episodes")
        
    except Exception as e:
        print(f"✗ Error in MPC mode: {e}")
    
    print("\n=== Test Summary ===")
    curiosity_success = os.path.exists(curiosity_output_file)
    mpc_success = os.path.exists(mpc_output_file)
    
    print(f"Curiosity mode: {'✓ PASSED' if curiosity_success else '✗ FAILED'}")
    print(f"MPC mode: {'✓ PASSED' if mpc_success else '✗ FAILED'}")
    
    if curiosity_success and mpc_success:
        print("🎉 All tests passed!")
    else:
        print("⚠️  Some tests failed. Check the error messages above.")
