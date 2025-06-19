import os
import h5py
import random
import torch
from GNN.curiosity import CuriosityPlanner
from GNN.utils import load_yaml
from scripts.utils import load_mpc_data

def generate_data(model_path, train_config_path, input_file, output_file):
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

    for i in range(n_episodes // n_explore_iter):
        # Extract phystwin_states and phystwin_robot_mask.
        # Just use the first episode for now, since all episodes start with the same initial states.
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        phystwin_states, phystwin_robot_mask, _ = load_mpc_data(0, input_file, device)        
        curiosity_planner = CuriosityPlanner(model_path, train_config_path, mpc_config_path, phystwin_states, phystwin_robot_mask, "single_push_rope")
        for j in range(n_explore_iter):
            phystwin_states = curiosity_planner.explore(output_file, phystwin_states) # pass final states from previous exploration as initial states for next exploration