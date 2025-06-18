import torch
import h5py
from .curiosity import CuriosityPlanner
from .paths import get_model_paths

def test_curiosity_planner():
    """
    Simple test script to load episode 0 and test CuriosityPlanner.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load data from mixed_push_rope.h5
    data_file = "PhysTwin/generated_data/mixed_push_rope.h5"
    episode_idx = 0
    
    with h5py.File(data_file, 'r') as f:
        episode_group = f[f'episode_{episode_idx:06d}']
        
        # Load first frame
        object_data = torch.tensor(episode_group['object'][0], dtype=torch.float32, device=device)
        robot_data = torch.tensor(episode_group['robot'][0], dtype=torch.float32, device=device)
        
        # Combine into phystwin format
        phystwin_states = torch.cat([object_data, robot_data], dim=0)  # [n_particles, 3]
        
        # Create robot mask (False for object, True for robot)
        n_object = len(object_data)
        n_robot = len(robot_data)
        phystwin_robot_mask = torch.cat([
            torch.zeros(n_object, dtype=torch.bool, device=device),
            torch.ones(n_robot, dtype=torch.bool, device=device)
        ], dim=0)
    
    print(f"Loaded episode {episode_idx}:")
    print(f"  Object particles: {n_object}")
    print(f"  Robot particles: {n_robot}")
    print(f"  Total particles: {len(phystwin_states)}")
    
    # Initialize CuriosityPlanner
    model_name = "dropout_0.2"
    model_paths = get_model_paths(model_name)
    model_path = str(model_paths['net_best'])
    train_config_path = str(model_paths['config'])
    mpc_config_path = "GNN/config/mpc/curiosity.yaml"
    
    curiosity_planner = CuriosityPlanner(
        model_path=model_path,
        train_config_path=train_config_path,
        mpc_config_path=mpc_config_path,
        phystwin_states=phystwin_states,
        phystwin_robot_mask=phystwin_robot_mask,
        case_name="single_push_rope"
    )
        
    # Test explore method
    print("\nTesting explore method...")
    output_file = "GNN/test_curiosity_output.h5"
    
    try:
        curiosity_planner.explore(output_file)
        print(f"Explore method completed successfully!")
    except Exception as e:
        print(f"Error in explore method: {e}")
    
    return curiosity_planner

if __name__ == "__main__":
    test_curiosity_planner() 