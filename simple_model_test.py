import torch
import numpy as np
from dataset.dataset_gnn_dyn import ParticleDataset
from model.gnn_dyn import PropNetDiffDenModel
from utils import load_yaml

def test_model():
    """Simple test of trained GNN model"""
    
    # Load config and model
    config = load_yaml('config/train/gnn_dyn.yaml')
    model_path = "data/gnn_dyn_model/2025-05-29-14-52-41-654478/net_best.pth"
    
    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = PropNetDiffDenModel(config, torch.cuda.is_available())
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    print(f"Loaded model from: {model_path}")
    print(f"Using device: {device}")
    
    # Load test data
    test_dataset = ParticleDataset(config['train']['data_root'], config, phase='valid')
    print(f"Test dataset size: {len(test_dataset)}")
    
    # Test on a few episodes
    total_error = 0
    num_tests = 0
    
    for episode_idx in range(min(3, len(test_dataset))):
        states, states_delta, attrs, particle_num = test_dataset[episode_idx]
        
        print(f"\nTesting episode {episode_idx} (particles: {particle_num})")
        
        episode_errors = []
        
        # Test prediction for first 5 timesteps
        for t in range(min(5, states.shape[0] - 1)):
            # Current state
            s_cur = states[t][:particle_num].unsqueeze(0).to(device)
            a_cur = attrs[t][:particle_num].unsqueeze(0).to(device)
            s_delta = states_delta[t][:particle_num].unsqueeze(0).to(device)
            
            # Ground truth next state
            s_next_gt = states[t + 1][:particle_num]
            
            # Predict next state
            with torch.no_grad():
                s_next_pred = model.predict_one_step(a_cur, s_cur, s_delta)
                s_next_pred = s_next_pred.squeeze(0).cpu()
            
            # Calculate error
            error = torch.nn.functional.mse_loss(s_next_pred, s_next_gt).item()
            episode_errors.append(error)
            total_error += error
            num_tests += 1
            
            print(f"  Timestep {t}: MSE = {error:.6f}")
        
        avg_episode_error = np.mean(episode_errors)
        print(f"  Episode {episode_idx} average MSE: {avg_episode_error:.6f}")
    
    # Overall results
    overall_avg_error = total_error / num_tests
    print(f"\n{'='*50}")
    print(f"OVERALL RESULTS:")
    print(f"Average MSE: {overall_avg_error:.6f}")
    print(f"RMSE: {np.sqrt(overall_avg_error):.6f}")
    print(f"{'='*50}")
    
    # Interpret results
    if overall_avg_error < 0.001:
        print("🎉 Excellent! Very low prediction error.")
    elif overall_avg_error < 0.01:
        print("✅ Good! Reasonable prediction accuracy.")
    elif overall_avg_error < 0.1:
        print("⚠️  Moderate prediction error. Model may need more training.")
    else:
        print("❌ High prediction error. Model needs improvement.")

if __name__ == "__main__":
    test_model() 