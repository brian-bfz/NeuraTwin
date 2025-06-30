from .inference import InferenceEngine, InferenceDataset
import torch
import argparse
from scripts.utils import parse_episodes
from .utils.math_ops import calculate_NLL
from torch.utils.data import DataLoader
from .utils.training import collate_fn
from .utils import load_yaml
import numpy as np

def predict_ensemble(models, states, states_delta, attrs, particle_nums, topological_edges, first_states):
    """
    Make predictions using an ensemble of models, then calculate mean and variance of the predictions
    Args:
        models: list[InferenceEngine] - ensemble of models
        states: [n_sample, n_timesteps, n_particles, 3] - batched episode states with history padding
        states_delta: [n_sample, n_timesteps-1, n_particles, 3] - batched particle displacements
        attrs: [n_sample, n_timesteps, n_particles] - batched particle attributes
        particle_nums: [n_sample] - particle counts per sample
        topological_edges: [n_sample, n_particles, n_particles] - batched adjacency matrices
        first_states: [n_sample, n_particles, 3] - batched first frame states

    Returns:
        means: [n_sample, n_timesteps, n_particles, 3] - mean of the predictions
        variances: [n_sample, n_timesteps, n_particles, 3] - variance of the predictions
    """
    predicted_states = []
    for model in models:
        predicted_states.append(model.predict_batch_rollout(states, states_delta, attrs, particle_nums, topological_edges, first_states))
    
    # Stack predictions from all models
    stacked_predictions = torch.stack(predicted_states, dim=0)  # [n_models, n_sample, n_timesteps, n_particles, 3]

    # Calculate mean and variance across models
    means = torch.mean(stacked_predictions, dim=0)  # [n_sample, n_timesteps, n_particles, 3]
    variances = torch.var(stacked_predictions, dim=0)  # [n_sample, n_timesteps, n_particles, 3]
    
    return means, variances

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_paths", type=str, nargs="+", required=True,
                       help="Paths to the models")
    parser.add_argument("--config_path", type=str, required=True,
                       help="Path to the config file")
    parser.add_argument("--data_file", type=str, default=None,
                       help="Path to the data file (optional)")
    parser.add_argument("--episodes", nargs='+', type=str, default=["0-4"],
                       help="Episodes to test. Format: space-separated list (0 1 2 3 4) or range (0-4)")
    args = parser.parse_args()

    # Parse episodes
    test_episodes = parse_episodes(args.episodes)

    # Load ensemble of models
    models = []
    for model_path in args.model_paths:
        model = InferenceEngine(model_path, args.config_path, args.data_file)
        models.append(model)
    
    print(f"Loaded {len(models)} models for ensemble prediction")
    
    dataset = InferenceDataset(models[0].dataset, test_episodes)
    dataloader = DataLoader(
        dataset,
        batch_size=512,
        shuffle=False,
        num_workers=10,
        collate_fn=collate_fn
    )

    print("="*60)
    print("ENSEMBLE UNCERTAINTY TEST")
    print("="*60)
    print(f"Number of models: {len(models)}")
    print(f"Episodes to test: {test_episodes}")

    all_nlls = []
    
    for batch_idx, data in enumerate(dataloader):
        states, states_delta, attrs, particle_nums, topological_edges, first_states = data
        
        # Move data to device (use first model's device)
        device = models[0].device
        states = states.to(device)
        states_delta = states_delta.to(device)
        attrs = attrs.to(device)
        particle_nums = particle_nums.to(device)
        topological_edges = topological_edges.to(device)
        first_states = first_states.to(device)

        with torch.no_grad():
            means, variances = predict_ensemble(models, states, states_delta, attrs, particle_nums, topological_edges, first_states)
            
            # Calculate NLL for this batch, passing particle_nums for proper handling
            batch_nll = calculate_NLL(states, means, variances, models[0].n_history, particle_nums)
            all_nlls.append(batch_nll)
            
            print(f"Batch {batch_idx + 1}: NLL = {batch_nll:.6f}")

    if all_nlls:
        print(f"\n{'='*60}")
        print("OVERALL RESULTS")
        print(f"{'='*60}")
        mean_nll = np.mean(all_nlls)
        print(f"Episodes tested: {len(test_episodes)}")
        print(f"Number of batches: {len(all_nlls)}")
        print(f"Mean NLL: {mean_nll:.6f}")