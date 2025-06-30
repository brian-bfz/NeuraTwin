import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from .dataset.dataset_gnn_dyn import ParticleDataset
from .model.gnn_dyn import PropNetDiffDenModel
from .model.rollout import Rollout
from .utils import load_yaml, Visualizer
from .utils.training import collate_fn
from .paths import *
import argparse
from scripts.utils import parse_episodes


class InferenceDataset(Dataset):
    """
    Dataset wrapper for inference that loads full episodes.
    """
    def __init__(self, original_dataset, episode_indices):
        self.original_dataset = original_dataset
        self.episode_indices = episode_indices

    def __len__(self):
        return len(self.episode_indices)

    def __getitem__(self, idx):
        episode_num = self.episode_indices[idx]
        return self.original_dataset.load_full_episode(episode_num)


class InferenceEngine:
    """
    GNN-based model inference engine for object motion prediction.
    Handles loading trained models and generating predictions with error calculation.
    """
    
    def __init__(self, model_path, config_path, data_file):
        """
        Initialize the inference engine with trained model and data pipeline.
        
        Args:
            model_path: str - path to trained model checkpoint
            config_path: str - path to training configuration file
            data_file: str - path to HDF5 dataset file
        """
        self.config = load_yaml(config_path)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        if data_file is None:
            data_file = self.config['dataset']['file']
        self.dataset = ParticleDataset(data_file, self.config, 'train')
        
        self.n_history = self.config['train']['n_history']
        
        self.model = PropNetDiffDenModel(self.config, torch.cuda.is_available())
        self.model.load_state_dict(torch.load(model_path, weights_only=True, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

        self.downsample_rate = self.config['dataset']['downsample_rate']
        
        print(f"Loaded model from: {model_path}")
        print(f"Using device: {self.device}")

    def predict_batch_rollout(self, states, states_delta, attrs, particle_nums, topological_edges, first_states):
        """
        Predict object motion for a batch of episodes using autoregressive rollout.
        
        Args:
            states: [B, T, P, 3] - batched episode states with history padding
            states_delta: [B, T-1, P, 3] - batched particle displacements
            attrs: [B, T, P] - batched particle attributes
            particle_nums: [B] - particle counts per sample
            topological_edges: [B, P, P] - batched adjacency matrices
            first_states: [B, P, 3] - batched first frame states
            
        Returns:
            predicted_states: [B, T, P, 3] - complete predicted trajectories for the batch
        """
        n_rollout = states.shape[1] - self.n_history
        
        predictor = Rollout(
            self.model, 
            self.config,
            states[:, self.n_history - 1, :, :],
            states_delta[:, :self.n_history - 1, :, :],
            attrs[:, self.n_history - 1, :],
            particle_nums,
            topological_edges,
            first_states
        )
        
        predicted_states_list = [states[:, i, :, :].clone() for i in range(self.n_history)]
        
        for i in range(n_rollout):
            step_idx = i + self.n_history
            next_delta = states_delta[:, step_idx - 1, :, :]
            predicted_state = predictor.forward(next_delta)
            predicted_states_list.append(predicted_state)
        
        return torch.stack(predicted_states_list, dim=1)

def calculate_prediction_error(predicted_states, actual_states):
    """
    Calculate MSE errors between predicted and ground truth trajectories.
        
    Args:
        predicted_states: [timesteps, particles, 3] - predicted trajectory
        actual_states: [timesteps, particles, 3] - ground truth trajectory
            
    Returns:
        errors: list[float] - MSE error for each timestep
    """
    errors = []
    n_timesteps = min(len(predicted_states), len(actual_states))
        
    for t in range(n_timesteps):
        error = torch.nn.functional.mse_loss(predicted_states[t], actual_states[t]).item()
        errors.append(error)
            
    return errors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True,
                       help="Model name (e.g., 2025-05-31-21-01-09-427982 or custom_model_name)")
    parser.add_argument("--camera_calib_path", type=str, default="PhysTwin/data/different_types/single_push_rope")
    parser.add_argument("--episodes", nargs='+', type=str, default=["0-4"],
                       help="Episodes to test. Format: space-separated list (0 1 2 3 4) or range (0-4)")
    parser.add_argument("--data_file", type=str, default=None)
    parser.add_argument("--video", action='store_true',
                       help="Generate visualization videos (optional)")
    parser.add_argument("--config_path", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()
    
    try:
        test_episodes = parse_episodes(args.episodes)
    except ValueError as e:
        print(f"Error parsing episodes: {e}")
        return
    
    model_paths = get_model_paths(args.model)
    model_path = str(model_paths['net_best'])
    config_path = str(model_paths['config']) if args.config_path is None else args.config_path
    
    inference_engine = InferenceEngine(model_path, config_path, args.data_file)
    
    inference_dataset = InferenceDataset(inference_engine.dataset, test_episodes)
    dataloader = DataLoader(
        inference_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn
    )
    
    visualizer = Visualizer(args.camera_calib_path, inference_engine.downsample_rate) if args.video else None
    
    print("="*60)
    print("OBJECT MOTION PREDICTION TEST")
    print("="*60)
    print(f"Video generation: {'Enabled' if args.video else 'Disabled'}")
    
    all_errors = []
    
    for batch_idx, data in enumerate(dataloader):
        states, states_delta, attrs, particle_nums, topological_edges, first_states = data
        
        states = states.to(inference_engine.device)
        states_delta = states_delta.to(inference_engine.device)
        attrs = attrs.to(inference_engine.device)
        particle_nums = particle_nums.to(inference_engine.device)
        topological_edges = topological_edges.to(inference_engine.device)
        first_states = first_states.to(inference_engine.device)

        with torch.no_grad():
            predicted_states = inference_engine.predict_batch_rollout(states, states_delta, attrs, particle_nums, topological_edges, first_states)
            
            batch_size = states.shape[0]
            for i in range(batch_size):
                p_num = particle_nums[i].item()
                errors = calculate_prediction_error(predicted_states[i][:, :p_num, :], states[i][:, :p_num, :])
                all_errors.extend(errors)
                if args.video:

                    global_idx = batch_idx * args.batch_size + i
                    if global_idx >= len(test_episodes): continue
                    
                    episode_num = test_episodes[global_idx]

                    ep_predicted = predicted_states[i, :, :p_num, :].cpu()
                    ep_actual = states[i, :, :p_num, :].cpu()
                    ep_attrs = attrs[i, :, :p_num].cpu()
                    ep_topo_edges = topological_edges[i, :p_num, :p_num].cpu()
                    
                    n_obj = (ep_attrs[0] == 0).sum().item()
                    actual_objects = ep_actual[:, :n_obj, :]
                    tool_mask = (ep_attrs[0] > 0.5).bool()
                    
                    video_path = str(model_paths['video_dir'] / f"prediction_{episode_num}.mp4")
                    visualizer.visualize_object_motion(
                        ep_predicted, tool_mask, actual_objects, video_path, ep_topo_edges
                    )
    
    if all_errors:
        print(f"\n{'='*60}")
        print("OVERALL RESULTS")
        print(f"{'='*60}")
        rmse = np.sqrt(np.mean(all_errors))
        print(f"Episodes tested: {len(test_episodes)}")
        print(f"Total timesteps: {len(all_errors)}")
        print(f"Average MSE: {np.mean(all_errors):.6f}")
        print(f"RMSE: {rmse:.6f}")
        print(f"Std Dev: {np.std(all_errors):.6f}")
        
        if rmse < 0.003: print("🎉 Excellent prediction accuracy!")
        elif rmse < 0.01: print("✅ Good prediction accuracy!")
        elif rmse < 0.05: print("⚠️  Moderate prediction accuracy")
        else: print("❌ Poor prediction accuracy - model needs improvement")
        
        if args.video:
            print(f"\nVideos saved in '{model_paths['video_dir']}' directory")

if __name__ == "__main__":
    main()
