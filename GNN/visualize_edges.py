from .utils import load_yaml, construct_edges_from_numpy, construct_edges_with_attrs, visualize_edges
import argparse
from .paths import CONFIG_TRAIN_GNN_DYN
import h5py
import torch
import open3d as o3d

"""
Visualize edges in a specified episode and frame

Args:
    --config: str - path to the config file. GNN/config/train/gnn_dyn.yaml by default. 
    --data_file: str - path to the data directory. config['dataset']['file'] by default. 
    --episode: int - episode number. Required.
    --frame: int - frame number. Required.
"""


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=str(CONFIG_TRAIN_GNN_DYN))
    parser.add_argument('--data_file', type=str, default=None)
    parser.add_argument('--episode', type=int, required=True)
    parser.add_argument('--frame', type=int, required=True)

    args = parser.parse_args()

    config = load_yaml(args.config)
    data_file = args.data_file if args.data_file is not None else config['dataset']['file']
    episode = args.episode
    frame = args.frame

    # Load data
    with h5py.File(data_file, 'r') as f:
        data = f[f'episode_{episode:06d}']
        object = torch.tensor(data['object'])[frame]
        robot = torch.tensor(data['robot'])[frame]
        object_edges = torch.tensor(data['object_edges'])
    
    # print(object.shape, robot.shape, object_edges.shape)
    positions = torch.cat([object, robot], dim=0)
    tool_mask = torch.zeros(positions.shape[0], dtype=torch.bool, device=positions.device)
    tool_mask[object.shape[0]:] = True
    topological_edges = torch.zeros(positions.shape[0], positions.shape[0], device=positions.device)
    topological_edges[:object.shape[0], :object.shape[0]] = object_edges
    adj_thresh = config['train']['edges']['collision']['adj_thresh']
    topk = config['train']['edges']['collision']['topk']
    colors = [[1.0, 0.6, 0.2], [0.3, 0.6, 0.3]]

    # Visualize edges
    line_sets = visualize_edges(positions, topological_edges, tool_mask, adj_thresh, topk, False, colors)
    o3d.visualization.draw_geometries(line_sets)