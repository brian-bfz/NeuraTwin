"""
Central path configuration for GNN package.
All paths are calculated relative to the GNN package location.
"""
from pathlib import Path

# Get the directory where this file (paths.py) is located - this is the GNN package root
GNN_ROOT = Path(__file__).parent

# Config directories
CONFIG_ROOT = GNN_ROOT / "config"
CONFIG_TRAIN_GNN_DYN = CONFIG_ROOT / "train" / "gnn_dyn.yaml"

# Data directories
DATA_ROOT = GNN_ROOT / "data"
GNN_DYN_MODEL_ROOT = DATA_ROOT / "gnn_dyn_model"
VIDEO_ROOT = DATA_ROOT / "video"

def get_model_paths(model_name):
    """
    Get paths for a specific GNN model.
    
    Args:
        model_name: str - name of the model
        
    Returns:
        dict - dictionary containing model-related paths
    """
    model_dir = GNN_DYN_MODEL_ROOT / model_name
    return {
        'model_dir': model_dir,
        'net_best': model_dir / "net_best.pth",
        'config': model_dir / "config.yaml",
        'video_dir': VIDEO_ROOT / model_name,
    } 