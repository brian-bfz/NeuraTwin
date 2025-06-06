"""
Central path configuration for PhysTwin package.
All paths are calculated relative to the PhysTwin package location.
"""
from pathlib import Path

# Get the directory where this file (paths.py) is located - this is the PhysTwin package root
PHYSTWIN_ROOT = Path(__file__).parent

# Data directories
DATA_ROOT = PHYSTWIN_ROOT / "data"
DATA_DIFFERENT_TYPES = DATA_ROOT / "different_types"
DATA_BG_IMG = DATA_ROOT / "bg.png"

# Config directories
CONFIGS_ROOT = PHYSTWIN_ROOT / "configs"
CONFIG_CLOTH = CONFIGS_ROOT / "cloth.yaml"
CONFIG_REAL = CONFIGS_ROOT / "real.yaml"

# Output directories
GENERATED_DATA_DIR = PHYSTWIN_ROOT / "generated_data"
GENERATED_VIDEOS_DIR = PHYSTWIN_ROOT / "generated_videos"
TEMP_EXPERIMENTS_DIR = PHYSTWIN_ROOT / "temp_experiments"

# Model and optimization directories
EXPERIMENTS_DIR = PHYSTWIN_ROOT / "experiments"
EXPERIMENTS_OPTIMIZATION_DIR = PHYSTWIN_ROOT / "experiments_optimization"
GAUSSIAN_OUTPUT_DIR = PHYSTWIN_ROOT / "gaussian_output"

# Robot assets
URDF_XARM7 = PHYSTWIN_ROOT / "xarm" / "xarm7_with_gripper.urdf"

def get_case_paths(case_name):
    """
    Get all paths related to a specific case/experiment.
    
    Args:
        case_name: str - name of the case/experiment
        
    Returns:
        dict - dictionary containing all relevant paths for the case
    """
    return {
        'base_dir': TEMP_EXPERIMENTS_DIR / case_name,
        'optimal_params': EXPERIMENTS_OPTIMIZATION_DIR / case_name / "optimal_params.pkl",
        'model_dir': EXPERIMENTS_DIR / case_name / "train",
        'data_dir': DATA_DIFFERENT_TYPES / case_name,
    }

def get_gaussian_path(gaussian_path, case_name, exp_name="init=hybrid_iso=True_ldepth=0.001_lnormal=0.0_laniso_0.0_lseg=1.0"):
    """
    Get the path to gaussian splatting point cloud file.
    
    Args:
        gaussian_path: str - base gaussian output path
        case_name: str - case name
        exp_name: str - experiment name
        
    Returns:
        Path - path to the point cloud file
    """
    if gaussian_path.startswith('./'):
        # Convert relative path to absolute based on PhysTwin root
        gaussian_base = PHYSTWIN_ROOT / gaussian_path[2:]
    else:
        gaussian_base = Path(gaussian_path)
    
    return gaussian_base / case_name / exp_name / "point_cloud" / "iteration_10000" / "point_cloud.ply" 