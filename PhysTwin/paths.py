"""
Central path configuration for PhysTwin package.
All paths are calculated relative to the PhysTwin package location.
"""
from pathlib import Path

# Get the directory where this file (paths.py) is located - this is the PhysTwin package root
PHYSTWIN_ROOT = Path(__file__).parent

# Assets directories
ASSETS_ROOT = PHYSTWIN_ROOT / "assets"

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