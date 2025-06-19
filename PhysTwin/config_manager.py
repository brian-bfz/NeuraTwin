"""
Configuration management for PhysTwin system.
Centralizes all repeated configuration setup code from scripts.
"""
import os
import glob
import pickle
import json
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple

from .qqtt.utils import logger, cfg
from .paths import (
    CONFIG_CLOTH, CONFIG_REAL, DATA_DIFFERENT_TYPES, 
    get_case_paths, URDF_XARM7, GAUSSIAN_OUTPUT_DIR,
    DATA_BG_IMG, TEMP_EXPERIMENTS_DIR
)
from .robot import RobotLoader, RobotController


class PhysTwinConfig:
    """Centralized configuration management for PhysTwin system"""
    
    def __init__(self, case_name: str, base_path: Optional[str] = None, 
                 bg_img_path: Optional[str] = None, gaussian_path: Optional[str] = None, inference: Optional[bool] = True):
        """
        Initialize configuration for a specific case
        
        Args:
            case_name: Name of the case/experiment
            base_path: Base path for data (defaults to DATA_DIFFERENT_TYPES)
            bg_img_path: Path to background image (defaults to DATA_BG_IMG)
            gaussian_path: Path to gaussian output directory (defaults to GAUSSIAN_OUTPUT_DIR)
        """
        self.case_name = case_name
        self.base_path = base_path or str(DATA_DIFFERENT_TYPES)
        self.bg_img_path = bg_img_path or str(DATA_BG_IMG)
        self.gaussian_path = gaussian_path or str(GAUSSIAN_OUTPUT_DIR)
        
        # Initialize paths and configuration
        self.case_paths = get_case_paths(case_name)
        self._setup_config()
        self._load_optimal_params()
        self._load_calibration_data()
        if inference: 
            self.setup_logging("inference_log")

    def _setup_config(self) -> None:
        """Load case-specific configuration (cloth vs real)"""
        if "cloth" in self.case_name or "package" in self.case_name:
            cfg.load_from_yaml(str(CONFIG_CLOTH))
        else:
            cfg.load_from_yaml(str(CONFIG_REAL))
        
        logger.info(f"Loaded configuration for case: {self.case_name}")
        logger.info(f"Data type: {cfg.data_type}")
    
    def _load_optimal_params(self) -> None:
        """Load and set optimal parameters"""
        optimal_path = str(self.case_paths['optimal_params'])
        logger.info(f"Loading optimal parameters from: {optimal_path}")
        
        if not os.path.exists(optimal_path):
            raise FileNotFoundError(f"{self.case_name}: Optimal parameters not found: {optimal_path}")
        
        with open(optimal_path, "rb") as f:
            optimal_params = pickle.load(f)
        cfg.set_optimal_params(optimal_params)
        
        logger.info("Optimal parameters loaded and set")
    
    def _load_calibration_data(self) -> None:
        """Load camera calibration data"""
        calibrate_path = f"{self.base_path}/{self.case_name}/calibrate.pkl"
        metadata_path = f"{self.base_path}/{self.case_name}/metadata.json"
        
        # Load camera poses
        with open(calibrate_path, "rb") as f:
            c2ws = pickle.load(f)
        w2cs = [np.linalg.inv(c2w) for c2w in c2ws]
        cfg.c2ws = np.array(c2ws)
        cfg.w2cs = np.array(w2cs)
        
        # Load camera intrinsics and metadata
        with open(metadata_path, "r") as f:
            data = json.load(f)
        cfg.intrinsics = np.array(data["intrinsics"])
        cfg.WH = data["WH"]
        
        # Set background image path
        cfg.bg_img_path = self.bg_img_path
        
        logger.info("Camera calibration data loaded")
    
    def setup_logging(self, log_name: str = "inference_log") -> None:
        """Setup logging for the session"""
        base_dir = str(self.case_paths['base_dir'])
        logger.set_log_file(path=base_dir, name=log_name)
    
    def get_best_model_path(self) -> str:
        """Get path to the best trained model"""
        model_pattern = str(self.case_paths['model_dir'] / "best_*.pth")
        model_files = glob.glob(model_pattern)
        if not model_files:
            raise FileNotFoundError(f"No best model found in {self.case_paths['model_dir']}")
        return model_files[0]
    
    def get_gaussian_path(self, exp_name: str = "init=hybrid_iso=True_ldepth=0.001_lnormal=0.0_laniso_0.0_lseg=1.0") -> str:
        """Get path to gaussian splatting model"""
        return f"{self.gaussian_path}/{self.case_name}/{exp_name}/point_cloud/iteration_10000/point_cloud.ply"
    
    def get_data_path(self) -> str:
        """Get path to final data file"""
        return str(self.case_paths['data_dir'] / "final_data.pkl")
    
    def get_temp_base_dir(self) -> str:
        """Get temporary base directory for experiments"""
        return str(TEMP_EXPERIMENTS_DIR / self.case_name)
    
    def get_robot_controller(self, robot_type: str = "default", n_ctrl_parts: int = 1, device: str = 'cuda'):
        """
        Create a robot controller with the correct initial pose.
        
        Args:
            robot_type: Type of robot configuration ("default", "interactive", "video")
            n_ctrl_parts: Number of control parts (default: 1)
            device: PyTorch device (e.g., 'cuda', 'cuda:0', 'cuda:1')
            
        Returns:
            RobotController: Configured robot controller instance
        """
        
        # Get initial pose
        R = np.array([[0.0, -1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, -1.0]])
        
        init_pose = np.eye(4)
        init_pose[:3, :3] = R

        if robot_type == "interactive":
            init_pose[:3, 3] = [0.2, 0.0, 0.23]
        else:
            init_pose[:3, 3] = [0.0, 0.0, 0.0]
        
        # Create robot loader with transform
        urdf_path = str(URDF_XARM7)
        robot_loader = RobotLoader(
            urdf_path, 
            link_names=["left_finger", "right_finger"],
            transform=init_pose
        )
        
        # Create robot controller with robot loader and specified device
        return RobotController(robot_loader, n_ctrl_parts, device)
    
    def get_paths(self) -> Dict[str, Path]:
        """Get all relevant paths for the case"""
        return self.case_paths.copy()
    
    def to_dict(self) -> Dict:
        """Convert configuration to dictionary for logging/debugging"""
        return {
            'case_name': self.case_name,
            'base_path': self.base_path,
            'bg_img_path': self.bg_img_path,
            'gaussian_path': self.gaussian_path,
            'case_paths': {k: str(v) for k, v in self.case_paths.items()},
            'cfg_dict': cfg.to_dict()
        }


def create_common_parser():
    """Create argument parser with common options for all scripts"""
    from argparse import ArgumentParser
    
    parser = ArgumentParser()
    parser.add_argument(
        "--base_path",
        type=str,
        default=str(DATA_DIFFERENT_TYPES),
        help="Base path for data"
    )
    parser.add_argument(
        "--case_name", 
        type=str, 
        required=True,
        help="Name of the case/experiment"
    )
    parser.add_argument(
        "--bg_img_path",
        type=str,
        default=str(DATA_BG_IMG),
        help="Path to background image"
    )
    parser.add_argument(
        "--gaussian_path",
        type=str,
        default=str(GAUSSIAN_OUTPUT_DIR),
        help="Path to gaussian output directory"
    )
    
    return parser 