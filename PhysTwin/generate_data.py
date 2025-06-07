# import sys
# sys.path.append("./gaussian_splatting")
from .qqtt import InvPhyTrainerWarp
from .qqtt.utils import logger, cfg
from .paths import *
from datetime import datetime
import random
import numpy as np
import torch
from argparse import ArgumentParser
import glob
import os
import pickle
from .SampleRobot import RobotPcSampler
import h5py
import sys
from scripts.utils import parse_episodes

# def set_all_seeds(seed):
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False


# seed = 
# set_all_seeds(seed)

import warnings
warnings.simplefilter("ignore")

random.seed()

def random_movement(n_ctrl_parts, num_movements=10, frames_per_movement=10):
    # Define possible keys for each hand
    hand1_keys = ['w', 's', 'a', 'd']  # Left hand keys
    hand2_keys = ['i', 'k', 'j', 'l']  # Right hand keys
    
    sequence = []

    # temporary code to set position
    # movement = []
    # movement.append("m")
    # movement.append("d")
    # sequence.extend([movement] * 2 * frames_per_movement)

    for _ in range(num_movements):
        # Generate random movements for each hand
        movement = []
        # Left hand movement
        movement.append(random.choice(hand1_keys))
        
        # Right hand movement (if n_ctrl_parts > 1)
        if n_ctrl_parts > 1:
            movement.append(random.choice(hand2_keys))
        
        # Repeat this movement for frames_per_movement frames
        sequence.extend([movement] * frames_per_movement)
    
    return sequence

def initialize_data_file(data_file_path):
    """Initialize the shared HDF5 data file if it doesn't exist"""
    if not os.path.exists(data_file_path):
        # Create empty HDF5 file
        with h5py.File(data_file_path, 'w') as f:
            # Add global metadata
            f.attrs['created'] = datetime.now().isoformat()
            f.attrs['description'] = 'PhysTwin episode data collection'
        print(f"Initialized data file: {data_file_path}")
    else:
        print(f"Using existing data file: {data_file_path}")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--base_path",
        type=str,
        default=str(DATA_DIFFERENT_TYPES),
    )
    parser.add_argument(
        "--gaussian_path",
        type=str,
        default=str(GAUSSIAN_OUTPUT_DIR),
    )
    parser.add_argument("--data_file", type=str, default=str(GENERATED_DATA_DIR / "data.h5"))
    parser.add_argument("--case_name", type=str, default="double_lift_cloth_3")
    parser.add_argument("--n_ctrl_parts", type=int, default=1)
    parser.add_argument("--custom_ctrl_points", type=str, help="Path to directory containing custom control points")
    parser.add_argument("--episodes", nargs='+', type=str, required=True,
                       help="Episodes to generate. Format: space-separated list (0 1 2 3 4) or range (0-4)")
    parser.add_argument("--include_gaussian", action="store_true")
    parser.add_argument("--overwrite", action="store_true",
                       help="Overwrite existing episode data if it already exists")
    args = parser.parse_args()

    # Parse episode specification
    try:
        episode_list = parse_episodes(args.episodes)
    except ValueError as e:
        print(f"Error parsing episodes: {e}")
        print("Examples:")
        print("  Space-separated: --episodes 0 1 2 3 4")
        print("  Range format: --episodes 0-4")
        sys.exit(1)

    base_path = args.base_path
    case_name = args.case_name

    if "cloth" in case_name or "package" in case_name:
        cfg.load_from_yaml(str(CONFIG_CLOTH))
    else:
        cfg.load_from_yaml(str(CONFIG_REAL))
    print(cfg.__dict__)

    case_paths = get_case_paths(case_name)
    base_dir = str(case_paths['base_dir'])

    # Read the first-satage optimized parameters to set the indifferentiable parameters
    optimal_path = str(case_paths['optimal_params'])
    logger.info(f"Load optimal parameters from: {optimal_path}")
    assert os.path.exists(
        optimal_path
    ), f"{case_name}: Optimal parameters not found: {optimal_path}"
    with open(optimal_path, "rb") as f:
        optimal_params = pickle.load(f)
    cfg.set_optimal_params(optimal_params)

    # Load the robot finger
    urdf_path = str(URDF_XARM7)
    R = np.array([[0.0, -1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, -1.0]])

    init_pose = np.eye(4)
    init_pose[:3, :3] = R
    init_pose[:3, 3] = [0.0, 0.0, 0.0]

    exp_name = "init=hybrid_iso=True_ldepth=0.001_lnormal=0.0_laniso_0.0_lseg=1.0"
    gaussians_path = f"{args.gaussian_path}/{case_name}/{exp_name}/point_cloud/iteration_10000/point_cloud.ply"

    logger.set_log_file(path=base_dir, name="inference_log")

    best_model_path = glob.glob(str(case_paths['model_dir'] / "best_*.pth"))[0]
    
    # Create timestamped folder for this simulation run
    # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # print(case_name, timestamp, f"{case_name}_{timestamp}")
    sample_robot = RobotPcSampler(
        urdf_path, link_names=["left_finger", "right_finger"], init_pose=init_pose
    )
    trainer = InvPhyTrainerWarp(
        data_path=f"{base_path}/{case_name}/final_data.pkl",
        base_dir=base_dir,
        pure_inference_mode=True,
        static_meshes=[],
        robot=sample_robot,
        include_gaussian=args.include_gaussian,
    )

    # Initialize shared data file
    initialize_data_file(args.data_file)

    # Generate episodes
    for i in episode_list:
        # Check if episode already exists and handle overwrite
        if not args.overwrite:
            try:
                with h5py.File(args.data_file, 'r') as f:
                    if f'episode_{i:06d}' in f:
                        print(f"Episode {i} already exists, skipping (use --overwrite to replace)")
                        continue
            except (FileNotFoundError, OSError):
                pass  # File doesn't exist yet, continue
        
        trainer.generate_data(
            best_model_path, 
            gaussians_path, 
            args.n_ctrl_parts, 
            args.data_file,
            i,
            overwrite=args.overwrite
        )

    # Load custom control points if provided
    # custom_control_points = None
    # if args.custom_ctrl_points:
    #     cp_path = os.path.join(args.custom_ctrl_points, "cp.pt")
    #     if not os.path.exists(cp_path):
    #         raise FileNotFoundError(f"Control points file not found: {cp_path}")
    #     custom_control_points = torch.load(cp_path)
    #     logger.info(f"Loaded custom control points from {cp_path}")

    # pressed_keys_sequence = random_movement(args.n_ctrl_parts)
    # print(pressed_keys_sequence)

    
