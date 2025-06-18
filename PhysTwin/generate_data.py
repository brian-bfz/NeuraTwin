# import sys
# sys.path.append("./gaussian_splatting")
from .qqtt import InvPhyTrainerWarp
from .qqtt.utils import logger, cfg
from .paths import *
from .config_manager import PhysTwinConfig, create_common_parser
from datetime import datetime
import random
import os
import h5py
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
    # Create parser with common arguments
    parser = create_common_parser()
    
    # Add script-specific arguments
    parser.add_argument("--n_ctrl_parts", type=int, default=1)
    parser.add_argument("--custom_ctrl_points", type=str, help="Path to directory containing custom control points")
    parser.add_argument("--episodes", nargs='+', type=str, required=True,
                       help="Episodes to generate. Format: space-separated list (0 1 2 3 4) or range (0-4)")
    parser.add_argument("--include_gaussian", action="store_true")
    args = parser.parse_args()

    episode_list = parse_episodes(args.episodes)

    # Initialize configuration - this replaces ~50 lines of setup code
    config = PhysTwinConfig(
        case_name=args.case_name,
        base_path=args.base_path,
        bg_img_path=args.bg_img_path,
        gaussian_path=args.gaussian_path
    )
    
    # Create trainer
    trainer = InvPhyTrainerWarp(
        data_path=config.get_data_path(),
        base_dir=str(config.case_paths['base_dir']),
        pure_inference_mode=True,
        static_meshes=[],
        robot_loader=config.create_robot_loader(),
        robot_initial_pose=config.get_robot_initial_pose("default"),
        include_gaussian=args.include_gaussian,
    )

    # Initialize shared data file
    save_dir = GENERATED_DATA_DIR
    os.makedirs(save_dir, exist_ok=True)
    data_file_path = str(save_dir / "data.h5")
    initialize_data_file(data_file_path)

    # Generate episodes
    best_model_path = config.get_best_model_path()
    gaussians_path = config.get_gaussian_path()
    
    for i in episode_list:
        trainer.generate_data(
            best_model_path, 
            gaussians_path, 
            args.n_ctrl_parts, 
            data_file_path,
            i,
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

    
