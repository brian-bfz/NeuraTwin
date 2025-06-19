
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
import torch
import torch.multiprocessing as mp
from torch.distributed import init_process_group, destroy_process_group

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

def ddp_setup(rank, world_size):
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

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

def initialize_data_file(data_file_path, rank=None):
    """Initialize the HDF5 data file for a specific GPU if it doesn't exist"""
    if not os.path.exists(data_file_path):
        # Create empty HDF5 file
        with h5py.File(data_file_path, 'w') as f:
            # Add global metadata
            f.attrs['created'] = datetime.now().isoformat()
            f.attrs['description'] = 'PhysTwin episode data collection'
            if rank is not None:
                f.attrs['gpu_rank'] = rank
        print(f"Initialized data file: {data_file_path}" + (f" for GPU {rank}" if rank is not None else ""))
    else:
        print(f"Using existing data file: {data_file_path}" + (f" for GPU {rank}" if rank is not None else ""))

def generate_episodes_distributed(rank=None, world_size=None, case_name=None, base_path=None, 
                                bg_img_path=None, gaussian_path=None, n_ctrl_parts=1, 
                                episode_list=None, include_gaussian=False, save_dir=None):
    """
    Distributed episode generation function.
    
    Args:
        rank: Process rank for distributed processing (None for single GPU)
        world_size: Total number of processes for distributed processing
        case_name: Name of the case/experiment
        base_path: Base path for data
        bg_img_path: Path to background image
        gaussian_path: Path to gaussian output directory
        n_ctrl_parts: Number of control parts
        episode_list: List of episodes to generate
        include_gaussian: Whether to include gaussian data
        save_dir: Directory to save data files
    """
    
    # ========================================================================
    # DISTRIBUTED SETUP - MUST BE FIRST
    # ========================================================================
    if rank is not None:
        ddp_setup(rank, world_size)
        device = f'cuda:{rank}'
        print(f"Process {rank}/{world_size} starting on device {device}")
    else:
        device = 'cuda'
        print("Single GPU processing")

    # ========================================================================
    # EPISODE DISTRIBUTION ACROSS GPUS
    # ========================================================================
    if rank is not None:
        # Distribute episodes across GPUs
        episodes_per_gpu = len(episode_list) // world_size
        remainder = len(episode_list) % world_size
        
        start_idx = rank * episodes_per_gpu + min(rank, remainder)
        end_idx = start_idx + episodes_per_gpu + (1 if rank < remainder else 0)
        
        local_episode_list = episode_list[start_idx:end_idx]
        print(f"GPU {rank} processing episodes {start_idx}-{end_idx-1}: {local_episode_list}")
        
        # Create separate data file for each GPU
        data_file_path = str(save_dir / f"lift_data_gpu{rank}.h5")
    else:
        local_episode_list = episode_list
        print(f"Single GPU processing all episodes: {local_episode_list}")
        
        # Single GPU uses original filename
        data_file_path = str(save_dir / "lift_data.h5")

    # Initialize data file for this process/GPU
    initialize_data_file(data_file_path, rank)

    # ========================================================================
    # CONFIGURATION SETUP
    # ========================================================================
    
    # Initialize configuration - this replaces ~50 lines of setup code
    config = PhysTwinConfig(
        case_name=case_name,
        base_path=base_path,
        bg_img_path=bg_img_path,
        gaussian_path=gaussian_path,
        inference=True
    )
    
    # Create trainer with device-specific robot controller
    trainer = InvPhyTrainerWarp(
        data_path=config.get_data_path(),
        base_dir=str(config.case_paths['base_dir']),
        pure_inference_mode=True,
        static_meshes=[],
        robot_controller=config.get_robot_controller("default", n_ctrl_parts=1, device=device),
        include_gaussian=include_gaussian,
        device=device,
    )

    # ========================================================================
    # EPISODE GENERATION
    # ========================================================================
    
    # Get model paths
    best_model_path = config.get_best_model_path()
    gaussians_path = config.get_gaussian_path()
    
    for i in local_episode_list:
        if rank is not None:
            print(f"GPU {rank} generating episode {i}")
        else:
            print(f"Generating episode {i}")
            
        trainer.generate_data(
            best_model_path, 
            gaussians_path, 
            n_ctrl_parts, 
            data_file_path,
            i,
        )

    # ========================================================================
    # CLEANUP
    # ========================================================================
    if rank is not None:
        destroy_process_group()
        print(f"GPU {rank} finished processing")

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

    # Setup save directory
    save_dir = GENERATED_DATA_DIR
    os.makedirs(save_dir, exist_ok=True)

    # ========================================================================
    # MULTI-GPU SETUP AND EXECUTION
    # ========================================================================
    
    world_size = torch.cuda.device_count()
    if world_size > 1:
        print(f"Using {world_size} GPUs for distributed episode generation")
        print(f"Total episodes to generate: {len(episode_list)}")
        print(f"Each GPU will write to separate data files: lift_data_gpu0.h5, lift_data_gpu1.h5, etc.")
        
        # Use multiprocessing to spawn processes for each GPU
        mp.spawn(
            generate_episodes_distributed, 
            nprocs=world_size, 
            args=(
                world_size,
                args.case_name,
                args.base_path, 
                args.bg_img_path,
                args.gaussian_path,
                args.n_ctrl_parts,
                episode_list,
                args.include_gaussian,
                save_dir
            )
        )
    else:
        print("Using single GPU for episode generation")
        generate_episodes_distributed(
            rank=None,
            world_size=None,
            case_name=args.case_name,
            base_path=args.base_path,
            bg_img_path=args.bg_img_path,
            gaussian_path=args.gaussian_path,
            n_ctrl_parts=args.n_ctrl_parts,
            episode_list=episode_list,
            include_gaussian=args.include_gaussian,
            save_dir=save_dir
        )

    print("All episode generation completed!")
    
    # Print summary of generated files
    if world_size > 1:
        print("\nGenerated data files:")
        for rank in range(world_size):
            data_file = save_dir / f"lift_data_gpu{rank}.h5"
            if data_file.exists():
                print(f"  GPU {rank}: {data_file}")
    else:
        data_file = save_dir / "lift_data.h5"
        if data_file.exists():
            print(f"\nGenerated data file: {data_file}")

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

    
