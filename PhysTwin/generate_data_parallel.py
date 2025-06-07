import multiprocessing as mp
import torch
import os
import sys
from concurrent.futures import ProcessPoolExecutor
import h5py
import numpy as np
from pathlib import Path

# Import the original generate_data components
from PhysTwin.qqtt import InvPhyTrainerWarp
from PhysTwin.qqtt.utils import logger, cfg
from PhysTwin.paths import *
from PhysTwin.SampleRobot import RobotPcSampler
from scripts.utils import parse_episodes
import glob
import pickle
from datetime import datetime
import warnings
import random

warnings.simplefilter("ignore")

def generate_episode_on_gpu(args_tuple):
    """
    Generate a single episode on a specified GPU.
    This function will be called in parallel processes.
    """
    episode_id, gpu_id, case_name, data_file_path, n_ctrl_parts, include_gaussian, base_path, overwrite = args_tuple
    
    # Set the GPU for this process
    device = f"cuda:{gpu_id}"
    # Don't restrict CUDA_VISIBLE_DEVICES - instead use explicit device assignment
    # os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)  # This causes the error
    
    print(f"Episode {episode_id} starting on GPU {gpu_id}")
    
    # Reload configuration (each process needs its own config)
    if "cloth" in case_name or "package" in case_name:
        cfg.load_from_yaml(str(CONFIG_CLOTH))
    else:
        cfg.load_from_yaml(str(CONFIG_REAL))
    
    case_paths = get_case_paths(case_name)
    base_dir = str(case_paths['base_dir'])
    
    # Load optimal parameters
    optimal_path = str(case_paths['optimal_params'])
    with open(optimal_path, "rb") as f:
        optimal_params = pickle.load(f)
    cfg.set_optimal_params(optimal_params)
    
    # Setup robot
    urdf_path = str(URDF_XARM7)
    R = np.array([[0.0, -1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, -1.0]])
    init_pose = np.eye(4)
    init_pose[:3, :3] = R
    init_pose[:3, 3] = [0.0, 0.0, 0.0]
    
    sample_robot = RobotPcSampler(
        urdf_path, link_names=["left_finger", "right_finger"], init_pose=init_pose
    )
    
    # Create trainer with specified device
    trainer = InvPhyTrainerWarp(
        data_path=f"{base_path}/{case_name}/final_data.pkl",
        base_dir=base_dir,
        pure_inference_mode=True,
        static_meshes=[],
        robot=sample_robot,
        include_gaussian=include_gaussian,
        device=device  # Force this specific GPU
    )
    
    # Get paths
    best_model_path = glob.glob(str(case_paths['model_dir'] / "best_*.pth"))[0]
    exp_name = "init=hybrid_iso=True_ldepth=0.001_lnormal=0.0_laniso_0.0_lseg=1.0"
    gaussians_path = f"{GAUSSIAN_OUTPUT_DIR}/{case_name}/{exp_name}/point_cloud/iteration_10000/point_cloud.ply"
    
    # Check if episode already exists and handle overwrite
    if not overwrite:
        # Check if episode already exists
        try:
            with h5py.File(data_file_path, 'r') as f:
                if f'episode_{episode_id:06d}' in f:
                    print(f"Episode {episode_id} already exists, skipping (use --overwrite to replace)")
                    return f"Episode {episode_id} skipped (already exists)"
        except (FileNotFoundError, OSError):
            pass  # File doesn't exist yet, continue
    
    # Generate the episode
    try:
        trainer.generate_data(
            best_model_path, 
            gaussians_path, 
            n_ctrl_parts, 
            data_file_path,
            episode_id,
            overwrite=overwrite
        )
        print(f"Episode {episode_id} completed on GPU {gpu_id}")
        return f"Episode {episode_id} completed successfully"
    except Exception as e:
        print(f"Episode {episode_id} failed on GPU {gpu_id}: {e}")
        return f"Episode {episode_id} failed: {e}"

def generate_data_parallel():
    """
    Main function for parallel data generation across multiple GPUs.
    """
    from argparse import ArgumentParser
    
    parser = ArgumentParser()
    parser.add_argument("--base_path", type=str, default=str(DATA_DIFFERENT_TYPES))
    parser.add_argument("--data_file", type=str, default=str(GENERATED_DATA_DIR / "data.h5"))
    parser.add_argument("--case_name", type=str, default="double_lift_cloth_3")
    parser.add_argument("--n_ctrl_parts", type=int, default=1)
    parser.add_argument("--episodes", nargs='+', type=str, required=True,
                       help="Episodes to generate. Format: space-separated list (0 1 2 3 4) or range (0-4)")
    parser.add_argument("--include_gaussian", action="store_true")
    parser.add_argument("--max_workers", type=int, default=None,
                       help="Maximum number of parallel workers (default: number of GPUs)")
    parser.add_argument("--overwrite", action="store_true",
                       help="Overwrite existing episode data if it already exists")
    args = parser.parse_args()
    
    # Parse episodes
    try:
        episode_list = parse_episodes(args.episodes)
    except ValueError as e:
        print(f"Error parsing episodes: {e}")
        sys.exit(1)
    
    # Check available GPUs
    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        print("No GPUs available!")
        sys.exit(1)
    
    print(f"Found {num_gpus} GPUs available")
    
    # Set max workers
    max_workers = args.max_workers if args.max_workers else num_gpus
    max_workers = min(max_workers, num_gpus, len(episode_list))
    
    print(f"Using {max_workers} parallel workers for {len(episode_list)} episodes")
    
    # Initialize data file
    if not os.path.exists(args.data_file):
        os.makedirs(os.path.dirname(args.data_file), exist_ok=True)
        with h5py.File(args.data_file, 'w') as f:
            f.attrs['created'] = datetime.now().isoformat()
            f.attrs['description'] = 'PhysTwin parallel episode data collection'
        print(f"Initialized data file: {args.data_file}")
    
    # Prepare arguments for parallel execution
    args_list = []
    for i, episode_id in enumerate(episode_list):
        gpu_id = i % num_gpus  # Distribute episodes across GPUs
        args_tuple = (
            episode_id, 
            gpu_id, 
            args.case_name, 
            args.data_file, 
            args.n_ctrl_parts, 
            args.include_gaussian,
            args.base_path,
            args.overwrite
        )
        args_list.append(args_tuple)
    
    print(f"Episode distribution across GPUs:")
    for gpu_id in range(num_gpus):
        episodes_on_gpu = [args_list[i][0] for i in range(len(args_list)) if args_list[i][1] == gpu_id]
        print(f"  GPU {gpu_id}: Episodes {episodes_on_gpu}")
    
    # Run episodes in parallel
    print("Starting parallel episode generation...")
    
    # Use spawn method to avoid CUDA context issues
    mp.set_start_method('spawn', force=True)
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(generate_episode_on_gpu, args_list))
    
    print("\nParallel generation completed!")
    print("Results:")
    for result in results:
        print(f"  {result}")

if __name__ == "__main__":
    generate_data_parallel() 