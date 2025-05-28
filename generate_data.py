# import sys
# sys.path.append("./gaussian_splatting")
from qqtt import InvPhyTrainerWarp
from qqtt.utils import logger, cfg
from datetime import datetime
import random
import numpy as np
import torch
from argparse import ArgumentParser
import glob
import os
import pickle
from SampleRobot import RobotPcSampler

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

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--base_path",
        type=str,
        default="./data/different_types",
    )
    parser.add_argument(
        "--gaussian_path",
        type=str,
        default="./gaussian_output",
    )
    parser.add_argument("--case_name", type=str, default="double_lift_cloth_3")
    parser.add_argument("--n_ctrl_parts", type=int, default=1)
    parser.add_argument("--custom_ctrl_points", type=str, help="Path to directory containing custom control points")
    args = parser.parse_args()

    base_path = args.base_path
    case_name = args.case_name

    if "cloth" in case_name or "package" in case_name:
        cfg.load_from_yaml("configs/cloth.yaml")
    else:
        cfg.load_from_yaml("configs/real.yaml")

    base_dir = f"./temp_experiments/{case_name}"

    # Read the first-satage optimized parameters to set the indifferentiable parameters
    optimal_path = f"./experiments_optimization/{case_name}/optimal_params.pkl"
    logger.info(f"Load optimal parameters from: {optimal_path}")
    assert os.path.exists(
        optimal_path
    ), f"{case_name}: Optimal parameters not found: {optimal_path}"
    with open(optimal_path, "rb") as f:
        optimal_params = pickle.load(f)
    cfg.set_optimal_params(optimal_params)

    # Load the robot finger
    urdf_path = "xarm/xarm7_with_gripper.urdf"
    R = np.array([[0.0, -1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, -1.0]])

    init_pose = np.eye(4)
    init_pose[:3, :3] = R
    init_pose[:3, 3] = [0.0, 0.0, 0.0]

    exp_name = "init=hybrid_iso=True_ldepth=0.001_lnormal=0.0_laniso_0.0_lseg=1.0"
    gaussians_path = f"{args.gaussian_path}/{case_name}/{exp_name}/point_cloud/iteration_10000/point_cloud.ply"

    logger.set_log_file(path=base_dir, name="inference_log")

    best_model_path = glob.glob(f"experiments/{case_name}/train/best_*.pth")[0]
    
    # Create timestamped folder for this simulation run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # print(case_name, timestamp, f"{case_name}_{timestamp}")

    for i in range(2):
        sample_robot = RobotPcSampler(
            urdf_path, link_names=["left_finger", "right_finger"], init_pose=init_pose
        )
        trainer = InvPhyTrainerWarp(
            data_path=f"{base_path}/{case_name}/final_data.pkl",
            base_dir=base_dir,
            pure_inference_mode=True,
            static_meshes=[],
            robot=sample_robot,
        )
        save_dir = os.path.join("generated_data", f"{i}")
        os.makedirs(os.path.join(save_dir, "object"), exist_ok=True)
        os.makedirs(os.path.join(save_dir, "gaussians"), exist_ok=True)
        os.makedirs(os.path.join(save_dir, "robot"), exist_ok=True)
        trainer.generate_data(
            best_model_path, 
            gaussians_path, 
            args.n_ctrl_parts, 
            save_dir, 
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

    
