from .qqtt import InvPhyTrainerWarp
from .qqtt.utils import logger, cfg
from .config_manager import PhysTwinConfig, create_common_parser
import random
import numpy as np
import torch
import open3d as o3d
import sapien.core as sapien
from urdfpy import URDF
from .paths import *

# Import GNN modules
from GNN.model.gnn_dyn import PropNetDiffDenModel
from GNN.utils import load_yaml
from GNN.paths import get_model_paths


def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed = 42
set_all_seeds(seed)


def trimesh_to_open3d(trimesh_mesh):
    o3d_mesh = o3d.geometry.TriangleMesh()
    o3d_mesh.vertices = o3d.utility.Vector3dVector(trimesh_mesh.vertices)
    o3d_mesh.triangles = o3d.utility.Vector3iVector(trimesh_mesh.faces)
    o3d_mesh.paint_uniform_color([1, 0, 0])
    o3d_mesh.compute_vertex_normals()
    return o3d_mesh


# Remove duplicate RobotPcSampler class - use the one from robot module


if __name__ == "__main__":
    # Create parser with common arguments
    parser = create_common_parser()
    
    # Add script-specific arguments
    parser.add_argument("--n_ctrl_parts", type=int, default=1)
    parser.add_argument(
        "--inv_ctrl", action="store_true", help="invert horizontal control direction"
    )
    parser.add_argument(
        "--gnn_model", type=str, default=None,
        help="GNN model name for comparison visualization"
    )
    parser.add_argument(
        "--gnn_config", type=str, default=None,
        help="GNN config path (if not provided, uses model's config)"
    )
    args = parser.parse_args()

    # Initialize configuration - this replaces ~80 lines of setup code
    config = PhysTwinConfig(
        case_name=args.case_name,
        base_path=args.base_path,
        bg_img_path=args.bg_img_path,
        gaussian_path=args.gaussian_path
    )


    # Load the static_meshes (keeping existing static mesh setup)
    static_meshes = []

    # Load GNN model and config
    gnn_model = None
    gnn_config = None
    if args.gnn_model:
        try:
            logger.info(f"Loading GNN model: {args.gnn_model}")
            model_paths = get_model_paths(args.gnn_model)
            
            # Load config
            if args.gnn_config:
                config_path = args.gnn_config
            else:
                config_path = str(model_paths['config'])
            
            gnn_config = load_yaml(config_path)
            logger.info(f"Loaded GNN config from: {config_path}")
            
            # Load model
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            gnn_model = PropNetDiffDenModel(gnn_config, torch.cuda.is_available())
            model_checkpoint = torch.load(str(model_paths['net_best']), map_location=device)
            gnn_model.load_state_dict(model_checkpoint)
            gnn_model.to(device)
            gnn_model.eval()
            
            logger.info(f"Loaded GNN model from: {model_paths['net_best']}")
            logger.info(f"GNN model ready for comparison visualization")
            
        except Exception as e:
            logger.warning(f"Failed to load GNN model: {e}")
            logger.warning("Continuing without GNN comparison")
            gnn_model = None
            gnn_config = None

    # Create trainer
    trainer = InvPhyTrainerWarp(
        data_path=config.get_data_path(),
        base_dir=config.get_temp_base_dir(),
        pure_inference_mode=True,
        static_meshes=static_meshes,
        robot_loader=config.create_robot_loader(),
        robot_initial_pose=config.get_robot_initial_pose("interactive"),
    )

    # Run interactive session
    best_model_path = config.get_best_model_path()
    gaussians_path = config.get_gaussian_path()
    
    trainer.interactive_robot(
        best_model_path,
        gaussians_path,
        args.n_ctrl_parts,
        args.inv_ctrl,
        gnn_model=gnn_model,
        gnn_config=gnn_config,
    )
