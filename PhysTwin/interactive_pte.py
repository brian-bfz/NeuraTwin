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


class RobotPcSampler:
    def __init__(self, urdf_path, link_names, init_pose):
        self.engine = sapien.Engine()
        self.scene = self.engine.create_scene()
        loader = self.scene.create_urdf_loader()
        self.sapien_robot = loader.load(urdf_path)
        self.robot_model = self.sapien_robot.create_pinocchio_model()
        self.urdf_robot = URDF.load(urdf_path)

        # load meshes and offsets from urdf_robot
        self.meshes = {}
        self.scales = {}
        self.offsets = {}
        prev_offset = np.eye(4)
        for link in self.urdf_robot.links:
            if link.name not in link_names:
                continue
            if len(link.collisions) > 0:
                collision = link.collisions[0]
                prev_offset = collision.origin
                if collision.geometry.mesh != None:
                    if len(collision.geometry.mesh.meshes) > 0:
                        mesh = collision.geometry.mesh.meshes[0]
                        self.meshes[link.name] = trimesh_to_open3d(mesh)
                        self.scales[link.name] = (
                            collision.geometry.mesh.scale[0]
                            if collision.geometry.mesh.scale is not None
                            else 1.0
                        )
                self.offsets[link.name] = prev_offset

        self.finger_link_names = list(self.meshes.keys())
        self.finger_meshes = [self.meshes[link_name] for link_name in link_names]
        self.finger_vertices = [
            np.copy(np.asarray(mesh.vertices)) for mesh in self.finger_meshes
        ]
        self.init_pose = init_pose

    def compute_mesh_poses(self, qpos, link_names=None):
        fk = self.robot_model.compute_forward_kinematics(qpos)
        if link_names is None:
            link_names = self.meshes.keys()
        link_idx_ls = []
        for link_name in link_names:
            for link_idx, link in enumerate(self.sapien_robot.get_links()):
                if link.name == link_name:
                    link_idx_ls.append(link_idx)
                    break
        link_pose_ls = np.stack(
            [
                np.asarray(
                    self.robot_model.get_link_pose(link_idx).to_transformation_matrix()
                )
                for link_idx in link_idx_ls
            ]
        )
        meshes_ls = [self.meshes[link_name] for link_name in link_names]
        offsets_ls = [self.offsets[link_name] for link_name in link_names]
        scales_ls = [self.scales[link_name] for link_name in link_names]
        poses = self.get_mesh_poses(
            poses=link_pose_ls, offsets=offsets_ls, scales=scales_ls
        )
        return poses

    def get_mesh_poses(self, poses, offsets, scales):
        try:
            assert poses.shape[0] == len(offsets)
        except:
            raise RuntimeError("poses and meshes must have the same length")

        N = poses.shape[0]
        all_mats = []
        for index in range(N):
            mat = poses[index]
            tf_obj_to_link = offsets[index]
            mat = mat @ tf_obj_to_link
            all_mats.append(mat)
        return np.stack(all_mats)

    def get_finger_mesh(self, gripper_openness=1.0):
        g = 800 * gripper_openness  # gripper openness
        g = (800 - g) * 180 / np.pi
        base_qpos = (
            np.array(
                [
                    0,
                    -45,
                    0,
                    30,
                    0,
                    75,
                    0,
                    g * 0.001,
                    g * 0.001,
                    g * 0.001,
                    g * 0.001,
                    g * 0.001,
                    g * 0.001,
                ]
            )
            * np.pi
            / 180
        )

        poses = sample_robot.compute_mesh_poses(
            base_qpos, link_names=self.finger_link_names
        )
        for i, origin_vertice in enumerate(self.finger_vertices):
            vertices = np.copy(origin_vertice)
            vertices = vertices @ poses[i][:3, :3].T + poses[i][:3, 3]
            vertices = vertices @ self.init_pose[:3, :3].T + self.init_pose[:3, 3]
            self.finger_meshes[i].vertices = o3d.utility.Vector3dVector(vertices)

        return self.finger_meshes


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

    # Create robot with interactive pose
    sample_robot = config.create_robot("interactive")

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
        robot=sample_robot,
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
