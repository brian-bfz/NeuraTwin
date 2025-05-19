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
import json
import open3d as o3d
import sapien.core as sapien
from urdfpy import URDF

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

def random_movement(n_ctrl_parts, num_movements=10, frames_per_movement=10):
    # Define possible keys for each hand
    hand1_keys = ['w', 's', 'a', 'd', 'e', 'q']  # Left hand keys
    hand2_keys = ['i', 'k', 'j', 'l', 'o', 'u']  # Right hand keys
    
    sequence = []

    # temporary code to set position
    movement = []
    movement.append("m")
    movement.append("d")
    sequence.extend([movement] * 2 * frames_per_movement)

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
    init_pose[:3, 3] = [0.2, 0.0, 0.23]
    sample_robot = RobotPcSampler(
        urdf_path, link_names=["left_finger", "right_finger"], init_pose=init_pose
    )

    exp_name = "init=hybrid_iso=True_ldepth=0.001_lnormal=0.0_laniso_0.0_lseg=1.0"
    gaussians_path = f"{args.gaussian_path}/{case_name}/{exp_name}/point_cloud/iteration_10000/point_cloud.ply"

    logger.set_log_file(path=base_dir, name="inference_log")
    trainer = InvPhyTrainerWarp(
        data_path=f"{base_path}/{case_name}/final_data.pkl",
        base_dir=base_dir,
        pure_inference_mode=True,
        static_meshes=[],
        robot=sample_robot,
    )

    best_model_path = glob.glob(f"experiments/{case_name}/train/best_*.pth")[0]
    
    # Create timestamped folder for this simulation run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # print(case_name, timestamp, f"{case_name}_{timestamp}")
    save_dir = os.path.join("generated_data", f"{case_name}_{timestamp}")
    os.makedirs(os.path.join(save_dir, "x"), exist_ok=True)
    os.makedirs(os.path.join(save_dir, "gaussians"), exist_ok=True)
    os.makedirs(os.path.join(save_dir, "controller_points"), exist_ok=True)

    # Load custom control points if provided
    # custom_control_points = None
    # if args.custom_ctrl_points:
    #     cp_path = os.path.join(args.custom_ctrl_points, "cp.pt")
    #     if not os.path.exists(cp_path):
    #         raise FileNotFoundError(f"Control points file not found: {cp_path}")
    #     custom_control_points = torch.load(cp_path)
    #     logger.info(f"Loaded custom control points from {cp_path}")

    pressed_keys_sequence = random_movement(args.n_ctrl_parts)
    print(pressed_keys_sequence)
    
    trainer.generate_data(
        best_model_path, 
        gaussians_path, 
        args.n_ctrl_parts, 
        save_dir, 
        pressed_keys_sequence, 
    )
