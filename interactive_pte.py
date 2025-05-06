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
    parser.add_argument(
        "--bg_img_path",
        type=str,
        default="./data/bg.png",
    )
    parser.add_argument("--case_name", type=str, default="double_lift_cloth_3")
    parser.add_argument("--n_ctrl_parts", type=int, default=2)
    parser.add_argument(
        "--inv_ctrl", action="store_true", help="invert horizontal control direction"
    )
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

    # Set the intrinsic and extrinsic parameters for visualization
    with open(f"{base_path}/{case_name}/calibrate.pkl", "rb") as f:
        c2ws = pickle.load(f)
    w2cs = [np.linalg.inv(c2w) for c2w in c2ws]
    cfg.c2ws = np.array(c2ws)
    cfg.w2cs = np.array(w2cs)
    with open(f"{base_path}/{case_name}/metadata.json", "r") as f:
        data = json.load(f)
    cfg.intrinsics = np.array(data["intrinsics"])
    cfg.WH = data["WH"]
    cfg.bg_img_path = args.bg_img_path

    # Load the static_meshes
    static_meshes = []
    clip_mesh = o3d.io.read_triangle_mesh("clip.stl")
    clip_mesh.scale(0.0015, center=[0, 0, 0])
    static_poses = np.array(
        [
            [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, -0.005], [0, 0, 0, 1]],
            [[1, 0, 0, 0.1], [0, -1, 0, 0.3], [0, 0, -1, -0.005], [0, 0, 0, 1]],
        ],
        dtype=np.float32,
    )
    # static_poses = np.array(
    #     [
    #         [[0, -1, 0, 0], [-1, 0, 0, 0], [0, 0, -1, -0.005], [0, 0, 0, 1]],
    #         [[0, -1, 0, 0.2], [-1, 0, 0, 0.2], [0, 0, -1, -0.005], [0, 0, 0, 1]],
    #     ],
    #     dtype=np.float32,
    # )
    for static_pose in static_poses:
        new_mesh = o3d.geometry.TriangleMesh(clip_mesh)
        new_mesh.transform(static_pose)
        new_mesh.paint_uniform_color([0.929, 0, 1])
        new_mesh.compute_vertex_normals()
        static_meshes.append(new_mesh)
    # coordinate = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
    # o3d.visualization.draw_geometries(static_meshes + [coordinate])

    # Load the robot finger
    urdf_path = "xarm/xarm7_with_gripper.urdf"
    R = np.array([[0.0, -1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, -1.0]])

    init_pose = np.eye(4)
    init_pose[:3, :3] = R
    init_pose[:3, 3] = [0.2, 0.0, 0.23]
    sample_robot = RobotPcSampler(
        urdf_path, link_names=["left_finger", "right_finger"], init_pose=init_pose
    )
    # meshes = sample_robot.get_finger_mesh(gripper_openness=1.0)
    # coordinate = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
    # o3d.visualization.draw_geometries(meshes + [coordinate])

    exp_name = "init=hybrid_iso=True_ldepth=0.001_lnormal=0.0_laniso_0.0_lseg=1.0"
    gaussians_path = f"{args.gaussian_path}/{case_name}/{exp_name}/point_cloud/iteration_10000/point_cloud.ply"

    logger.set_log_file(path=base_dir, name="inference_log")
    trainer = InvPhyTrainerWarp(
        data_path=f"{base_path}/{case_name}/final_data.pkl",
        base_dir=base_dir,
        pure_inference_mode=True,
        static_meshes=static_meshes,
        robot=sample_robot,
    )

    best_model_path = glob.glob(f"experiments/{case_name}/train/best_*.pth")[0]
    trainer.interactive_robot(
        best_model_path,
        gaussians_path,
        args.n_ctrl_parts,
        args.inv_ctrl,
    )
