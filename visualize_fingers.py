import os
import sapien.core as sapien
from pathlib import Path
from urdfpy import URDF
import numpy as np
import open3d as o3d


def trimesh_to_open3d(trimesh_mesh):
    o3d_mesh = o3d.geometry.TriangleMesh()
    o3d_mesh.vertices = o3d.utility.Vector3dVector(trimesh_mesh.vertices)
    o3d_mesh.triangles = o3d.utility.Vector3iVector(trimesh_mesh.faces)
    return o3d_mesh


class RobotPcSampler:
    def __init__(self, urdf_path, link_names):
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
                        self.scales[link.name] = collision.geometry.mesh.scale[0] if collision.geometry.mesh.scale is not None else 1.0
                self.offsets[link.name] = prev_offset

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
        link_pose_ls = np.stack([np.asarray(self.robot_model.get_link_pose(link_idx).to_transformation_matrix()) for link_idx in link_idx_ls])
        meshes_ls = [self.meshes[link_name] for link_name in link_names]
        offsets_ls = [self.offsets[link_name] for link_name in link_names]
        scales_ls = [self.scales[link_name] for link_name in link_names]
        poses = self.get_mesh_poses(poses=link_pose_ls, offsets=offsets_ls, scales=scales_ls)
        return poses
    
    def get_mesh_poses(self, poses, offsets, scales):
        try:
            assert poses.shape[0] == len(offsets)
        except:
            raise RuntimeError('poses and meshes must have the same length')

        N = poses.shape[0]
        all_mats = []
        for index in range(N):
            mat = poses[index]
            tf_obj_to_link = offsets[index]
            mat = mat @ tf_obj_to_link
            all_mats.append(mat)
        return np.stack(all_mats)


def visualize_mesh(gripper_openness=1.0):

    urdf_path = "xarm/xarm7_with_gripper.urdf"

    sample_robot = RobotPcSampler(urdf_path, link_names=['left_finger', 'right_finger'])

    g = 800 * gripper_openness  # gripper openness
    g = (800 - g) * 180 / np.pi
    base_qpos = np.array([0, -45, 0, 30, 0, 75, 0, 
                          g*0.001, g*0.001, g*0.001, g*0.001, g*0.001, g*0.001]) * np.pi / 180
    
    link_names = list(sample_robot.meshes.keys())
    meshes = [sample_robot.meshes[link_name] for link_name in link_names]
    poses = sample_robot.compute_mesh_poses(base_qpos, link_names=link_names)

    for i, mesh in enumerate(meshes):
        vertices = np.asarray(mesh.vertices)
        vertices = vertices @ poses[i][:3, :3].T + poses[i][:3, 3]
        import pdb
        pdb.set_trace()
        mesh.vertices = o3d.utility.Vector3dVector(vertices)
    
    coordinate = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
    o3d.visualization.draw_geometries(meshes + [coordinate])  # type: ignore

if __name__ == '__main__':
    visualize_mesh(1.0)
