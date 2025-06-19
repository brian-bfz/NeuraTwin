"""
Stateless robot URDF loading and mesh generation.
Refactored from SampleRobot.py to remove mutable state.
"""
import open3d as o3d
import sapien.core as sapien
from urdfpy import URDF
import numpy as np


def trimesh_to_open3d(trimesh_mesh):
    """Convert trimesh mesh to open3d mesh."""
    o3d_mesh = o3d.geometry.TriangleMesh()
    o3d_mesh.vertices = o3d.utility.Vector3dVector(trimesh_mesh.vertices)
    o3d_mesh.triangles = o3d.utility.Vector3iVector(trimesh_mesh.faces)
    o3d_mesh.paint_uniform_color([1, 0, 0])
    o3d_mesh.compute_vertex_normals()
    return o3d_mesh


class RobotLoader:
    """
    Stateless robot loader for URDF parsing and mesh generation.
    
    This class handles:
    - URDF loading and parsing
    - Forward kinematics (joint angles → mesh vertices)
    - Mesh generation for given gripper opening
    
    It is stateless - all methods take pose/transform as parameters.
    """
    
    def __init__(self, urdf_path, link_names, transform=None):
        """
        Initialize robot loader.
        
        Args:
            urdf_path: Path to URDF file
            link_names: List of link names to load meshes for
            transform: np.ndarray [4,4] - transformation matrix to apply (optional)
        """
        self.urdf_path = urdf_path
        self.link_names = link_names
        self.transform = transform
        
        # Initialize SAPIEN and URDF
        self.engine = sapien.Engine()
        self.scene = self.engine.create_scene()
        loader = self.scene.create_urdf_loader()
        self.sapien_robot = loader.load(urdf_path)
        self.robot_model = self.sapien_robot.create_pinocchio_model()
        self.urdf_robot = URDF.load(urdf_path)

        # Load meshes and offsets from URDF
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

    def compute_mesh_poses(self, qpos, link_names=None):
        """Compute mesh poses for given joint positions."""
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
        """Apply offsets to link poses."""
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

    def get_finger_mesh(self, gripper_openness=0.0):
        """
        Get finger meshes for given gripper opening.
        
        Args:
            gripper_openness: float [0,1] - gripper opening amount
            
        Returns:
            List of finger meshes
        """
        # Calculate joint positions from gripper openness
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

        # Compute mesh poses from joint positions
        poses = self.compute_mesh_poses(base_qpos, link_names=self.finger_link_names)
        
        # Create copies of meshes and apply transforms
        result_meshes = []
        for i, origin_vertices in enumerate(self.finger_vertices):
            vertices = np.copy(origin_vertices)
            
            # Apply robot kinematics
            vertices = vertices @ poses[i][:3, :3].T + poses[i][:3, 3]
            
            # Apply additional transform if provided
            if self.transform is not None:
                vertices = vertices @ self.transform[:3, :3].T + self.transform[:3, 3]
            
            # Update mesh vertices
            mesh_copy = o3d.geometry.TriangleMesh(self.finger_meshes[i])
            mesh_copy.vertices = o3d.utility.Vector3dVector(vertices)
            result_meshes.append(mesh_copy)

        return result_meshes

    def get_finger_vertices(self, gripper_openness=0.0):
        """
        Get finger vertices as numpy array.
        
        Args:
            gripper_openness: float [0,1] - gripper opening amount
            transform: np.ndarray [4,4] - transformation matrix to apply (optional)
            
        Returns:
            np.ndarray [n_vertices, 3] - concatenated finger vertices
        """
        finger_meshes = self.get_finger_mesh(gripper_openness)
        vertices_list = [np.asarray(mesh.vertices) for mesh in finger_meshes]
        return np.concatenate(vertices_list, axis=0) 