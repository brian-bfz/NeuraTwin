"""
Stateful robot pose and movement controller.
Refactored from RobotMovementController to be the single source of robot state truth.
"""
import torch
import numpy as np
import open3d as o3d


def axis_angle_to_matrix(axis_angle: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as axis/angle to rotation matrices.
    
    Args:
        axis_angle: Rotations given as a vector in axis angle form,
            as a tensor of shape (..., 3), where the magnitude is
            the angle turned anticlockwise in radians around the
            vector's direction.
    
    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    shape = axis_angle.shape
    device, dtype = axis_angle.device, axis_angle.dtype

    angles = torch.norm(axis_angle, p=2, dim=-1, keepdim=True).unsqueeze(-1)

    rx, ry, rz = axis_angle[..., 0], axis_angle[..., 1], axis_angle[..., 2]
    zeros = torch.zeros(shape[:-1], dtype=dtype, device=device)
    cross_product_matrix = torch.stack(
        [zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros], dim=-1
    ).view(shape + (3,))
    cross_product_matrix_sqrd = cross_product_matrix @ cross_product_matrix

    identity = torch.eye(3, dtype=dtype, device=device)
    angles_sqrd = angles * angles
    angles_sqrd = torch.where(angles_sqrd == 0, 1, angles_sqrd)
    return (
        identity.expand(cross_product_matrix.shape)
        + torch.sinc(angles / torch.pi) * cross_product_matrix
        + ((1 - torch.cos(angles)) / angles_sqrd) * cross_product_matrix_sqrd
    )


class RobotController:
    """
    Handles robot state management and movement control.
    
    This class is the single source of truth for robot state including:
    - Position and rotation
    - Gripper opening
    - Mesh vertices
    
    All internal computations use PyTorch tensors for efficiency.
    Only converts to NumPy at interface boundaries (Open3D, file I/O).
    """
    
    def __init__(self, robot_loader, n_ctrl_parts=1, device='cuda'):
        """
        Initialize the robot controller.
        
        Args:
            robot_loader: RobotLoader instance for getting finger meshes
            n_ctrl_parts: Number of control parts (default: 1)
            device: PyTorch device
        """
        self.n_ctrl_parts = n_ctrl_parts
        self.device = device
        self.robot_loader = robot_loader
    
        self.accumulate_trans = torch.zeros((n_ctrl_parts, 3), dtype=torch.float32, device=device)
        self.origin_force_judge = torch.tensor(
            [[-1, 0, 0], [1, 0, 0]], dtype=torch.float32, device=device
        )
        
        # Initialize to default state
        self._reset()
               
    def _reset(self):
        """Reset all state variables to initial values."""
        self.accumulate_trans.zero_()  # Reset to zero without changing type
        self.accumulate_rot = torch.eye(3, dtype=torch.float32, device=self.device)
        self.current_finger = 1.0
        self.is_closing = True
        self.current_force_judge = self.origin_force_judge.clone()

        # Update finger meshes and vertices for current finger position
        self.finger_meshes = self.robot_loader.get_finger_mesh(self.current_finger)
        finger_vertices = [np.asarray(mesh.vertices) for mesh in self.finger_meshes]
        base_finger_vertices_np = np.concatenate(finger_vertices, axis=0)
        self.base_finger_vertices = torch.tensor(base_finger_vertices_np, dtype=torch.float32, device=self.device)
        self.initial_gripper_center = torch.mean(self.base_finger_vertices, dim=0)
        self.relative_finger_vertices = self.base_finger_vertices - self.initial_gripper_center # base_finger_vertices and intiial_gripper_center are never changed

        # Set initial mesh vertices (already a tensor)
        self.current_trans_dynamic_points = self.base_finger_vertices.clone()
        
    def quick_robot_movement(self, target_change, current_finger=0.0, rot_change=None):
        """
        Quickly teleport robot to desired pose (for initialization).
        
        Args:
            target_change: torch.Tensor [n_ctrl_parts, 3] - translation change
            current_finger: float - target finger opening value [0.0, 1.0]
            rot_change: torch.Tensor [3] - rotation change as axis-angle (default: None)
            
        Returns:
            torch.Tensor: current_trans_dynamic_points [n_vertices, 3]
        """
        # Handle default rotation
        if rot_change is None:
            rot_change = torch.zeros(3, dtype=torch.float32, device=self.device)
            
        # Update rotation
        if torch.norm(rot_change) > 0:
            rot_mat = axis_angle_to_matrix(rot_change.unsqueeze(0))[0]
            self.accumulate_rot = torch.matmul(self.accumulate_rot, rot_mat)
        
        # Update internal states
        self.accumulate_trans += target_change
        self.current_finger = max(0.0, min(1.0, current_finger))
        self.is_closing = True if self.current_finger == 0.0 else False
        
        # Get updated finger meshes and vertices
        self.finger_meshes = self.robot_loader.get_finger_mesh(self.current_finger)
        finger_vertices = [np.asarray(mesh.vertices) for mesh in self.finger_meshes]
        base_finger_vertices_np = np.concatenate(finger_vertices, axis=0)
        self.relative_finger_vertices = torch.tensor(base_finger_vertices_np, dtype=torch.float32, device=self.device) - self.initial_gripper_center
        
        # Apply transformations about gripper center
        current_gripper_center = self.initial_gripper_center + self.accumulate_trans[0]
        
        # Apply rotation about gripper center
        rotated_vertices = self.relative_finger_vertices @ self.accumulate_rot.T
        
        # Translate to final position
        self.current_trans_dynamic_points = rotated_vertices + current_gripper_center
        
        # Update force judge direction
        self.current_force_judge = self.origin_force_judge.clone() @ self.accumulate_rot

        return self.current_trans_dynamic_points
        
    def fine_robot_movement(self, target_change, close_flag=True, finger_change=0.0, rot_change=None):
        """
        Smoothly move robot with interpolated motion (for simulation).
        
        Args:
            target_change: torch.Tensor [n_ctrl_parts, 3] - translation change for this step
            finger_change: float - change in finger opening (default: 0.0)
            rot_change: torch.Tensor [3] - rotation change as axis-angle (default: None)
            
        Returns:
            dict containing:
                interpolated_dynamic_points: [num_substeps, n_robot_vertices, 3] - robot mesh points
                interpolated_center: [num_substeps, 3] - center points for each substep
                dynamic_velocity: [3] - robot velocity
                dynamic_omega: [3] - robot angular velocity
                current_trans_dynamic_points: [n_robot_vertices, 3] - final robot mesh points
        """
        # Store previous state
        prev_accumulate_rot = self.accumulate_rot.clone()
        prev_trans_dynamic_points = self.current_trans_dynamic_points.clone()

        # Import cfg here to avoid circular imports
        from ..qqtt.utils import cfg
        
        # Handle default rotation
        if rot_change is None:
            rot_change = torch.zeros(3, dtype=torch.float32, device=self.device)
            
        # Update is_closing flag based on finger_change
        if finger_change > 0:
            self.is_closing = False
        elif finger_change < 0:
            self.is_closing = True

        if self.is_closing:
            if not close_flag:
                finger_change = 0.0
            else: 
                finger_change = -0.05
        else:
            finger_change = 0.05
            
        # Update finger position
        self.current_finger += finger_change
        self.current_finger = max(0.0, min(1.0, self.current_finger))
        
        # Get updated finger meshes for the new finger position
        if self.current_finger != 0.0:
            self.finger_meshes = self.robot_loader.get_finger_mesh(self.current_finger)
            finger_vertices = [np.asarray(mesh.vertices) for mesh in self.finger_meshes]
            base_finger_vertices_np = np.concatenate(finger_vertices, axis=0)
            self.relative_finger_vertices = torch.tensor(base_finger_vertices_np, dtype=torch.float32, device=self.device) - self.initial_gripper_center
        
        # Apply shared movement logic (translation update)
        self.accumulate_trans += target_change
        
        # Calculate current mesh vertices after translation
        current_trans_dynamic_points = self.relative_finger_vertices + self.initial_gripper_center + self.accumulate_trans[0]
        
        # Calculate interpolated points considering finger and translation
        ratios = (
            torch.linspace(1, cfg.num_substeps, cfg.num_substeps, device=self.device).view(-1, 1, 1)
            / cfg.num_substeps
        )
        
        # Interpolate from previous to current mesh positions
        interpolated_trans_dynamic_points = (
            prev_trans_dynamic_points.unsqueeze(0)
            + (current_trans_dynamic_points - prev_trans_dynamic_points).unsqueeze(0)
            * ratios
        )
        interpolated_center = torch.mean(interpolated_trans_dynamic_points, dim=1)
        
        # Do the rotation on the interpolated points
        interpolated_rot_angle = rot_change.unsqueeze(0) * ratios.reshape(-1, 1)
        interpolated_rot_temp = axis_angle_to_matrix(interpolated_rot_angle)
        interpolated_rot_mat = prev_accumulate_rot.unsqueeze(0) @ interpolated_rot_temp
        self.accumulate_rot = interpolated_rot_mat[-1]
        
        # Apply progressive rotation about interpolated centers
        interpolated_dynamic_points = (
            interpolated_trans_dynamic_points - interpolated_center.unsqueeze(1)
        ) @ interpolated_rot_mat.permute(0, 2, 1) + interpolated_center.unsqueeze(1)
        
        # Update final mesh vertices
        self.current_trans_dynamic_points = interpolated_dynamic_points[-1]
        
        # Update force judge direction
        self.current_force_judge = self.origin_force_judge.clone() @ interpolated_rot_mat[-1]
        
        # Calculate velocity and omega
        dynamic_velocity = target_change[0] / (2 * cfg.dt * cfg.num_substeps)
        dynamic_omega = rot_change / (2 * cfg.dt * cfg.num_substeps)
        
        return {
            'interpolated_dynamic_points': interpolated_dynamic_points,
            'interpolated_center': interpolated_center,
            'dynamic_velocity': dynamic_velocity,
            'dynamic_omega': dynamic_omega,
            'current_trans_dynamic_points': self.current_trans_dynamic_points
        }
        
    def set_to_match_vertices(self, target_vertices):
        """
        Set robot state to match given mesh vertices by calculating the required translation.
        
        Args:
            target_vertices: torch.Tensor [n_vertices, 3] - target mesh vertices
        """
        current_mesh_center = self.initial_gripper_center + self.accumulate_trans[0]
        target_mesh_center = torch.mean(target_vertices, dim=0)
        
        # Calculate translation from current to target
        translation = (target_mesh_center - current_mesh_center).unsqueeze(0)  # Shape: [1, 3] for n_ctrl_parts=1
        
        self.quick_robot_movement(
            target_change=translation,
            current_finger=0.0,
            rot_change=None
        )
                
    def get_current_state(self):
        """Get current robot state for serialization/debugging."""
        return {
            'accumulate_trans': self.accumulate_trans.clone(),
            'accumulate_rot': self.accumulate_rot.clone(),
            'current_finger': self.current_finger,
            'is_closing': self.is_closing,
            'current_trans_dynamic_points': self.current_trans_dynamic_points.clone(),
            'current_force_judge': self.current_force_judge.clone()
        }
        
    def get_current_force_judge(self):
        """Get current force judge for collision detection."""
        return self.current_force_judge
        
    def get_current_finger(self):
        """Get current finger opening value."""
        return self.current_finger
        
    def get_current_center(self):
        """Get current center of the robot."""
        return self.initial_gripper_center + self.accumulate_trans[0]
