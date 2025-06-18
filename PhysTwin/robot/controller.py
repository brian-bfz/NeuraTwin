"""
Stateful robot pose and movement controller.
Refactored from RobotMovementController to be the single source of robot state truth.
"""
import torch
import numpy as np


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
    
    It uses RobotLoader for stateless mesh generation.
    """
    
    def __init__(self, robot_loader, n_ctrl_parts=1, device='cuda'):
        """
        Initialize the robot controller.
        
        Args:
            robot_loader: RobotLoader instance for mesh generation
            n_ctrl_parts: Number of control parts (default: 1)
            device: PyTorch device
        """
        self.robot_loader = robot_loader
        self.n_ctrl_parts = n_ctrl_parts
        self.device = device
        
        # Initialize force judge reference
        self.origin_force_judge = torch.tensor(
            [[-1, 0, 0], [1, 0, 0]], dtype=torch.float32, device=device
        )
        
        # Initialize to default state
        self.reset()
               
    def reset(self):
        """Reset all state variables to initial values."""
        self.accumulate_trans = np.zeros((self.n_ctrl_parts, 3), dtype=np.float32)
        self.accumulate_rot = torch.eye(3, dtype=torch.float32, device=self.device)
        self.current_finger = 0.0
        self.current_force_judge = self.origin_force_judge.clone()

        # Generate initial mesh at default state
        self._update_mesh_vertices()
        
    def set_pose(self, position, rotation=None, finger_opening=0.0):
        """
        Set complete robot pose and state.
        
        Args:
            position: np.ndarray [3] - world position
            rotation: np.ndarray [3,3] - rotation matrix (optional, defaults to identity)
            finger_opening: float [0,1] - gripper opening amount
        """
        if rotation is None:
            rotation = np.eye(3)
            
        # Update state
        self.accumulate_trans[0] = position
        self.accumulate_rot = torch.tensor(rotation, dtype=torch.float32, device=self.device)
        self.current_finger = finger_opening
        
        # Update force judge
        self.current_force_judge = self.origin_force_judge.clone() @ self.accumulate_rot
        
        # Regenerate mesh vertices
        self._update_mesh_vertices()
        
    def get_pose(self):
        """
        Get current robot pose as 4x4 transformation matrix.
        
        Returns:
            np.ndarray [4,4] - transformation matrix
        """
        transform = np.eye(4)
        transform[:3, :3] = self.accumulate_rot.cpu().numpy()
        transform[:3, 3] = self.accumulate_trans[0]
        return transform
        
    def _update_mesh_vertices(self):
        """Update current mesh vertices based on current state."""
        transform = self.get_pose()
        vertices = self.robot_loader.get_finger_vertices(
            gripper_openness=self.current_finger,
            transform=transform
        )
        self.current_trans_dynamic_points = torch.tensor(
            vertices, dtype=torch.float32, device=self.device
        )
        
    def update_robot_movement(self, target_change, finger_change=0.0, rot_change=None):
        """
        Update robot movement for one simulation step.
        
        Args:
            target_change: [3] numpy array - translation change for this step
            finger_change: float - change in finger opening (default: 0.0)
            rot_change: [3] numpy array - rotation change as axis-angle (default: None)
            
        Returns:
            dict containing:
                interpolated_dynamic_points: [num_substeps, n_robot_vertices, 3] - robot mesh points
                interpolated_center: [num_substeps, 3] - center points for each substep
                dynamic_velocity: [3] - robot velocity
                dynamic_omega: [3] - robot angular velocity
                current_trans_dynamic_points: [n_robot_vertices, 3] - final robot mesh points
        """
        # Import cfg here to avoid circular imports
        from ..qqtt.utils import cfg
        
        if rot_change is None:
            rot_change = np.zeros(3, dtype=np.float32)
            
        # Store previous state
        prev_trans_dynamic_points = self.current_trans_dynamic_points.clone()
        
        # Update finger position
        self.current_finger += finger_change
        self.current_finger = max(0.0, min(1.0, self.current_finger))
        
        # Update accumulated translation
        self.accumulate_trans += target_change
        
        # Update rotation
        new_rot = torch.tensor(rot_change, dtype=torch.float32, device=self.device)
        if torch.norm(new_rot) > 0:
            rot_mat = axis_angle_to_matrix(new_rot.unsqueeze(0))[0]
            self.accumulate_rot = torch.matmul(self.accumulate_rot, rot_mat)
        
        # Update mesh vertices
        self._update_mesh_vertices()
        
        # Calculate interpolated points for smooth movement
        ratios = (
            torch.linspace(1, cfg.num_substeps, cfg.num_substeps, device=self.device).view(-1, 1, 1)
            / cfg.num_substeps
        )
        
        interpolated_trans_dynamic_points = (
            prev_trans_dynamic_points.unsqueeze(0)
            + (self.current_trans_dynamic_points - prev_trans_dynamic_points).unsqueeze(0) * ratios
        )
        interpolated_center = torch.mean(interpolated_trans_dynamic_points, dim=1)
        
        # Handle rotation interpolation
        interpolated_rot_angle = new_rot.unsqueeze(0) * ratios.reshape(-1, 1)
        interpolated_rot_temp = axis_angle_to_matrix(interpolated_rot_angle)
        # Get rotation before update for interpolation
        prev_rot = self.accumulate_rot @ torch.inverse(axis_angle_to_matrix(new_rot.unsqueeze(0))[0])
        interpolated_rot_mat = torch.matmul(prev_rot.unsqueeze(0), interpolated_rot_temp)
        
        # Apply rotation to interpolated points
        interpolated_dynamic_points = (
            interpolated_trans_dynamic_points - interpolated_center.unsqueeze(1)
        ) @ interpolated_rot_mat.permute(0, 2, 1) + interpolated_center.unsqueeze(1)
        
        # Calculate velocity and omega
        dynamic_velocity = torch.tensor(
            target_change[0] / (2 * cfg.dt * cfg.num_substeps),
            dtype=torch.float32,
            device=self.device,
        )
        dynamic_omega = torch.tensor(
            rot_change / (2 * cfg.dt * cfg.num_substeps),
            dtype=torch.float32,
            device=self.device,
        )
        
        # Update force judge direction
        self.current_force_judge = self.origin_force_judge.clone() @ self.accumulate_rot
        
        return {
            'interpolated_dynamic_points': interpolated_dynamic_points,
            'interpolated_center': interpolated_center,
            'dynamic_velocity': dynamic_velocity,
            'dynamic_omega': dynamic_omega,
            'current_trans_dynamic_points': self.current_trans_dynamic_points
        }
        
    def set_from_mesh_vertices(self, target_vertices):
        """
        Set robot state to match given mesh vertices.
        
        This is a convenience method for the use case where we have target
        mesh vertices and want to set the robot state to match them.
        
        Args:
            target_vertices: torch.Tensor [n_vertices, 3] - target mesh vertices
        """
        target_vertices_np = target_vertices.cpu().numpy()
        
        # Calculate translation needed to match target vertices
        # Current mesh center
        current_mesh_center = torch.mean(self.current_trans_dynamic_points, dim=0).cpu().numpy()
        
        # Target mesh center  
        target_mesh_center = np.mean(target_vertices_np, axis=0)
        
        # Calculate translation from current to target
        translation = target_mesh_center - current_mesh_center
        
        # Apply translation to current position
        current_position = self.accumulate_trans[0]
        new_position = current_position + translation
        
        # Keep current rotation and finger opening
        current_rotation = self.accumulate_rot.cpu().numpy()
        current_finger = self.current_finger
        
        # Set pose and directly override with exact target vertices
        self.set_pose(new_position, current_rotation, current_finger)
        self.current_trans_dynamic_points = target_vertices.to(self.device)
        
    def get_current_state(self):
        """Get current robot state for serialization/debugging."""
        return {
            'accumulate_trans': self.accumulate_trans.copy(),
            'accumulate_rot': self.accumulate_rot.clone(),
            'current_finger': self.current_finger,
            'current_trans_dynamic_points': self.current_trans_dynamic_points.clone() if self.current_trans_dynamic_points is not None else None,
            'current_force_judge': self.current_force_judge.clone()
        }
        
    def get_current_force_judge(self):
        """Get current force judge for collision detection."""
        return self.current_force_judge
        
    def get_current_finger(self):
        """Get current finger opening value."""
        return self.current_finger 