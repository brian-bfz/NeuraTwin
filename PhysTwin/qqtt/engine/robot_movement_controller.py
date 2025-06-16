import torch
import numpy as np
from ..utils import cfg


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


class RobotMovementController:
    """Handles robot movement, interpolation, and physics simulation updates"""
    
    def __init__(self, robot, n_ctrl_parts=1, device='cuda'):
        """
        Initialize the robot movement controller.
        
        Args:
            robot: Robot instance with get_finger_mesh method
            n_ctrl_parts: Number of control parts (default: 1)
            device: PyTorch device
        """
        self.robot = robot
        self.n_ctrl_parts = n_ctrl_parts
        self.device = device
        
        # Initialize state variables
        self.origin_force_judge = torch.tensor(
            [[-1, 0, 0], [1, 0, 0]], dtype=torch.float32, device=device
        )
        self.reset()
               
    def reset(self):
        """Reset all state variables to initial values"""
        self.accumulate_trans = np.zeros((self.n_ctrl_parts, 3), dtype=np.float32)
        self.accumulate_rot = torch.eye(3, dtype=torch.float32, device=self.device)
        self.current_finger = 0.0
        self.current_force_judge = self.origin_force_judge.clone()

        finger_meshes = self.robot.get_finger_mesh(0.0)
        dynamic_vertices = torch.tensor(
            [np.asarray(finger_mesh.vertices) for finger_mesh in finger_meshes],
            device=self.device,
            dtype=torch.float32
        )
        self.current_trans_dynamic_points = torch.reshape(dynamic_vertices, (-1, 3))
        
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
        if rot_change is None:
            rot_change = np.zeros(3, dtype=np.float32)
            
        # Update finger position
        self.current_finger += finger_change
        self.current_finger = max(0.0, min(1.0, self.current_finger))
        
        # Update accumulated translation
        self.accumulate_trans += target_change
        
        # Get finger meshes and calculate dynamic vertices
        finger_meshes = self.robot.get_finger_mesh(self.current_finger)
        dynamic_vertices = torch.tensor(
            [np.asarray(finger_mesh.vertices) + self.accumulate_trans[0] for finger_mesh in finger_meshes],
            device=self.device,
            dtype=torch.float32
        )
        
        # Store previous state and update current
        prev_trans_dynamic_points = self.current_trans_dynamic_points
        self.current_trans_dynamic_points = torch.reshape(dynamic_vertices, (-1, 3))
        
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
        
        # Handle rotation
        new_rot = torch.tensor(rot_change, dtype=torch.float32, device=self.device)
        interpolated_rot_angle = new_rot.unsqueeze(0) * ratios.reshape(-1, 1)
        interpolated_rot_temp = axis_angle_to_matrix(interpolated_rot_angle)
        interpolated_rot_mat = torch.matmul(self.accumulate_rot.unsqueeze(0), interpolated_rot_temp)
        self.accumulate_rot = interpolated_rot_mat[-1]
        
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
        self.current_force_judge = self.origin_force_judge.clone() @ interpolated_rot_mat[-1]
        
        return {
            'interpolated_dynamic_points': interpolated_dynamic_points,
            'interpolated_center': interpolated_center,
            'dynamic_velocity': dynamic_velocity,
            'dynamic_omega': dynamic_omega,
            'current_trans_dynamic_points': self.current_trans_dynamic_points
        }
        
    def get_current_state(self):
        """Get current robot state for serialization/debugging"""
        return {
            'accumulate_trans': self.accumulate_trans.copy(),
            'accumulate_rot': self.accumulate_rot.clone(),
            'current_finger': self.current_finger,
            'current_trans_dynamic_points': self.current_trans_dynamic_points.clone() if self.current_trans_dynamic_points is not None else None,
            'current_force_judge': self.current_force_judge.clone()
        }
        
    def get_current_force_judge(self):
        """Get current force judge for collision detection"""
        return self.current_force_judge
        
    def get_current_finger(self):
        """Get current finger opening value"""
        return self.current_finger 