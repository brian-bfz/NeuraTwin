import numpy as np
import cv2 
from PIL import Image, ImageEnhance
import os
import torch
from .edges import construct_collision_edges, construct_topological_edges
import open3d as o3d
import pickle
import json


def visualize_edges(positions, topological_edges, tool_mask, adj_thresh, topk, connect_tools_all, colors):
    """
    Create edges visualization using construct_edges_with_attrs
    
    Args:
        positions: tensor [N, 3] - particle positions
        topological_edges: tensor [N, N] - topological adjacency matrix
        tool_mask: tensor [N] - boolean mask for tool particles
        adj_thresh: float - distance threshold for collision edges
        topk: int - maximum neighbors per particle
        connect_tools_all: bool - whether to connect tools to all objects
        colors: list of [r,g,b] colors for [collision, topological] edges
        
    Returns:
        o3d.geometry.LineSet - line set for visualization or None
    """
    
    # Add batch dimension for construct_edges_with_attrs
    positions = positions.unsqueeze(0)  # [N, 3] -> [1, N, 3]
    topological_edges = topological_edges.unsqueeze(0)  # [N, N] -> [1, N, N]
    tool_mask = tool_mask.unsqueeze(0)  # [N] -> [1, N]
    
    B, N, _ = positions.shape
    
    # Create mask for valid particles
    mask = torch.ones(B, N, dtype=torch.bool, device=positions.device)
    
    # Use construct_collision_edges to get edges
    Rr_collision, Rs_collision = construct_collision_edges(
        positions, adj_thresh, mask, tool_mask, 
        topk=topk, connect_tools_all=connect_tools_all, topological_edges=topological_edges
    )

    Rr_topo, Rs_topo, first_edge_lengths = construct_topological_edges(
        topological_edges, positions
    )
    
    # Create line sets for each edge type
    collision_lineset = create_lineset_from_Rr_Rs(Rr_collision, Rs_collision, colors[0], positions[0])
    topological_lineset = create_lineset_from_Rr_Rs(Rr_topo, Rs_topo, colors[1], positions[0])
    
    return [collision_lineset, topological_lineset]


def create_lineset_from_Rr_Rs(Rr, Rs, color, positions):
    """
    Create Open3D LineSet from sparse receiver/sender matrices
    
    Args:
        Rr: [B, n_rel, N] tensor - receiver matrix
        Rs: [B, n_rel, N] tensor - sender matrix  
        color: [r, g, b] color for edges
        positions: [N, 3] tensor - particle positions
        
    Returns:
        o3d.geometry.LineSet or None
    """
    # Convert sparse matrices to edge list
    # Find non-zero entries in Rr and Rs to get the actual edges
    rr_nonzero = Rr[0].nonzero()  # [n_edges, 2] where columns are [edge_idx, receiver_idx]
    rs_nonzero = Rs[0].nonzero()  # [n_edges, 2] where columns are [edge_idx, sender_idx]
            
    if len(rr_nonzero) > 0 and len(rs_nonzero) > 0:
        # Match edge indices to ensure we get correct sender-receiver pairs
        edge_indices = rr_nonzero[:, 0]  # Edge indices from receiver matrix
        receiver_indices = rr_nonzero[:, 1]  # Receiver particle indices
                
        # Find corresponding sender indices for the same edge indices
        rs_dict = {edge_idx.item(): sender_idx.item() for edge_idx, sender_idx in rs_nonzero}
        sender_indices = [rs_dict.get(edge_idx.item(), -1) for edge_idx in edge_indices]
                
        # Filter out any edges where sender index wasn't found
        valid_edges = [(s, r) for s, r in zip(sender_indices, receiver_indices.tolist()) if s != -1]
                
        if len(valid_edges) > 0:
            edges = np.array(valid_edges)
            positions = positions.numpy()
                    
            lineset = o3d.geometry.LineSet()
            lineset.points = o3d.utility.Vector3dVector(positions)
            lineset.lines = o3d.utility.Vector2iVector(edges)
            lineset.colors = o3d.utility.Vector3dVector(np.array([color] * len(edges)))
            return lineset
    
    return None


def gen_goal_shape(name, h, w, font_name='helvetica_thin'):
    """
    Generate goal shape from font.
    
    Args:
        name: str - character/shape name
        h, w: int - target height and width
        font_name: str - font name
        
    Returns:
        tuple: (goal_dist, goal_img) - distance transform and image
    """
    root_dir = f'env/target_shapes/{font_name}'
    shape_path = os.path.join(root_dir, 'helvetica_' + name + '.npy')
    goal = np.load(shape_path)
    goal = cv2.resize(goal, (w, h), interpolation=cv2.INTER_AREA)
    goal = (goal <= 0.5).astype(np.uint8)
    goal_dist = np.minimum(cv2.distanceTransform(1-goal, cv2.DIST_L2, 5), 1e4)
    goal_img = (goal * 255)[..., None].repeat(3, axis=-1).astype(np.uint8)
    return goal_dist, goal_img


# ============================================================================
# DEPRECATED
# ============================================================================

def create_edges_for_points(positions, distance_threshold):
    """
    Create connectivity edges between nearby particles for visualization.
        
    Args:
        positions: [n_points, 3] - particle positions
        distance_threshold: float - maximum distance for connections
            
    Returns:
        edges: [n_edges, 2] - indices of connected particle pairs
    """
    edges = []
    n_points = positions.shape[0]
        
    for i in range(n_points):
        for j in range(i + 1, n_points):
            distance = np.linalg.norm(positions[i] - positions[j])
            if distance <= distance_threshold:
                edges.append([i, j])
            
    return np.array(edges) if edges else np.empty((0, 2), dtype=int)


def drawRotatedRect(img, s, e, width=1):
    """
    Draw a rotated rectangle on image with color gradient.
    
    Args:
        img: (h, w, 3) numpy array - input image
        s: (x, y) tuple - start point
        e: (x, y) tuple - end point
        width: int - rectangle width
        
    Returns:
        numpy array - image with drawn rectangle
    """
    l = int(np.sqrt((s[0] - e[0]) ** 2 + (s[1] - e[1]) ** 2) + 1)
    theta = np.arctan2(e[1] - s[1], e[0] - s[0])
    theta_ortho = theta + np.pi / 2
    for i in range(l):
        color = (255, int(255 * i / l), 0)
        x = int(s[0] + (e[0] - s[0]) * i / l)
        y = int(s[1] + (e[1] - s[1]) * i / l)
        img = cv2.line(img.copy(), (int(x - 0.5 * width * np.cos(theta_ortho)), int(y - 0.5 * width * np.sin(theta_ortho))), 
                    (int(x + 0.5 * width * np.cos(theta_ortho)), int(y + 0.5 * width * np.sin(theta_ortho))), color, 1)
    return img


def drawPushing(img, s, e, width):
    """
    Draw pushing action visualization on image.
    
    Args:
        img: (h, w, 3) numpy array - input image
        s: (x, y) tuple - start point
        e: (x, y) tuple - end point
        width: float - line width
        
    Returns:
        numpy array - image with drawn pushing action
    """
    l = int(np.sqrt((s[0] - e[0]) ** 2 + (s[1] - e[1]) ** 2) + 1)
    theta = np.arctan2(e[1] - s[1], e[0] - s[0])
    theta_ortho = theta + np.pi / 2
    img = cv2.line(img.copy(), (int(s[0] - 0.5 * width * np.cos(theta_ortho)), int(s[1] - 0.5 * width * np.sin(theta_ortho))), 
                (int(s[0] + 0.5 * width * np.cos(theta_ortho)), int(s[1] + 0.5 * width * np.sin(theta_ortho))), (255,99,71), 5)
    img = cv2.line(img.copy(), (int(e[0] - 0.5 * width * np.cos(theta_ortho)), int(e[1] - 0.5 * width * np.sin(theta_ortho))),
                (int(e[0] + 0.5 * width * np.cos(theta_ortho)), int(e[1] + 0.5 * width * np.sin(theta_ortho))), (255,99,71), 5)
    img = cv2.arrowedLine(img.copy(), (int(s[0]), int(s[1])), (int(e[0]), int(e[1])), (255,99,71), 5)
    return img


def gt_rewards(mask, subgoal):
    """
    Calculate ground truth reward based on object mask and subgoal.
    
    Args:
        mask: (h, w) numpy array - object mask
        subgoal: (h, w) numpy array - subgoal region
        
    Returns:
        float - reward value
    """
    subgoal_mask = subgoal < 0.5
    obj_dist = cv2.distanceTransform(1 - mask.astype(np.uint8), cv2.DIST_L2, 5)
    return np.sum(mask * subgoal) / mask.sum() + np.sum(obj_dist * subgoal_mask) / subgoal_mask.sum()


def gt_rewards_norm_by_sum(mask, subgoal):
    """
    Calculate ground truth reward normalized by sum.
    
    Args:
        mask: (h, w) numpy array - object mask
        subgoal: (h, w) numpy array - subgoal region
        
    Returns:
        float - normalized reward value
    """
    subgoal_mask = subgoal < 0.5
    obj_dist = cv2.distanceTransform(1 - mask.astype(np.uint8), cv2.DIST_L2, 5)
    return np.sum(mask * subgoal) / subgoal.sum() + np.sum(obj_dist * subgoal_mask) / obj_dist.sum()


def gen_ch_goal(name, h, w):
    """
    Generate Chinese character goal shape.
    
    Args:
        name: str - character name
        h, w: int - target height and width
        
    Returns:
        tuple: (goal_dist, goal_img) - distance transform and image
    """
    root_dir = 'env/target_shapes/720_ch'
    shape_path = os.path.join(root_dir, name + '.npy')
    goal = np.load(shape_path)
    goal = cv2.resize(goal, (w, h), interpolation=cv2.INTER_AREA)
    goal = (goal <= 0.5).astype(np.uint8)
    goal_dist = cv2.distanceTransform(1-goal, cv2.DIST_L2, 5)
    goal_img = (goal * 255)[..., None].repeat(3, axis=-1).astype(np.uint8)
    return goal_dist, goal_img


def gen_subgoal(c_row, c_col, r, h=64, w=64):
    """
    Generate circular subgoal region.
    
    Args:
        c_row, c_col: int - center coordinates
        r: float - radius
        h, w: int - image dimensions
        
    Returns:
        tuple: (subgoal, mask) - distance transform and binary mask
    """
    mask = np.zeros((h, w))
    grid = np.mgrid[0:h, 0:w]
    grid[0] = grid[0] - c_row
    grid[1] = grid[1] - c_col
    dist = np.sqrt(np.sum(grid**2, axis=0))
    mask[dist < r] = 1
    subgoal = np.minimum(cv2.distanceTransform((1-mask).astype(np.uint8), cv2.DIST_L2, 5), 1e4)
    return subgoal, mask


# Color constants
dodger_blue_RGB = (30, 144, 255)
dodger_blue_BGR = (255, 144, 30)
tomato_RGB = (255, 99, 71)
tomato_BGR = (71, 99, 255)


def lighten_img(img, factor=1.2):
    # img: assuming an RGB image
    assert img.dtype == np.uint8
    assert img.shape[2] == 3
    cv2.imwrite('tmp_1.png', img)
    img = Image.open('tmp_1.png').convert("RGB")
    img_enhancer = ImageEnhance.Brightness(img)
    enhanced_output = img_enhancer.enhance(factor)
    enhanced_output.save("tmp_2.png")
    color_lighten_img = cv2.imread('tmp_2.png')
    os.system('rm tmp_1.png tmp_2.png')
    return color_lighten_img


class Visualizer:
    """
    Visualizer for particle trajectories with camera calibration.
    Handles 3D visualization of particle motion with proper camera setup.
    """
    
    def __init__(self, camera_calib_path, downsample_rate=1):
        """
        Initialize visualizer with camera calibration data.
        
        Args:
            camera_calib_path: str - path to camera calibration data directory
        """
        # Load camera to world transforms
        with open(os.path.join(camera_calib_path, "calibrate.pkl"), "rb") as f:
            self.c2ws = pickle.load(f)
        self.w2cs = [np.linalg.inv(c2w) for c2w in self.c2ws]
            
        with open(os.path.join(camera_calib_path, "metadata.json"), "r") as f:
            data = json.load(f)
        self.intrinsics = np.array(data["intrinsics"])
        self.WH = data["WH"]
        self.FPS = data["fps"] / downsample_rate  # Handle missing downsample_rate
        
        print(f"Loaded camera calibration from: {camera_calib_path}")
        print(f"Resolution: {self.WH}, FPS: {self.FPS}")

    def visualize_object_motion(self, predicted_states, tool_mask, actual_objects, save_path, topological_edges=None, target=None):
        """
        Create 3D visualization comparing predicted vs actual object motion.
        Renders particles as colored point clouds with connectivity edges.
        
        Args:
            predicted_states: [timesteps, n_particles, 3] tensor - predicted trajectory (objects + robots)
            tool_mask: [n_particles] tensor - boolean mask (False=object, True=robot)
            actual_objects: [timesteps, n_obj, 3] tensor - ground truth object trajectory  
            save_path: str - output video file path
            topological_edges: [N, N] tensor - topological edges for object and robot particles
            target: [N, 3] tensor - target point cloud for MPC
            
        Returns:
            save_path: str - path where video was saved
        """        
        # Video parameters
        width, height = self.WH
        fps = self.FPS
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        out = cv2.VideoWriter(save_path, fourcc, fps, (width, height))
        
        # Create Open3D visualizer
        vis = o3d.visualization.Visualizer()
        vis.create_window(width=width, height=height, visible=False)
        
        render_option = vis.get_render_option()
        render_option.point_size = 10.0

        # Move tensors to CPU
        predicted_states = predicted_states.cpu()
        tool_mask = tool_mask.cpu()
        if topological_edges is not None:
            topological_edges = topological_edges.cpu()
        
        # Split predicted states into objects and robots using tool_mask. Convert to numpy for o3d. Keep predicted_states as tensor. 
        actual_objects = actual_objects.cpu().numpy()
        pred_objects = predicted_states[:, ~tool_mask, :].numpy()
        pred_robots = predicted_states[:, tool_mask, :].numpy()
        
        # Create point clouds            
        actual_pcd = o3d.geometry.PointCloud()
        actual_pcd.points = o3d.utility.Vector3dVector(actual_objects[0])
        actual_pcd.paint_uniform_color([0.0, 0.0, 1.0])  # Blue for actual
            
        robot_pcd = o3d.geometry.PointCloud()
        robot_pcd.points = o3d.utility.Vector3dVector(pred_robots[0])
        robot_pcd.paint_uniform_color([1.0, 0.0, 0.0])  # Red for robot
        
        pred_pcd = o3d.geometry.PointCloud()
        pred_pcd.points = o3d.utility.Vector3dVector(pred_objects[0])
        pred_pcd.paint_uniform_color([0.0, 1.0, 0.0])  # Green for predicted

        target_pcd = o3d.geometry.PointCloud()
        target_pcd.points = o3d.utility.Vector3dVector(target)
        target_pcd.paint_uniform_color([1.0, 0.6, 0.2])  # Orange for target

        # Add geometries to visualizer
        vis.add_geometry(actual_pcd)
        vis.add_geometry(robot_pcd)
        vis.add_geometry(pred_pcd)

        n_frames = min(len(predicted_states), len(actual_objects))
        print(f"Rendering {n_frames} frames...")
        
        # Set up camera parameters if available
        if self.w2cs is not None:
            view_control = vis.get_view_control()
            camera_params = o3d.camera.PinholeCameraParameters()
            intrinsic_parameter = o3d.camera.PinholeCameraIntrinsic(
                width, height, self.intrinsics[0]  # Using first camera
            )
            camera_params.intrinsic = intrinsic_parameter
            camera_params.extrinsic = self.w2cs[0]
            view_control.convert_from_pinhole_camera_parameters(
                camera_params, allow_arbitrary=True
            )

        # Initialize edge sets
        pred_line_sets = []

        # Render each frame
        for frame_idx in range(n_frames):
            # Update particle positions
            pred_obj_pos = pred_objects[frame_idx]
            actual_obj_pos = actual_objects[frame_idx]
            robot_pos = pred_robots[frame_idx]
            
            if frame_idx == 0:
                print(f"Sampled object particles: {pred_obj_pos.shape[0]}")
                print(f"Robot particles: {robot_pos.shape[0]}")
            
            # Update point cloud positions
            pred_pcd.points = o3d.utility.Vector3dVector(pred_obj_pos)
            robot_pcd.points = o3d.utility.Vector3dVector(robot_pos)
            actual_pcd.points = o3d.utility.Vector3dVector(actual_obj_pos)

            vis.update_geometry(pred_pcd)
            vis.update_geometry(actual_pcd)
            vis.update_geometry(robot_pcd)

            # Remove old edges
            for line_set in pred_line_sets:
                if line_set is not None:
                    vis.remove_geometry(line_set, reset_bounding_box=False)
            
            # Add topological edges if available
            if topological_edges is not None:
                pred_line_sets = visualize_edges(
                    predicted_states[frame_idx], topological_edges, tool_mask, 
                    adj_thresh=0.05, topk=5, connect_tools_all=False, 
                    colors=[[1.0, 0.6, 0.2], [0.3, 0.6, 0.3]]  # light orange, light green
                )
                for line_set in pred_line_sets:
                    if line_set is not None:
                        vis.add_geometry(line_set, reset_bounding_box=False)
                        
            # Render frame
            vis.poll_events()
            vis.update_renderer()
            
            static_image = np.asarray(
                vis.capture_screen_float_buffer(do_render=True)
            )
            static_image = (static_image * 255).astype(np.uint8)
            static_image_bgr = cv2.cvtColor(static_image, cv2.COLOR_RGB2BGR)
            out.write(static_image_bgr)
                        
            if frame_idx % 10 == 0:
                print(f"  Rendered frame {frame_idx}/{n_frames}")
        
        # Cleanup
        out.release()
        vis.destroy_window()
        
        print(f"Video saved to: {save_path}")
        return save_path
