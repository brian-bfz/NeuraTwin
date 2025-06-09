import numpy as np
import cv2 
from PIL import Image, ImageEnhance
import os
import torch
from .edges import construct_edges_with_attrs
import open3d as o3d



def visualize_edges(positions, adj_thresh, topk, connect_tools_all, topological_edges, colors):
    # Create new edges for predicted objects
    # Convert to torch tensors for construct_edges_with_attrs
    positions = torch.tensor(positions, dtype=torch.float32, device='cpu').unsqueeze(0)  # [1, N, 3]
    N = positions.shape[1]
            
    # Create masks - all particles are valid objects (not tools)
    mask = torch.ones(1, N, dtype=torch.bool)
    tool_mask = torch.zeros(1, N, dtype=torch.bool)

    # Use construct_edges_with_attrs to get collision edges
    Rr, Rs, edge_attrs = construct_edges_with_attrs(
        positions, adj_thresh, mask, tool_mask, 
        topk=topk, connect_tools_all=connect_tools_all, topological_edges=topological_edges
    )

    return create_lineset_from_Rr_Rs(Rr, Rs, edge_attrs, colors, positions)


def create_lineset_from_Rr_Rs(Rr, Rs, edge_attrs, colors, positions):
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
            edge_attrs_flat = edge_attrs[0, edge_indices, 0].numpy()
                    
            lineset = o3d.geometry.LineSet()
            lineset.points = o3d.utility.Vector3dVector(positions)
            lineset.lines = o3d.utility.Vector2iVector(edges)
                    
            # Color edges based on type: lighter red for topological (1), even lighter for collision (0)
            # Base color is red [1.0, 0.0, 0.0] to match predicted object points
            line_colors = []
            for attr in edge_attrs_flat:
                if attr > 0.5:  # Topological edge
                    line_colors.append(colors[0]) 
                else:  # Collision edge
                    line_colors.append(colors[1]) 
                    
            lineset.colors = o3d.utility.Vector3dVector(np.array(line_colors))
            return lineset


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
