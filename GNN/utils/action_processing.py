"""
Action preprocessing and transformation utilities.
"""

import numpy as np
import torch
from .math_ops import rect_from_coord, check_within_rect

# ============================================================================
# DEPRECATED
# ============================================================================

def preprocess_action_segment(act):
    """
    Generate action frame to illustrate pushing segment.
    Each position in the pushing segment contains the offset to the end.
    
    Args:
        act: (4,) numpy array - action coordinates [start_x, start_y, end_x, end_y] 
        
    Returns:
        numpy array - flattened action frame representation
    """
    width = 32
    height = 32
    bar_width = 32. / 500 * 80

    act = act + 0.5

    act_frame = np.zeros((2, height, width))

    uxi = float(width) * act[0]
    uyi = float(height) * act[1]
    uxf = float(width) * act[2]
    uyf = float(height) * act[3]

    st = np.array([uxi, uyi])
    ed = np.array([uxf, uyf])

    rect = rect_from_coord(uxi, uyi, uxf, uyf, bar_width)

    direct = np.array([uxf - uxi, uyf - uyi])
    direct = direct / np.linalg.norm(direct, ord=2)

    for i in range(height):
        for j in range(width):
            x = j + 0.5
            y = (height - i) - 0.5
            cur = np.array([x, y])

            if check_within_rect(x, y, rect):
                to_ed = ed - cur
                to_ed = to_ed / np.linalg.norm(to_ed, ord=2)
                angle = np.arccos(np.dot(direct, to_ed))

                length = np.linalg.norm(ed - cur, ord=2) * np.cos(angle)
                offset = length * direct

                act_frame[:, i, j] = offset / np.array([width, height])

    return act_frame.reshape(-1)


def preprocess_action_repeat(act, width=32, height=32):
    """
    Generate action frame by appending spatial coordinates with action.
    Each position contains the coordinate and the action.
    
    Args:
        act: (action_dim,) numpy array - action to repeat
        width: int - frame width
        height: int - frame height
        
    Returns:
        numpy array - flattened action frame with spatial coordinates
    """
    act_dim = act.shape[0]
    act_frame = np.zeros((act_dim+2, height, width))

    act_frame[2:] = np.tile(act.reshape(-1, 1, 1), (1, height, width))
    width_1d = (np.arange(width) + 0.5) / width - 0.5
    height_1d = (height - np.arange(height) - 0.5) / height - 0.5
    act_frame[0] = np.tile(width_1d.reshape(1, 1, -1), (1, height, 1))
    act_frame[1] = np.tile(height_1d.reshape(1, -1, 1), (1, 1, width))

    return act_frame.reshape(-1)


def preprocess_action_repeat_tensor(act, width=32, height=32, pos_enc=None):
    """
    Generate action frame by appending spatial coordinates with action (tensor version).
    Each position contains the coordinate and the action.
    
    Args:
        act: (B, action_dim) torch.Tensor - batch of actions
        width: int - frame width  
        height: int - frame height
        pos_enc: torch.Tensor - precomputed positional encoding (optional)
        
    Returns:
        torch.Tensor - (B, (action_dim+2) * height * width) flattened action frames
    """
    assert type(act) == torch.Tensor

    B, act_dim = act.size()

    act_frame = torch.zeros((B, 2 + act_dim, height, width), dtype=torch.float32, device=act.device)
    if pos_enc is not None:
        act_frame[:, :2] = pos_enc.repeat(B, 1, 1, 1)
    else:
        act_frame[:, 0] = ((torch.arange(width).reshape(1, 1, -1) + 0.5) / width - 0.5).repeat(B, height, 1)
        act_frame[:, 1] = ((height - torch.arange(height).reshape(1, -1, 1) - 0.5) / height - 0.5).repeat(B, 1, width)
    act_frame[:, 2:] = act.reshape(B, act_dim, 1, 1).repeat(1, 1, height, width)

    # act_frame: B x (6 * height * width)
    return act_frame.view(B, -1).cuda() 