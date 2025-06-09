import numpy as np

# ============================================================================
# DEPRECATED
# ============================================================================

def opengl2cam(pcd, cam_extrinsic, global_scale):
    """
    Transform points from OpenGL coordinate system to camera coordinate system.
    
    Args:
        pcd: (n, 3) numpy array - points in OpenGL coordinates
        cam_extrinsic: (4, 4) numpy array - camera extrinsic matrix
        global_scale: float - scaling factor
        
    Returns:
        numpy array - points in camera coordinates
    """
    opencv_T_opengl = np.array([[1, 0, 0, 0],
                                [0, -1, 0, 0],
                                [0, 0, -1, 0],
                                [0, 0, 0, 1]])
    opencv_T_world = np.matmul(np.linalg.inv(cam_extrinsic), opencv_T_opengl)
    cam = np.matmul(np.linalg.inv(opencv_T_world), 
                    np.concatenate([pcd, np.ones((pcd.shape[0], 1))], axis=1).T).T[:, :3] / global_scale
    return cam


def depth2fgpcd(depth, mask, cam_params):
    """
    Convert depth image to foreground point cloud using camera parameters.
    
    Args:
        depth: (h, w) numpy array - depth image
        mask: (h, w) numpy array - foreground mask
        cam_params: tuple - camera parameters (fx, fy, cx, cy)
        
    Returns:
        numpy array - (n, 3) point cloud in camera coordinates
    """
    h, w = depth.shape
    mask = np.logical_and(mask, depth > 0)
    fgpcd = np.zeros((mask.sum(), 3))
    fx, fy, cx, cy = cam_params
    pos_x, pos_y = np.meshgrid(np.arange(w), np.arange(h))
    pos_x = pos_x[mask]
    pos_y = pos_y[mask]
    fgpcd[:, 0] = (pos_x - cx) * depth[mask] / fx
    fgpcd[:, 1] = (pos_y - cy) * depth[mask] / fy
    fgpcd[:, 2] = depth[mask]
    return fgpcd


def pcd2pix(pcd, cam_params, offset=(0, 0)):
    """
    Project 3D point cloud to 2D pixel coordinates.
    
    Args:
        pcd: (n, 3) numpy array - 3D points in camera coordinates
        cam_params: tuple - camera parameters (fx, fy, cx, cy)
        offset: tuple - pixel offset (row_offset, col_offset)
        
    Returns:
        numpy array - (n, 2) pixel coordinates as integers (row, col)
    """
    fx, fy, cx, cy = cam_params
    pix = np.zeros((pcd.shape[0], 2))
    try:
        pix[:, 1] = pcd[:, 0] * fx / pcd[:, 2] + cx  # col
        pix[:, 0] = pcd[:, 1] * fy / pcd[:, 2] + cy  # row
        pix[:, 0] += offset[0]
        pix[:, 1] += offset[1]
    except:
        print('pcd', pcd)
        exit(1)
    return pix.astype(np.int32) 