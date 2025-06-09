"""
Mathematical operations and calculations utilities.
"""

import numpy as np

def norm(x, p=2):
    """
    Calculate p-norm of a vector.
    
    Args:
        x: array - input vector
        p: int - norm type (default: 2 for L2 norm)
        
    Returns:
        float - p-norm of the vector
    """
    return np.power(np.sum(x ** p), 1. / p)

# ============================================================================
# DEPRECATED
# ============================================================================

def calc_dis(a, b):
    """
    Calculate Euclidean distance between two 2D points.
    
    Args:
        a: tuple or array - first point (x, y)
        b: tuple or array - second point (x, y)
        
    Returns:
        float - Euclidean distance between points
    """
    return np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)


def rect_from_coord(uxi, uyi, uxf, uyf, bar_width):
    """
    Generate rectangle vertices from start/end coordinates and width.
    Used for creating rectangular regions for action visualization.
    
    Args:
        uxi, uyi: float - start point coordinates
        uxf, uyf: float - end point coordinates  
        bar_width: float - width of the rectangle
        
    Returns:
        tuple - four corner points (st0, st1, ed1, ed0) of rectangle
    """
    # transform into angular coordinates
    theta = np.arctan2(uyf - uyi, uxf - uxi)
    length = np.linalg.norm(np.array([uxf - uxi, uyf - uyi]), ord=2)

    theta0 = theta - np.pi / 2.

    v = np.array([bar_width / 2.0 * np.cos(theta0),
                  bar_width / 2.0 * np.sin(theta0)])

    st = np.array([uxi, uyi])
    ed = np.array([uxf, uyf])

    st0 = st + v
    st1 = st - v
    ed0 = ed + v
    ed1 = ed - v

    return st0, st1, ed1, ed0


def check_side(a, b):
    """
    Check which side of a line a point is on using cross product.
    
    Args:
        a: array - vector from line to point
        b: array - direction vector of the line
        
    Returns:
        float - positive if point is on one side, negative if on the other
    """
    return a[0] * b[1] - b[0] * a[1]


def check_within_rect(x, y, rect):
    """
    Check if a point is within a rectangle defined by four vertices.
    
    Args:
        x, y: float - point coordinates
        rect: tuple - four corner points of rectangle
        
    Returns:
        bool - True if point is inside rectangle
    """
    p = np.array([x, y])
    p0, p1, p2, p3 = rect

    side0 = check_side(p - p0, p1 - p0)
    side1 = check_side(p - p1, p2 - p1)
    side2 = check_side(p - p2, p3 - p2)
    side3 = check_side(p - p3, p0 - p3)

    if side0 >= 0 and side1 >= 0 and side2 >= 0 and side3 >= 0:
        return True
    elif side0 <= 0 and side1 <= 0 and side2 <= 0 and side3 <= 0:
        return True
    else:
        return False


def findClosestPoint(pcd, point):
    """
    Find the index of the closest point in a point cloud to a target point.
    
    Args:
        pcd: (n, 3) numpy array - point cloud
        point: (3,) numpy array - target point
        
    Returns:
        int - index of closest point in point cloud
    """
    dist = np.linalg.norm(pcd - point[None, :], axis=1)
    return np.argmin(dist) 