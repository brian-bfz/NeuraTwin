"""
Image processing and manipulation utilities.
"""

import numpy as np
import cv2
from PIL import Image, ImageEnhance

# ============================================================================
# DEPRECATED
# ============================================================================

def resize(img, size, interpolation=Image.BILINEAR):
    """
    Resize PIL image to target size while maintaining aspect ratio.
    
    Args:
        img: PIL.Image - input image
        size: int or tuple - target size (int for shorter side, tuple for (width, height))
        interpolation: PIL interpolation mode
        
    Returns:
        PIL.Image - resized image
    """
    if isinstance(size, int):
        w, h = img.size
        if (w <= h and w == size) or (h <= w and h == size):
            return img
        if w < h:
            ow = size
            oh = int(size * h / w)
            return img.resize((ow, oh), interpolation)
        else:
            oh = size
            ow = int(size * w / h)
            return img.resize((ow, oh), interpolation)
    else:
        return img.resize(size[::-1], interpolation)


def crop(img, i, j, h, w):
    """
    Crop PIL image to specified rectangle.
    
    Args:
        img: PIL.Image - input image
        i, j: int - top-left corner coordinates
        h, w: int - crop height and width
        
    Returns:
        PIL.Image - cropped image
    """
    return img.crop((j, i, j + w, i + h))


def adjust_brightness(img, brightness_factor):
    """
    Adjust brightness of PIL image.
    
    Args:
        img: PIL.Image - input image
        brightness_factor: float - brightness multiplier (1.0 = no change)
        
    Returns:
        PIL.Image - brightness-adjusted image
    """
    enhancer = ImageEnhance.Brightness(img)
    img = enhancer.enhance(brightness_factor)
    return img


def adjust_contrast(img, contrast_factor):
    """
    Adjust contrast of PIL image.
    
    Args:
        img: PIL.Image - input image
        contrast_factor: float - contrast multiplier (1.0 = no change)
        
    Returns:
        PIL.Image - contrast-adjusted image
    """
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(contrast_factor)
    return img


def adjust_saturation(img, saturation_factor):
    """
    Adjust color saturation of PIL image.
    
    Args:
        img: PIL.Image - input image
        saturation_factor: float - saturation multiplier (1.0 = no change)
        
    Returns:
        PIL.Image - saturation-adjusted image
    """
    enhancer = ImageEnhance.Color(img)
    img = enhancer.enhance(saturation_factor)
    return img


def adjust_hue(img, hue_factor):
    """
    Adjust hue of PIL image.
    
    Args:
        img: PIL.Image - input image
        hue_factor: float - hue shift in range [-0.5, 0.5]
        
    Returns:
        PIL.Image - hue-adjusted image
    """
    if not(-0.5 <= hue_factor <= 0.5):
        raise ValueError('hue_factor is not in [-0.5, 0.5].'.format(hue_factor))

    input_mode = img.mode
    if input_mode in {'L', '1', 'I', 'F'}:
        return img

    h, s, v = img.convert('HSV').split()

    np_h = np.array(h, dtype=np.uint8)
    # uint8 addition take cares of rotation across boundaries
    with np.errstate(over='ignore'):
        np_h += np.uint8(hue_factor * 255)
    h = Image.fromarray(np_h, 'L')

    img = Image.merge('HSV', (h, s, v)).convert(input_mode)
    return img


def adjust_gamma(img, gamma, gain=1):
    """
    Apply gamma correction to PIL image.
    
    Args:
        img: PIL.Image - input image
        gamma: float - gamma value (must be non-negative)
        gain: float - gain factor (default: 1)
        
    Returns:
        PIL.Image - gamma-corrected image
    """
    if gamma < 0:
        raise ValueError('Gamma should be a non-negative real number')

    input_mode = img.mode
    img = img.convert('RGB')

    gamma_map = [255 * gain * pow(ele / 255., gamma) for ele in range(256)] * 3
    img = img.point(gamma_map)  # use PIL's point-function to accelerate this part

    img = img.convert(input_mode)
    return img


def rmbg(img, bg):
    """
    Remove background from image by comparing with background image.
    
    Args:
        img: (h, w, 3) numpy array - input image (uint8)
        bg: (h, w, 3) numpy array - background image (uint8)
        
    Returns:
        numpy array - image with background set to white
    """
    assert img.shape == bg.shape
    assert img.dtype == np.uint8
    img_diff = np.abs(img.astype(np.int32) - bg.astype(np.int32)).sum(axis=2)
    img[img_diff < 1e-3] = np.ones(3, dtype=np.uint8) * 255
    return img 