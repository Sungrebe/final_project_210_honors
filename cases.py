import numpy as np
import cv2

def blur_Image(size, sigma, window):
    """
    Apply Gaussian blur to the window (ROI).
    - size: Kernel size for blurring (odd integer).
    - sigma: Standard deviation for Gaussian blur.
    - window: The region of the frame to apply the blur effect.
    """
    if size % 2 == 0:  # Ensure the kernel size is odd
        size += 1
    return cv2.GaussianBlur(window, (size, size), sigma)


def sharpen_Image(size, level, window):
    """
    Apply median blur to the window (ROI) instead of sharpening.
    - size: Determines the kernel size (must be an odd integer, e.g., 3, 5, 7).
    - level: Not used for median blur but kept for interface consistency.
    - window: The region of the frame to apply the blur effect.
    """
    # Ensure `size` is odd; if not, make it the next odd integer
    if size % 2 == 0:
        size += 1

    # Apply median blur using cv2.medianBlur
    return cv2.medianBlur(window, size)




def pass_Image(window):
    """
    Return the window (ROI) unchanged.
    """
    return window


