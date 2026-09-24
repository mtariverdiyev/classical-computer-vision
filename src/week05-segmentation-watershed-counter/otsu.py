import cv2 as cv
import numpy as np

def otsu_threshold(image_gray):
    """
    Apply Otsu's thresholding method to a grayscale image and return the optimal threshold value and the binary image.
    Parameters:
    - image_gray: A single-channel grayscale image (numpy array).
    Returns:
    - optimal_thresh: The optimal threshold value determined by Otsu's method.
    - binary_img: The binary image obtained by applying the optimal threshold.
    """
    if image_gray.ndim != 2:
        raise ValueError("otsu_threshold expects a single-channel grayscale image")

    # 1. Build normalized histogram (probability of each intensity level)
    hist, _ = np.histogram(image_gray.ravel(), bins=256, range=(0, 256))
    total_pixels = image_gray.size
    prob = hist.astype(np.float64) / total_pixels

    # 2. Precompute cumulative sums needed for fast between-class variance
    levels = np.arange(256)
    omega = np.cumsum(prob)
    mu = np.cumsum(levels * prob)
    mu_total = mu[-1]

    # 3. Between-class variance for every possible threshold t:
    with np.errstate(divide="ignore", invalid="ignore"):
        numerator = (mu_total * omega - mu) ** 2
        denominator = omega * (1.0 - omega)
        sigma_b_squared = np.where(denominator > 0, numerator / denominator, 0.0)

    # 4. The optimal threshold maximizes between-class variance
    optimal_thresh = int(np.argmax(sigma_b_squared))

    # 5. Apply threshold to produce the binary image
    binary_img = np.where(image_gray > optimal_thresh, 255, 0).astype(np.uint8)

    return optimal_thresh, binary_img
