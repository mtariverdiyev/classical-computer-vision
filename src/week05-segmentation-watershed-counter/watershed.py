import cv2 as cv
import numpy as np
from otsu import otsu_threshold

def fix_polarity(binary):
    """
    We fix the polarity based on the heuristic below:
    Heuristic: coins essentially never touch all four edges of the image, 
    so whichever class dominates the image border is almost
    certainly background. If the 255-class dominates the border, the
    mask is inverted relative to what the pipeline expects, so flip it.
    """    
    edges = np.concatenate([binary[0, :], binary[-1, :], binary[:, 0], binary[:, -1]])
    frac_white_on_edges = np.mean(edges == 255)
    return cv.bitwise_not(binary) if frac_white_on_edges > 0.5 else binary        

def get_adaptive_params(image_shape):
    """
    Compute resolution-scaled parameters (blur kernel, morphology kernel,
    and peak min_distance) from an image's diagonal.

    The dataset mixes tiny thumbnails with high-res macro shots, so fixed
    pixel sizes cannot work across the range. Scaling to the diagonal keeps
    each parameter roughly proportional to the expected coin size in a
    given photo, regardless of its resolution.

    Returns
    -------
    (blur_kernel_size, morph_kernel_size, min_distance) : tuple of int
        All odd (except where clamped to the minimum), safe for OpenCV.
    """
    diagonal = np.sqrt(image_shape[0] ** 2 + image_shape[1] ** 2)

    blur_kernel_size = max(5, int(round(diagonal / 60)))
    if blur_kernel_size % 2 == 0:
        blur_kernel_size += 1
 
    morph_kernel_size = max(3, int(round(diagonal / 150)))
    if morph_kernel_size % 2 == 0:
        morph_kernel_size += 1  # odd kernel sizes are centered symmetrically
    min_distance = max(5, int(round(diagonal / 60)))
    return blur_kernel_size, morph_kernel_size, min_distance

def watershed(image_bgr):
    """
    Perform Watershed segmentation on a BGR image and return the count of detected objects and the segmented image.
    Parameters:
    - image_bgr: A BGR image (numpy array).
    Returns:
    - count: The number of detected objects (coins).
    - segmented_image: The image with segmented regions colored differently and boundaries marked in red.
    """
    image_gray = cv.cvtColor(image_bgr, cv.COLOR_BGR2GRAY)
    image_gray = cv.GaussianBlur(image_gray, (5, 5), 0)

    blur_kernel_size, morph_kernel_size, min_distance = get_adaptive_params(image_gray.shape)
    
    for _ in range(2):
        image_gray = cv.bilateralFilter(image_gray, d = -1, sigmaColor = 60, sigmaSpace=blur_kernel_size)

    _, binary = otsu_threshold(image_gray)

    binary = fix_polarity(binary)

    kernel = np.ones((morph_kernel_size, morph_kernel_size), np.uint8)
    opening = cv.morphologyEx(binary, cv.MORPH_OPEN, kernel, iterations=2)
    opening_closing = cv.morphologyEx(opening, cv.MORPH_CLOSE, kernel, iterations=2)
   
    sure_background = cv.dilate(opening_closing, kernel, iterations=2)
    distance_transform = cv.distanceTransform(opening_closing, cv.DIST_L2, 5) 
   
    _, sure_foreground = cv.threshold(distance_transform, 0.5 * distance_transform.max(), 255, 0)
    sure_foreground = np.uint8(sure_foreground)
    
    unknown = cv.subtract(sure_background, sure_foreground)
    
    ret, markers = cv.connectedComponents(sure_foreground)
    markers = markers + 1
    markers[unknown == 255] = 0
    
    markers = cv.watershed(image_bgr, markers)
    
    max_label = np.max(markers)
    colors = np.random.randint(0, 255, size=(max_label + 1, 3), dtype=np.uint8)
    colors[0] = [0, 0, 0]
    if max_label >= 1:
        colors[1] = [0, 0, 0]
    
    temp_markers = markers.copy()
    temp_markers[temp_markers == -1] = 0
    colored_image = colors[temp_markers]
    colored_image[markers == -1] = [0, 0, 255] 
    
    return ret - 1, colored_image
