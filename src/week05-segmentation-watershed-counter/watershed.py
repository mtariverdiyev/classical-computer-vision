import cv2 as cv
import numpy as np
from otsu import otsu_threshold

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
    
    _, binary = otsu_threshold(image_gray)
    
    kernel = np.ones((3, 3), np.uint8)
    opening = cv.morphologyEx(binary, cv.MORPH_OPEN, kernel, iterations=4)
    opening_closing = cv.morphologyEx(opening, cv.MORPH_CLOSE, kernel, iterations=4)
    
    sure_background = cv.dilate(opening_closing, kernel, iterations=3)
    
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
