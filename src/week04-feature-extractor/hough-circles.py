import cv2 as cv
from matplotlib import image
import numpy as np

def detect_circles(image_gray):
    """
    Detect circles in a grayscale image using the Hough Circle Transform.
    """
    image_blur = cv.medianBlur(image_gray, ksize=5)
    circles = cv.HoughCircles(image_blur, cv.HOUGH_GRADIENT, dp=1, minDist=image_blur.shape[0]/30, param1=200, param2=22, minRadius=5, maxRadius=30)

    return circles


def draw_circles(image_bgr, circles):
    """
    Draw detected circles on the original BGR image.
    """
    if circles is not None:
        # Reshape to 2D array of (x, y, radius) and round coordinate values
        circles = np.round(circles[0, :]).astype(np.uint32)

        for (a, b, r) in circles:
            # Draw circle outer boundary (Red, thickness = 2)
            cv.circle(image_bgr, (a, b), r, color=(0, 0, 255), thickness=2)

            # Draw circle center point (Green, filled circle)
            cv.circle(image_bgr, (a, b), 2, color=(0, 255, 0), thickness=-1)

            # Calculate and render diameter text label above center
            diam = r * 2
            cv.putText(
                image_bgr,
                f"D = {diam}",
                (a - r, b),
                cv.FONT_HERSHEY_SIMPLEX,
                0.3,
                color=(255, 255, 255),
                thickness=1
            )
        return image_bgr   
    else:
        raise ValueError("No circles detected in the image.")

def hough_circles(image_bgr):
    """
    Perform Hough Circle Transform on a BGR image and return the image with detected circles drawn.
    """
    image_gray = cv.cvtColor(image_bgr, cv.COLOR_BGR2GRAY)
    circles = detect_circles(image_gray)
    return draw_circles(image_bgr, circles)