import cv2 as cv
import numpy as np

def detect_document_corners(imageBW):
    """
    Find the 4-corner outline of a document (ID card, license plate, etc.)
    in a binary Canny edge map, and return its corners as a polygon.
 
    Returns the corner points of the largest roughly-quadrilateral contour
    found, or None if nothing suitable is detected.
    """

    edges = cv.Canny(imageBW, 50, 150)
    contours, _ = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    # Raw Canny edges are often broken into small disconnected fragments. Dilating then eroding ("closing") bridges those small gaps so the document's actual outline becomes traceable as a single contour.
    kernel = np.ones((5, 5), np.uint8)
    closed = cv.dilate(edges, kernel, iterations=2)
    closed = cv.erode(closed, kernel, iterations=1)
 
    # RETR_EXTERNAL: we only care about the outermost boundary of the document, not internal edges (text, logos, etc. inside it).
    contours, _ = cv.findContours(closed, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
 
    # Check the largest contours first — the document should be one of the biggest shapes in the frame, so we don't waste approxPolyDP on noise.
    contours = sorted(contours, key=cv.contourArea, reverse=True)[:10]
 
    largest_quad = None
    largest_area = 0
 
    # Define a minimum area threshold (e.g., 5000 pixels)
    # Alternatively, base it on image size: min_area = (edges.shape[0] * edges.shape[1]) * 0.05
    min_area = 5000  
 
    for contour in contours:
        # Approximate the contour's outline with fewer vertices, tolerating
        # up to 2% of its perimeter worth of deviation. This turns a noisy,
        # many-point contour trace into a clean polygon (ideally 4 points
        # for a rectangular document).
        perimeter = cv.arcLength(contour, True)
        polygon = cv.approxPolyDP(contour, 0.02 * perimeter, True)
 
        if len(polygon) == 4:
            area = cv.contourArea(polygon)
 
            # Check if it passes both the relative size and the absolute size threshold
            if area > largest_area and area > min_area:
                largest_area = area
                largest_quad = polygon
 
    return largest_quad
