import numpy as np

def order_points(pts):
    """
    Sort 4 arbitrary (unordered) corner points into a consistent
    [top-left, top-right, bottom-right, bottom-left] order.
    """
    rect = np.zeros((4, 2), dtype=np.float32)
    # x + y is minimized for the top-left cornet and maximized for the bottom-right corner
    sum = np.sum(pts, axis=1)
    rect[0] = pts[np.argmin(sum)]  # top-left
    rect[2] = pts[np.argmax(sum)]  # bottom-right

    # y - x is maximized for the bottom-left corner and minimized for the top-right corner
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # top-right
    rect[3] = pts[np.argmax(diff)]  # bottom-left

    return rect

def compute_destination_points(image):
    """
    Compute the destination points for the rectified image based on the input image size.
    The destination points are the corners of a rectangle that will be used to warp the input image
    """
    image_size = image.shape
    height, width = image_size[0], image_size[1]
    # Define the destination points for the rectified image
    dst_pts = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype=np.float32)
    return dst_pts

def compute_homography(src_pts, dst_pts):
    # TODO: Implement a method to compute the homography matrix that maps the source points to the destination points.
    pass

def linear_interpolate(image, x, y):
    # TODO: Implement a method to perform linear interpolation on the output image at the specified (x, y) coordinates to find the corresponding pixel value in the input image.
    pass

def inverse_warp(image, H, output_size):
    # TODO: Implement a method to perform an inverse perspective transform on the output image using the inverse of a provided homography matrix.
    pass

def rectify(image, quad_pts):
    # TODO: Implement a method to rectify the input image based on the found quadrilateral corner points. This method should compute the destination points, homography matrix, perform the inverse warp, and return the rectified image.
    pass


