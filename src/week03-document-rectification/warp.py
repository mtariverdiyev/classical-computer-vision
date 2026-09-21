import numpy as np
import math

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

def compute_destination_points(width, height):
    """
    Compute the destination points for the rectified image based on the input image size.
    The destination points are the corners of a rectangle that will be used to warp the input image
    """
    # Define the destination points for the rectified image
    dst_pts = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype=np.float64)
    return dst_pts

def compute_homography(src_pts, dst_pts):
    """
    Compute the homography matrix that maps the source points to the destination points.
    """
    # Build 8x8 matrix A and 8x1 vector b, one (x_s, y_s) -> (x_d, y_d)
    # correspondence contributing 2 rows each.
    A = []
    b = []
    for i in range(4):
        x_s, y_s = src_pts[i, 0], src_pts[i, 1]
        x_d, y_d = dst_pts[i, 0], dst_pts[i, 1]
        A.append([x_s, y_s, 1, 0, 0, 0, -x_s * x_d, -y_s * x_d])
        A.append([0, 0, 0, x_s, y_s, 1, -x_s * y_d, -y_s * y_d])
 
        b.append(x_d)
        b.append(y_d)
 
    A = np.array(A)
    b = np.array(b)
 
    # Solve the 8x8 linear system for the 8 unknown homography entries
    # (h11..h32); h33 is fixed to 1 since H is only defined up to scale.
    A_inv = np.linalg.inv(A)
 
    h = A_inv @ b
    h = np.append(h, 1)  # h33 = 1 (homography defined up to scale)
 
    H = h.reshape(3, 3)
 
    return H

def linear_interpolate(image, x, y):
    """
    Perform bilinear interpolation to sample the pixel value at non-integer coordinates (x, y) in the image.
    This is used during inverse warping to compute the pixel value at a source location that may not align exactly with the pixel grid.
    """
    # 1. Find the 4 neighboring pixel coordinates
    x0 = math.floor(x)
    y0 = math.floor(y)
    x1 = min(x0 + 1, image.shape[1] - 1)  # Prevent out-of-bounds
    y1 = min(y0 + 1, image.shape[0] - 1)
 
    # 2. Extract fractional offsets (alpha, beta) — how far (x, y) sits
    #    between the integer pixel grid points, used as blend weights below.
    alpha = x - x0
    beta = y - y0
 
    # 3. Retrieve actual pixel values from the image array
    # Format note: NumPy indices are image[y, x]
    p00 = image[y0, x0]  # Top-Left
    p10 = image[y0, x1]  # Top-Right
    p01 = image[y1, x0]  # Bottom-Left
    p11 = image[y1, x1]  # Bottom-Right
 
    # 4. Elementwise bilinear blend (works for grayscale or multi-channel
    #    pixels, since alpha/beta are scalars and p00..p11 are broadcast
    #    over however many color channels the image has).
    top = (1 - alpha) * p00 + alpha * p10       # blend along x at y0
    bottom = (1 - alpha) * p01 + alpha * p11    # blend along x at y1
    pixel = (1 - beta) * top + beta * bottom    # blend along y
 
    return pixel

def inverse_warp(image, H, output_size):
    """
    Perform inverse warping of the input image using the provided homography matrix H.
    The output image will have the specified output_size (width, height).
    This function maps each pixel in the output image back to the corresponding pixel in the input image
    """

    output = np.zeros((output_size[1], output_size[0], 3), dtype=np.uint8)
 
    # We know H maps src (original photo) -> dst (rectified rectangle).
    # To go the other direction (output pixel -> source pixel), we need
    # its inverse.
    H_inv = np.linalg.inv(H)
 
    for v in range(output_size[1]):
        for u in range(output_size[0]):
 
            # Homogeneous destination coordinate for this output pixel
            dst = np.array([u, v, 1])
 
            # Map back into the original (unrectified) image's coordinate
            # space
            src = H_inv @ dst
 
            # Convert from homogeneous coordinates back to normal (x, y)
            # by dividing through by the homogeneous scale component
            x = src[0] / src[2]
            y = src[1] / src[2]
 
            # Only sample if the source coordinate actually falls inside
            # the original image; the "- 1" keeps room for the neighbor
            # pixel linear_interpolate needs at x+1/y+1.
            if 0 <= x < image.shape[1] - 1 and 0 <= y < image.shape[0] - 1:
                output[v, u] = linear_interpolate(image, x, y)
 
    return output

def euc_dist(pt1, pt2):
    """
    Compute the Euclidean distance between two points pt1 and pt2.
    Each point is expected to be a 2D coordinate (x, y).
    """
    return np.sqrt((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2)

def rectify(image, quad_pts):
    """
    Top-level entry point: given the original `image` and a detected
    4-point `quad` (the skewed document boundary), produce a straightened,
    top-down crop of just that document.
    """

    # Detected corners typically arrive as an (4, 1, 2) array from
    # cv2.approxPolyDP; flatten to (4, 2) for easier math.
    src_pts = quad_pts.reshape(4, 2)
 
    # Normalize corner order so it lines up consistently with compute_dst's
    # [top-left, top-right, bottom-right, bottom-left] convention.
    src_pts = order_points(src_pts)
 
    # Estimate the rectified output size from the actual side lengths of
    # the quad. Each side (e.g. top vs bottom) may differ slightly due to
    # perspective distortion, so we take the max of each pair to avoid
    # cropping/squashing the larger side.
    w_1 = euc_dist(src_pts[1], src_pts[0])  # top edge
    w_2 = euc_dist(src_pts[2], src_pts[3])  # bottom edge
 
    w = int(max(w_1, w_2))
 
    h_1 = euc_dist(src_pts[1], src_pts[2])  # right edge
    h_2 = euc_dist(src_pts[0], src_pts[3])  # left edge
 
    h = int(max(h_1, h_2))
 
    # Target rectangle corners for the straightened output
    dst_pts = compute_destination_points(w, h)
 
    # Homography mapping original (skewed) corners -> straightened corners
    H = compute_homography(src_pts, dst_pts)
 
    # Use inverse warping + bilinear sampling to render the final rectified
    # image
    return inverse_warp(image, H, (w, h))


