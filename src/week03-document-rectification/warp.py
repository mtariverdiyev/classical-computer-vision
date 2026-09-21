def order_points(pts):
    # TODO: Implement a method to order the four points in the input array such that they are in the order of top-left, top-right, bottom-right, and bottom-left.
    pass

def compute_destination_points(image_size):
    # TODO: Implement a method to compute the destination points for the perspective transform based on the size of the input image.
    pass

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


