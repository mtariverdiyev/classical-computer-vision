import numpy as np

def pad_image(image, pad_h, pad_w):
    """
    Pad the input image with edge values to handle border effects during filtering.
    The padding is applied to the height and width dimensions, while the channel dimension (if present) is left unchanged. This ensures that the filtering operations can be applied to all pixels, including those
    """
    pad_width = [(pad_h, pad_h), (pad_w, pad_w)]
 
    if image.ndim == 3:
        # Don't pad the channel axis for color images — only height/width.
        pad_width.append((0, 0))
    
    return np.pad(image, pad_width, mode = 'edge')

# build_gaussian_kernel function builds a Gaussian kernel based on the specified height, width, and standard deviation (sigma).
def build_gaussian_kernel(kernel_height, kernel_width, sigma = 1.0):
    offset_y = kernel_height // 2
    offset_x = kernel_width // 2
    kernel = np.zeros((kernel_height, kernel_width), dtype=np.float64)
    for y in range(-offset_y, offset_y + 1):
        for x in range(-offset_x, offset_x + 1):
            kernel[y + offset_y, x + offset_x] = np.exp(-(x**2 + y**2) / (2 * sigma**2))
    kernel /= np.sum(kernel)  # Normalize the kernel
    return kernel


# apply_gaussian_blur function applies Gaussian kernel to the input grayscale image to reduce Gaussain noise.
def apply_gaussian_blur(image_bgr, kernel_height=3, kernel_width=3, sigma=1.0):
    offset_y = kernel_height // 2
    offset_x = kernel_width // 2

    kernel = build_gaussian_kernel(kernel_height, kernel_width, sigma)

    # Pad the image to handle border effects during filtering
    image_padded = pad_image(image_bgr, offset_y, offset_x)

    denoised_image = np.zeros_like(image_bgr, dtype=np.float64)
    # Slide the kernel over every pixel position and every color channel,
    # computing a weighted sum of the local neighborhood.
    for y in range(image_bgr.shape[0]):
        for x in range(image_bgr.shape[1]):
            region = image_padded[y:y + kernel_height, x:x + kernel_width, :]
            for ch in range(image_bgr.shape[2]):
                denoised_image[y, x, ch] = np.sum(region[:, :, ch].astype(np.float64) * kernel)

    # Convert back to a valid 8-bit image: round to the nearest integer and
    # clip into the [0, 255] range (the weighted sum could drift slightly
    # outside that range due to floating point rounding).
    return np.clip(np.round(denoised_image), 0, 255).astype(np.uint8)