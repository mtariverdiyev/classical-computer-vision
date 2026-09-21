import numpy as np

# build_gaussian_kernel function builds a Gaussian kernel based on the specified height, width, and standard deviation (sigma).
def build_gaussian_kernel(kernel_height, kernel_width, sigma = 1.0):
    offset_y = kernel_height // 2
    offset_x = kernel_width // 2
    kernel = np.zeros((kernel_height, kernel_width), dtype=np.float32)
    for y in range(-offset_y, offset_y + 1):
        for x in range(-offset_x, offset_x + 1):
            kernel[y + offset_y, x + offset_x] = np.exp(-(x**2 + y**2) / (2 * sigma**2))
    kernel /= np.sum(kernel)  # Normalize the kernel
    return kernel


# apply_gaussian_blur function applies Gaussian kernel to the input grayscale image to reduce Gaussain noise.
def apply_gaussian_blur(imageBW, kernel_height=3, kernel_width=3, sigma=1.0):
    offset_y = kernel_height // 2
    offset_x = kernel_width // 2
    kernel = build_gaussian_kernel(kernel_height, kernel_width, sigma)
    imageBW_padded = np.pad(imageBW, (offset_y,offset_y), (offset_x,offset_x), mode='edge')
    denoised_image = np.zeros_like(imageBW.shape, dtype = np.float32)
    for y in range(imageBW.shape[0]):
        for x in range(imageBW.shape[1]):
            # Extract the neighborhood
            neighborhood = imageBW_padded[y:y + kernel_height, x:x + kernel_width]
            # Apply the Gaussian kernel to the neighborhood
            denoised_image[y, x] = np.sum(neighborhood.astype(np.float32) * kernel)
    return np.clip(denoised_image, 0, 255).astype(np.uint8)