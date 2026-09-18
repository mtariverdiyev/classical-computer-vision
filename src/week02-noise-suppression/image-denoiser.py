import os 
import cv2 as cv
from matplotlib import image
import matplotlib.pyplot as plt
import numpy as np


# pad_image function pads the input image with edge values to handle border effects during filtering. It takes the image and the padding sizes for y and x directions as inputs and returns the padded image.
def pad_image(imageBW, pad_y, pad_x):
    return np.pad(imageBW, ((pad_y, pad_y), (pad_x, pad_x)), mode='edge')


# classify_noise function determines the type of noise present in the image based on the proportion of extreme pixel values (0 and 255). If the proportion exceeds a specified threshold, it classifies the noise as 'salt_and_pepper'; otherwise, it classifies it as 'gaussian'.
def classify_noise(imageBW, threshold=0.02):
    salt_and_pepper_noise = np.sum(imageBW == 255) + np.sum(imageBW == 0)
    if salt_and_pepper_noise / imageBW.size > threshold:
        return 'salt_and_pepper'
    else:
        return 'gaussian'

# apply_median_blur function applies a median filter to the input grayscale image to reduce salt-and-pepper noise. It takes the image and the kernel size as inputs, pads the image to handle borders, and computes the median of the neighborhood for each pixel, returning the denoised image.
def apply_median_blur(imageBW, kernel_height=3, kernel_width=3):
    offset_y = kernel_height // 2
    offset_x = kernel_width // 2
    imageBW_padded = pad_image(imageBW, offset_y, offset_x)
    denoised_image = np.zeros_like(imageBW)
    imageBW_height, imageBW_width = imageBW.shape
    for y in range(imageBW_height):
        for x in range(imageBW_width):
            # Extract the neighborhood
            neighborhood = imageBW_padded[y:y + kernel_height, x:x + kernel_width]
            # Compute the median and assign it to the denoised image
            denoised_image[y, x] = np.median(neighborhood)
    return denoised_image

# build_gaussian_kernel function creates a Gaussian kernel based on the specified height, width, and standard deviation (sigma). It computes the Gaussian function for each position in the kernel and normalizes it so that the sum of all elements equals 1. This kernel is used for Gaussian blurring to reduce noise while preserving image quality.
def build_gaussian_kernel(kernel_height, kernel_width, sigma):
    offset_y = kernel_height // 2
    offset_x = kernel_width // 2
    kernel = np.zeros((kernel_height, kernel_width), dtype=np.float32)
    for y in range(-offset_y, offset_y + 1):
        for x in range(-offset_x, offset_x + 1):
            kernel[y + offset_y, x + offset_x] = np.exp(-(x**2 + y**2) / (2 * sigma**2))
    kernel /= np.sum(kernel)  # Normalize the kernel
    return kernel

#apply_gaussian_blur function applies a Gaussian filter to the input grayscale image to reduce Gaussian noise. It pads the image to handle borders, extracts neighborhoods for each pixel, and applies the Gaussian kernel to compute the weighted sum, returning the denoised image.
def apply_gaussian_blur(imageBW, kernel_height=3, kernel_width=3, sigma=1.0):
    offset_y = kernel_height // 2
    offset_x = kernel_width // 2
    imageBW_padded = pad_image(imageBW, offset_y, offset_x)
    gaussian_kernel = build_gaussian_kernel(kernel_height, kernel_width, sigma)
    imageBW_height, imageBW_width = imageBW.shape
    denoised_image = np.zeros_like(imageBW, dtype=np.float32)
    for y in range(imageBW_height):
        for x in range(imageBW_width):
            # Extract the neighborhood
            neighborhood = imageBW_padded[y:y + kernel_height, x:x + kernel_width]
            # Apply the Gaussian kernel to the neighborhood
            denoised_image[y, x] = np.sum(neighborhood.astype(np.float32) * gaussian_kernel)
    return np.clip(denoised_image, 0, 255).astype(np.uint8)

# pipeline function orchestrates the denoising process for a given image. It first converts the image to grayscale and classifies the noise type. Depending on whether the noise is classified as 'salt_and_pepper' or 'gaussian', it applies the appropriate filtering method (median or Gaussian blur) to each color channel (B, G, R) of the image. Finally, it merges the denoised channels back together and returns the noise type along with the denoised image.
def pipeline(image, filename=None):
    
    # Convert the input image to grayscale for noise classification
    imageBW = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    noise_type = classify_noise(imageBW)
    name, ext = os.path.splitext(filename) if filename else ("unknown", "")
    print(f"Identified noise type for {name} is {noise_type}")

    b, g, r = cv.split(image)

    if noise_type == "salt_and_pepper":
        print("Executing Optimal Filter -> Non-linear Median Filter...")
        b = apply_median_blur(b)
        g = apply_median_blur(g)
        r = apply_median_blur(r)
    else:
        print("Executing Optimal Filter -> Linear Gaussian Blur Filter...")
        b = apply_gaussian_blur(b, sigma = 1.5)
        g = apply_gaussian_blur(g, sigma = 1.5)
        r = apply_gaussian_blur(r, sigma = 1.5)

    denoised_image = cv.merge((b, g, r))

    return (noise_type, denoised_image)


# main function to run the image denoising pipeline and display results.
def main():
    scriptDir = os.path.dirname(os.path.abspath(__file__))
    inputFolder = os.path.join(scriptDir, 'noisy-images')

    filenames = [f for f in os.listdir(inputFolder)]

    for filename in filenames:
        imagePath = os.path.join(inputFolder, filename)

        # Read image in bgr.
        imageBGR = cv.imread(imagePath)

        if imageBGR is None:
            print(f'Warning: Skipping {filename}! OpenCV could not read this file.')
            continue

        imageRGB = cv.cvtColor(imageBGR, cv.COLOR_BGR2RGB)
        noise_type, image_denoised_BGR = pipeline(imageBGR, filename=filename)
        image_denoised_RGB = cv.cvtColor(image_denoised_BGR, cv.COLOR_BGR2RGB)

        # Show the original and denoised results side by side.
        figure, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(imageRGB)
        axes[0].set_title(f'Original: {filename}')
        axes[0].axis('off')
        axes[1].imshow(image_denoised_RGB)
        axes[1].set_title(f'{filename} - {noise_type} - denoised')
        axes[1].axis('off')
        figure.tight_layout()
        plt.show()

if __name__ == '__main__':
    main()