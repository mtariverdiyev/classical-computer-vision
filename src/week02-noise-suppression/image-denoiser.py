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

def apply_gaussian_blur():
    # TODO: Implement Gaussian blur filtering to reduce Gaussian noise while maintaining image quality.
    pass

def pipeline(image):
    # TODO: Implement the main pipeline logic to orchestrate the noise suppression process.
    pass

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
        noise_type = classify_noise(imageBGR)
        image_denoised_BGR = pipeline(imageBGR)
        image_denoised_RGB = cv.cvtColor(image_denoised_BGR, cv.COLOR_BGR2RGB)

        # Show the original and denoised results sequentially.
        plt.figure(figsize=(5, 5))
        plt.imshow(imageRGB)
        plt.title(f'Original: {filename}')
        plt.axis('off')
        plt.show()

        plt.figure(figsize=(5, 5))
        plt.imshow(image_denoised_RGB)
        plt.title(f'{filename} - {noise_type} - denoised')
        plt.axis('off')
        plt.show()

if __name__ == '__main__':
    main()