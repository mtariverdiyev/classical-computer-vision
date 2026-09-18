import os 
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np


# pad_image function pads the input image with edge values to handle border effects during filtering. It takes the image and the padding sizes for y and x directions as inputs and returns the padded image.
def pad_image(image, pad_y, pad_x):
    return np.pad(image, ((pad_y, pad_y), (pad_x, pad_x)), mode='edge')


# classify_noise function determines the type of noise present in the image based on the proportion of extreme pixel values (0 and 255). If the proportion exceeds a specified threshold, it classifies the noise as 'salt_and_pepper'; otherwise, it classifies it as 'gaussian'.
def classify_noise(image, threshold=0.02):
    salt_and_pepper_noise = np.sum(image == 255) + np.sum(image == 0)
    if salt_and_pepper_noise / image.size > threshold:
        return 'salt_and_pepper'
    else:
        return 'gaussian'
    
def apply_median_blur():
    # TODO: Implement median blur filtering to reduce Salt-and-Pepper noise while preserving edges.
    pass

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