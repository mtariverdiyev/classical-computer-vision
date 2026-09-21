import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os

from gaussian import apply_gaussian_blur
from detection import detect_document_corners
from warp import rectify


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    images_folder = os.path.join(script_dir, 'images')

    filenames = [f for f in os.listdir(images_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    for filename in filenames:
        image_path = os.path.join(images_folder, filename)

        # Read the input image in BGR format
        image_bgr = cv.imread(image_path)

        if image_bgr is None:
            print(f'Warning: Skipping {filename}! OpenCV could not read this file.')
            continue

        # Convert the image to grayscale
        image_gray = cv.cvtColor(image_bgr, cv.COLOR_BGR2GRAY)

        # Apply Gaussian blur to reduce noise
        blurred_gray_image = apply_gaussian_blur(image_gray, kernel_height=5, kernel_width=5, sigma=1.0)

        # Detect document corners in the blurred image
        corners = detect_document_corners(blurred_gray_image)

        if corners is None:
            print(f'Warning: No document corners detected in {filename}.')
            continue

        # Rectify the image based on detected corners
        # TODO: rectify function is expected to be implemented in warp.py
        rectified_image = rectify(image_bgr, corners)

        # Display the original and rectified images side by side
        
        # Show the original and denoised results side by side.
        figure, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(cv.cvtColor(image_bgr, cv.COLOR_BGR2RGB))
        axes[0].set_title(f'Original: {filename}')
        axes[0].axis('off')
        axes[1].imshow(cv.cvtColor(rectified_image, cv.COLOR_BGR2RGB))
        axes[1].set_title(f'Rectified: {filename}')
        axes[1].axis('off')
        figure.tight_layout()
        plt.show()

if __name__ == "__main__":
    main()
