import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import os

from otsu import otsu_threshold
from watershed import watershed


def main():
    """
    Run the complete coin-segmentation workflow for all images in the coin-images folder.
    Parameters:
    - None.
    Returns:
    - None. The function loads each image, applies Otsu thresholding and Watershed segmentation,
      prints the detected object count, and displays the original, thresholded, and segmented images side by side.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    image_dir = os.path.join(script_dir, "coin-images")

    filenames = [f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]

    for filename in filenames:
        image_path = os.path.join(image_dir, filename)
        
        # Load the image
        image_bgr = cv.imread(image_path)
        image_gray = cv.cvtColor(image_bgr, cv.COLOR_BGR2GRAY)
        image_rgb = cv.cvtColor(image_bgr, cv.COLOR_BGR2RGB)

        # Check if the image was loaded successfully
        if image_bgr is None:
            raise FileNotFoundError(f"Image not found at path: {image_path}")

        # Perform Watershed segmentation and count detected objects
        try:
            otsu_binary_image = otsu_threshold(image_gray)[1]

            count, segmented_image = watershed(image_bgr)

            # Convert BGR to RGB for displaying with matplotlib
            segmented_image_rgb = cv.cvtColor(segmented_image, cv.COLOR_BGR2RGB)

            # Display the original and segmented images side by side
            print(f"The number of detected objects is {count}")
    
            fig, axes = plt.subplots(1, 3, figsize = (15, 5))
            
            axes[0].imshow(image_rgb)
            axes[0].axis("off")
            axes[0].set_title(f"{filename}: Original Image")
            axes[1].imshow(otsu_binary_image)
            axes[1].axis("off")
            axes[1].set_title(f"{filename}: Otsu's Binary Image")
            axes[2].imshow(segmented_image_rgb)
            axes[2].axis("off")
            axes[2].set_title(f"{filename}: Segmented Image")
            plt.show()

        except Exception as e:
            print(f"Error occurred while processing image: {e}")

if __name__ == '__main__':
    main()