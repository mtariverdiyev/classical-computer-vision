from email.mime import image

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import os 

from hough_circles import hough_circles


def main():
    """
    Main function to perform Hough Circle Transform on images in the "circle-detection-images" directory.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    image_dir = os.path.join(script_dir, "circle-detection-images")

    filenames = [f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]

    for filename in filenames:
        image_path = os.path.join(image_dir, filename)
        
        # Load the image
        image_bgr = cv.imread(image_path)

        # Check if the image was loaded successfully
        if image_bgr is None:
            raise FileNotFoundError(f"Image not found at path: {image_path}")

        # Perform Hough Circle Transform and draw detected circles
        try:
            image_with_circles = hough_circles(image_bgr)

            # Convert BGR to RGB for displaying with matplotlib
            image_rgb = cv.cvtColor(image_with_circles, cv.COLOR_BGR2RGB)

            # Display the original and processed images side by side
            fig, axes = plt.subplots(1, 2, figsize = (15, 5))
            axes[0].imshow(cv.cvtColor(image_bgr, cv.COLOR_BGR2RGB))
            axes[0].set_title(f"{filename}: Original Image")       
            axes[0].axis("off")
            axes[1].imshow(cv.cvtColor(image_with_circles, cv.COLOR_BGR2RGB))
            axes[1].set_title(f"{filename}: Detected Circles")
            axes[1].axis("off")
            plt.tight_layout()
            plt.show()
        except Exception as e:
            print(f"Error occurred while processing image: {e}")

if __name__ == "__main__":
    main()

