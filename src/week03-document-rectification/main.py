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


        # Apply Gaussian blur to reduce noise
        blurred_image = apply_gaussian_blur(image_bgr, kernel_height=5, kernel_width=5, sigma=1.0)

        # Detect document corners in the blurred image
        corners = detect_document_corners(blurred_image)

        # Create a copy of the original image to draw detected corners on
        detected_corners = image_bgr.copy()

        if corners is None:
            print(f'Warning: No document corners detected in {filename}.')
            # No document found in this image — show a small black placeholder
            # instead of crashing, so the rest of the folder still processes.
            rectified_image = np.zeros((100, 100, 3), dtype = np.uint8)

        else:
            # Outline the detected corners in green for visual sanity-checking
            cv.drawContours(detected_corners, [corners], -1, (0, 255, 0), 3)
            # Warp the detected region into a straightened, top-down crop
            rectified_image = rectify(image_bgr, corners)


        # Display the original and rectified images side by side
        # Show the original and denoised results side by side.
        # Side-by-side comparison: original photo, detected outline, rectified
        # result. OpenCV loads/stores images as BGR, but matplotlib expects
        # RGB, hence the cv.cvtColor conversions before each imshow.
        fig, axes = plt.subplots(1, 3, figsize = (15, 5))
    
        axes[0].imshow(cv.cvtColor(image_bgr, cv.COLOR_BGR2RGB))
        axes[0].set_title(f"{filename}: Original Image")
        axes[0].axis("off")
    
        axes[1].imshow(cv.cvtColor(detected_corners, cv.COLOR_BGR2RGB))
        axes[1].set_title(f"{filename}: Detected Corners")
        axes[1].axis("off")
    
        axes[2].imshow(cv.cvtColor(rectified_image, cv.COLOR_BGR2RGB))
        axes[2].set_title(f"{filename}: Rectified Image")
        axes[2].axis("off")
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    main()
