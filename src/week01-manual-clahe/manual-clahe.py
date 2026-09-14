import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os

# ---------------------------------------------------------------------------
#   Build a histogram for a single tile, clip it, and redistribute
#         the clipped-off pixel counts uniformly across all bins.
# ---------------------------------------------------------------------------
def clipped_histogram(tile, clip_limit):
    """
    tile        : a 2D numpy array (a small block cut out of the image)
    clip_limit  : the maximum number of pixels any single histogram bin
                  is allowed to hold. Any pixels above this limit are
                  "clipped" and spread back out evenly over every bin.
                  This is what stops CLAHE from massively over-amplifying
                  noise in near-flat regions (the classic AHE problem).
    """
    # Build a 256-bin histogram of pixel intensities (0-255) for this tile.
    # np.bincount is a fast way to count occurrences of each integer value.
    hist = np.bincount(tile.flatten(), minlength=256).astype(np.float64)
 
    # Anything above clip_limit gets removed from that bin...
    excess = np.sum(np.maximum(hist - clip_limit, 0))
    hist = np.minimum(hist, clip_limit)
 
    # ...and redistributed equally across all 256 bins so no pixels
    # are lost (total histogram count must stay the same).
    hist += excess / 256.0
 
    return hist


# ---------------------------------------------------------------------------
#   Turn a (clipped) histogram into a pixel-value mapping function
#         (a lookup table, LUT) by using its cumulative distribution.
# ---------------------------------------------------------------------------
def histogram_to_lut(hist):
    """
    Standard histogram equalization idea: the mapping for intensity value
    'v' is the cumulative sum of the histogram up to 'v', rescaled so it
    spans the full 0-255 output range.
    """
    pass  # TODO: Implement this function to return a LUT based on the input histogram.
 

def manual_clahe(image, clip_limit=3.0, tile_grid_size=(8, 8)):
    """
    image          : grayscale image as a 2D numpy array
    clip_limit     : contrast-limiting strength
    tile_grid_size : (tiles_x, tiles_y) -> how many tiles across / down
    """

    pass  # TODO: Implement the manual CLAHE algorithm here, using the clipped_histogram function.


# ---------------------------------------------------------------------------
#   Load every image in the input folder and display original vs
#         manual-CLAHE side by side.
# ---------------------------------------------------------------------------
def main():
    scriptDir = os.path.dirname(os.path.abspath(__file__))
    inputFolder = os.path.join(scriptDir, 'low-light-images')
 
    filenames = [f for f in os.listdir(inputFolder)]
 
    for filename in filenames:
        imagePath = os.path.join(inputFolder, filename)
 
        # Read image in grayscale.
        imageBW = cv.imread(imagePath, 0)
 
        if imageBW is None:
            print(f'Warning: Skipping {filename}! OpenCV could not read this file.')
            continue
 
        claheResult = manual_clahe(imageBW, clip_limit=3.0, tile_grid_size=(8, 8))
 
        # Show original and CLAHE result next to each other.
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(imageBW, cmap='gray', vmin=0, vmax=255)
        axes[0].set_title(f'Original: {filename}')
        axes[0].axis('off')
 
        axes[1].imshow(claheResult, cmap='gray', vmin=0, vmax=255)
        axes[1].set_title('Manual CLAHE')
        axes[1].axis('off')
 
        plt.tight_layout()
        plt.show()

if __name__ == '__main__':
    main()