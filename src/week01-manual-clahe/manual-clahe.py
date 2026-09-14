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
    cdf = np.cumsum(hist)
    cdf_min = cdf[cdf > 0].min() if np.any(cdf > 0) else 0
    denom = max(cdf[-1] - cdf_min, 1)  # avoid division by zero on empty tiles
 
    lut = (cdf - cdf_min) / denom * 255.0
    lut = np.clip(lut, 0, 255)
    return lut
 

def manual_clahe(image, clip_limit=3.0, tile_grid_size=(8, 8)):
    """
    image          : grayscale image as a 2D numpy array
    clip_limit     : contrast-limiting strength
    tile_grid_size : (tiles_x, tiles_y) -> how many tiles across / down
    """
    tiles_x, tiles_y = tile_grid_size
    rows, cols = image.shape
    # --- Pad the image so it divides evenly into whole tiles -------------
    # We pad by repeating edge pixels ("edge" mode) so the padding doesn't
    # introduce fake dark/bright borders that would distort the histograms.
    padded_rows = int(np.ceil(rows / tiles_y) * tiles_y)
    padded_cols = int(np.ceil(cols / tiles_x) * tiles_x)
    pad_bottom = padded_rows - rows
    pad_right = padded_cols - cols
    padded = np.pad(image, ((0, pad_bottom), (0, pad_right)), mode='edge')
 
    tile_h = padded_rows // tiles_y
    tile_w = padded_cols // tiles_x
 
    # The clip limit is normally expressed relative to an "ideal" flat
    # histogram, so we scale it by the number of pixels per tile.
    clip_limit_pixels = max(clip_limit * tile_h * tile_w / 256.0, 1.0)
 
    # --- Build one LUT (256 values) per tile ------------------------------
    luts = np.zeros((tiles_y, tiles_x, 256), dtype=np.float64)
    for ty in range(tiles_y):
        for tx in range(tiles_x):
            tile = padded[ty * tile_h:(ty + 1) * tile_h,
                          tx * tile_w:(tx + 1) * tile_w]
            hist = clipped_histogram(tile, clip_limit_pixels)
            luts[ty, tx] = histogram_to_lut(hist)
 
    # --- Bilinear interpolation between neighbouring tile LUTs ------------
    # This is what makes CLAHE "adaptive" without creating visible blocky
    # edges at tile boundaries: every pixel blends the mapping of the four
    # tile-centers surrounding it, weighted by distance.
    y_idx, x_idx = np.meshgrid(np.arange(padded_rows), np.arange(padded_cols),
                                indexing='ij')
 
    # Express each pixel's position as a fractional tile coordinate,
    # measured from tile *centers* (hence the -0.5).
    ty_f = (y_idx / tile_h) - 0.5
    tx_f = (x_idx / tile_w) - 0.5
 
    ty0 = np.clip(np.floor(ty_f).astype(int), 0, tiles_y - 1)
    tx0 = np.clip(np.floor(tx_f).astype(int), 0, tiles_x - 1)
    ty1 = np.clip(ty0 + 1, 0, tiles_y - 1)
    tx1 = np.clip(tx0 + 1, 0, tiles_x - 1)
 
    wy = np.clip(ty_f - ty0, 0, 1)
    wx = np.clip(tx_f - tx0, 0, 1)
 
    # Fancy-index every tile's LUT at once using the pixel's own
    # intensity value as the lookup index -> fully vectorized, no python
    # loop over pixels.
    top_left = luts[ty0, tx0, padded]
    top_right = luts[ty0, tx1, padded]
    bottom_left = luts[ty1, tx0, padded]
    bottom_right = luts[ty1, tx1, padded]
 
    top = top_left * (1 - wx) + top_right * wx
    bottom = bottom_left * (1 - wx) + bottom_right * wx
    result = top * (1 - wy) + bottom * wy
 
    # Crop back to the original (unpadded) size and convert to uint8.
    result = result[:rows, :cols]
    return np.clip(result, 0, 255).astype(np.uint8)

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