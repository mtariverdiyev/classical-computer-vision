# Classical Computer Vision — Weeks 1–5

-This repo covers Weeks 1-5 of the Computer Vision & Object
-Detection. Each week is a self-contained mini-project. Each week is a self-contained mini-project built on plain NumPy + OpenCV, moving from manual pixel-level algorithm
implementations (Week 1–2) toward using OpenCV's built-in primitives as
building blocks for larger pipelines (Week 3–5).

| Week | Topic | Assignment | Core Files |
|---|---|---|---|
| 1 | Color/Histogram Math | Manual Histogram Optimizer (CLAHE) | `manual-clahe.py` |
| 2 | Spatial Filtering | Noise Suppression Pipeline | `image-denoiser.py`, `gaussian.py` |
| 3 | Geometric Warping | Document Rectification | `main.py`, `detection.py`, `gaussian.py`, `warp.py` |
| 4 | Edge/Hough Transforms | Industrial Feature Extractor | `main.py`, `hough_circles.py` |
| 5 | Segmentation | Overlapping Object Counter | `main.py`, `otsu.py`, `watershed.py` |

## Suggested Folder Layout

Because Weeks 3, 4, and 5 each ship their own `main.py`, keep every week in
its own folder so the scripts and their expected image folders don't
collide:

```
week1_histogram_optimizer/
    manual-clahe.py
    low-light-images/        <- put input .jpg/.png files here

week2_noise_suppression/
    image-denoiser.py
    noisy-images/

week3_document_rectification/
    main.py
    detection.py
    gaussian.py
    warp.py
    images/

week4_feature_extractor/
    main.py
    hough_circles.py
    circle-detection-images/

week5_object_counter/
    main.py
    otsu.py
    watershed.py
    coin-images/
```

## Requirements

```
pip install opencv-python numpy matplotlib
```

All scripts are run directly with `python <script>.py` from inside their
week's folder (they locate their input folder relative to `__file__`, so
the working directory doesn't matter as much as the script's own location).
Each script loops over every file in its input folder, processes it, and
pops up a `matplotlib` window comparing original vs. result. Close a
window to move on to the next image.

---

## Week 1 — Manual Histogram Optimizer (CLAHE)

**File:** `manual-clahe.py`

### What it does
Implements Contrast Limited Adaptive Histogram Equalization (CLAHE) from
scratch using only NumPy — no `cv.createCLAHE()`. CLAHE fixes uneven,
low-light exposure by equalizing contrast *locally* (per image tile)
instead of globally, which brings out detail in both dark and bright
regions of the same image without blowing out highlights.

### How it works
1. **`clipped_histogram`** — builds a 256-bin histogram for one tile,
   clips any bin above `clip_limit_pixels`, and redistributes the
   clipped-off pixel count evenly across all bins. This is the "contrast
   limited" part — it stops flat regions (e.g. sky, wall) from getting
   wildly over-amplified, which is the classic failure mode of plain
   Adaptive Histogram Equalization.
2. **`histogram_to_lut`** — turns that clipped histogram into a 256-value
   lookup table via its cumulative distribution function (standard
   histogram-equalization math), rescaled to span the full `0–255` range.
3. **`manual_clahe`** —
   - Pads the image (edge-replication) so it divides evenly into an
     `8×8` grid of tiles by default.
   - Computes one LUT per tile.
   - **Bilinearly interpolates between the 4 nearest tile LUTs** for every
     pixel, using its distance from each tile center as blend weight.
     This is what removes visible blocky seams at tile boundaries and is
     the part that makes CLAHE "adaptive" rather than just "tiled
     equalization."
   - Crops back to the original size and returns a `uint8` image.

### How to run
1. Create a `low-light-images/` folder next to `manual-clahe.py`.
2. Drop in grayscale-convertible images (any `.jpg`/`.png`).
3. Run:
   ```bash
   python manual-clahe.py
   ```
4. For each image, a side-by-side window opens: original vs. manual CLAHE
   result (`clip_limit=3.0`, `tile_grid_size=(8, 8)` by default — tune
   these inside `main()`).

---

## Week 2 — Noise Suppression Pipeline

**Files:** `image-denoiser.py` (self-contained version used by the
`main()` in this file), plus `gaussian.py` (a cleaner, reusable Gaussian
blur module used later in Week 3).

### What it does
Automatically detects *which kind* of noise is present in an image —
salt-and-pepper vs. Gaussian — and applies whichever filter is
mathematically the better fit, rather than always using one filter type.

### How it works
1. **`classify_noise`** — counts how many pixels sit at the extreme
   values (`0` or `255`). Salt-and-pepper noise slams pixels to these
   extremes, so if more than 2% of pixels are pure black/white, the image
   is classified `salt_and_pepper`; otherwise `gaussian`.
2. **`apply_median_blur`** (for salt-and-pepper) — a hand-rolled median
   filter: pads the image, slides a `k×k` window over every pixel, and
   replaces it with the neighborhood median. Median filters are
   non-linear and excellent at salt-and-pepper because they ignore
   outlier extreme values rather than averaging them in.
3. **`build_gaussian_kernel` / `apply_gaussian_blur`** (for Gaussian
   noise) — builds a normalized 2D Gaussian kernel by hand and convolves
   it with the image via a padded sliding window. Gaussian blur is the
   linear, mathematically-optimal filter for zero-mean Gaussian noise.
4. **`pipeline`** — orchestrates the above: converts to grayscale only to
   *classify* the noise, then applies the chosen filter independently to
   each of the B/G/R channels of the original color image and re-merges
   them.

### `gaussian.py` (used again in Week 3)
A tidier, standalone version of the Gaussian blur logic
(`pad_image`, `build_gaussian_kernel`, `apply_gaussian_blur`) that works
directly on multi-channel BGR images in one pass (loops over channels
internally) rather than needing the caller to split channels first.

### How to run
1. Create a `noisy-images/` folder next to `image-denoiser.py`.
2. Run:
   ```bash
   python image-denoiser.py
   ```
3. Console output tells you which noise type was detected per file; a
   matplotlib window shows original vs. denoised result.

> **Note:** the manual median/Gaussian loops are pure Python `for y / for
> x` pixel loops, so they're intentionally educational rather than fast —
> expect them to be noticeably slower than `cv.medianBlur` /
> `cv.GaussianBlur` on large images.

---

## Week 3 — Document Rectification

**Files:** `main.py`, `detection.py`, `gaussian.py`, `warp.py`

### What it does
Given a photo of a document (ID card, license plate, etc.) taken at an
angle, this pipeline finds its 4 corners and warps it into a flat,
top-down rectangle — like a mini document scanner.

### How it works (pipeline order, per `main.py`)
1. **Denoise** — `apply_gaussian_blur` (from `gaussian.py`) smooths the
   image slightly so Canny edge detection isn't overwhelmed by texture
   noise.
2. **`detect_document_corners`** (`detection.py`):
   - Runs `cv.Canny` to get a binary edge map.
   - **Closes** small gaps in the edges (`dilate` then `erode`) so the
     document's outline becomes one continuous, traceable contour instead
     of broken fragments.
   - Finds external contours, keeps the 10 largest by area (cheap
     pre-filter), and for each one approximates it to a simplified
     polygon with `cv.approxPolyDP`.
   - Keeps the largest polygon that has **exactly 4 vertices** and passes
     a minimum-area threshold — that's the document boundary.
3. **`rectify`** (`warp.py`) — the geometric core:
   - **`order_points`** — sorts the 4 unordered corners into
     top-left/top-right/bottom-right/bottom-left using the
     `x+y` (min/max) and `y-x` (min/max) trick.
   - Computes output width/height from the actual max side lengths of the
     detected quad (so perspective distortion doesn't squash the result).
   - **`compute_homography`** — solves the 8×8 linear system by hand
     (via `np.linalg.inv`) for the homography matrix mapping the skewed
     source corners to a flat destination rectangle.
   - **`inverse_warp`** — for every *output* pixel, maps it back into the
     *source* image using `H⁻¹`, then samples the source at that
     (generally non-integer) location with **`linear_interpolate`**
     (manual bilinear interpolation) to avoid aliasing/blockiness.
4. `main.py` visualizes 3 panels per image: original, detected corners
   outlined in green, and the final rectified crop.

### How to run
1. Create an `images/` folder next to `main.py` with photos of documents.
2. Run:
   ```bash
   python main.py
   ```
3. If no 4-point contour is found for an image, it's skipped gracefully
   (a small black placeholder is shown instead of crashing the batch).

---

## Week 4 — Industrial Feature Extractor (Hough Circles)

**Files:** `main.py`, `hough_circles.py`

### What it does
Detects circular mechanical parts (e.g. bolts, washers, gears) in a
top-down factory-line image using the Hough Circle Transform, draws each
detected circle with its center, and labels it with its measured
diameter in pixels.

### How it works
1. **`detect_circles`** — grayscales and median-blurs the image (removes
   small speckle noise that would otherwise create false circle votes),
   then calls `cv.HoughCircles` with:
   - `minDist` scaled to image height (`/30`) so genuinely close, distinct
     parts aren't merged into one detection.
   - `param1=200` (the internal Canny high threshold used by the Hough
     accumulator) and `param2=22` (accumulator vote threshold — lower
     means more, possibly noisier, detections).
   - `minRadius=5, maxRadius=30` to constrain the expected part size.
2. **`draw_circles`** — for each detected `(x, y, radius)`, draws the
   circle outline (red), a filled center dot (green), and overlays a
   `D = <diameter>` text label. Raises a clear error if nothing was
   found rather than silently returning an unmodified image.
3. **`hough_circles`** ties the two together and is the single function
   `main.py` calls per image.

### How to run
1. Create a `circle-detection-images/` folder next to `main.py`.
2. Run:
   ```bash
   python main.py
   ```
3. Each image pops up as a 2-panel comparison: original vs. detected
   circles with diameters labeled. Per-image errors (e.g. "no circles
   detected") are caught and printed without stopping the batch.

> **Tuning tip:** `param2` is the most sensitive knob — lower it to catch
> faint/low-contrast circles (at the risk of false positives), raise it
> for stricter, cleaner detections.

---

## Week 5 — Overlapping Object Counter (Otsu + Watershed)

**Files:** `main.py`, `otsu.py`, `watershed.py`

### What it does
Counts objects (e.g. coins) in an image even when several of them are
touching or overlapping — a case where a naive threshold + connected
components would merge touching objects into a single blob and
undercount.

### How it works
1. **`otsu_threshold`** (`otsu.py`) — implements Otsu's method by hand:
   - Builds a normalized 256-bin histogram (probability per intensity).
   - Computes cumulative pixel-fraction (`omega`) and cumulative mean
     (`mu`) arrays.
   - For every possible threshold `t`, computes the **between-class
     variance** in closed form; the threshold that *maximizes* it best
     separates foreground from background.
   - Returns both the optimal threshold and the resulting binary image.
2. **`watershed`** (`watershed.py`) — the segmentation core:
   - Blurs and Otsu-thresholds the grayscale image to get a clean binary
     mask.
   - **Morphological opening then closing** (4 iterations each) removes
     small noise specks and fills small holes in the mask.
   - **Sure background** = the opened/closed mask dilated outward (safely
     background).
   - **Sure foreground** = pixels far from any boundary, found via
     `cv.distanceTransform` thresholded at 50% of its max — i.e. the
     "core" of each object, even if two objects are touching at their
     edges.
   - **Unknown region** = background minus foreground — the ambiguous
     boundary zone between touching objects, which is exactly what
     watershed needs to resolve.
   - Labels each sure-foreground blob with `cv.connectedComponents`,
     marks the unknown region as `0`, then runs `cv.watershed`, which
     "floods" outward from each labeled seed and draws a boundary
     (`-1`) wherever two floods meet — this is what correctly separates
     touching/overlapping objects.
   - Colors each resulting region randomly and paints watershed
     boundaries red for visualization; returns the object count (`ret -
     1`, since one label is the background).
3. `main.py` runs Otsu thresholding and Watershed independently per
   image, prints the detected count, and shows all 3 stages side by side.

### How to run
1. Create a `coin-images/` folder next to `main.py`.
2. Run:
   ```bash
   python main.py
   ```
3. For each image, the console prints the detected object count, and a
   3-panel window shows: original, Otsu binary mask, and the final
   colored/segmented Watershed result.

### Quantitative evaluation (baseline)

To get an actual accuracy number rather than eyeballing a handful of
images, `watershed()` was benchmarked against the
[Kaggle "Count Coins Image Dataset"](https://www.kaggle.com/datasets/balabaskar/count-coins-image-dataset)
(1,444 labeled images across 7 currency folders, each with a
ground-truth coin count) using a separate evaluation harness,
`eval_coin_counter.py`. See that script's docstring for setup
instructions.

**Baseline results — the `watershed.py` implementation described above,
unmodified:**

| Metric | Value |
|---|---|
| Exact-match accuracy | 13.7% (198/1444) |
| Accuracy within ±1 | 31.2% (451/1444) |
| Mean Absolute Error (MAE) | 3.83 coins |
| RMSE | 6.84 coins |

| Folder | n | Exact-match | MAE |
|---|---|---|---|
| all_coins | 215 | 14.4% | 4.86 |
| china_coins | 210 | 8.6% | 3.52 |
| euro_coins | 231 | 11.3% | 4.59 |
| indian_coins | 220 | 15.9% | 4.89 |
| peso_coins | 184 | 11.4% | 2.91 |
| us_coins | 173 | 9.8% | 3.06 |
| yen_coins | 211 | 23.7% | 2.61 |

**Known inaccuracies in this implementation**, found by inspecting the
worst-predicted images rather than inferring from the aggregate numbers
alone:

1. **Otsu polarity is never checked.** `otsu_threshold()` returns `255`
   for "brighter than the cutoff" and `0` for everything else — it has
   no notion of which class is actually the coin. `watershed()` silently
   assumes `255 == coin` throughout (the distance transform, sure
   background/foreground logic all depend on it). This assumption breaks
   on any image where the background is brighter than the coins — e.g. a
   dark coin photographed on a white catalog page or product-listing
   background. On those images the pipeline effectively measures
   distance *into the background*, not the coins, producing wildly wrong
   counts (observed: a 96-coin grid predicted `2`).
2. **The foreground threshold is a single global cutoff, which breaks on
   mixed-scale images.** `sure_foreground` is computed by thresholding
   the distance transform at a flat `0.5 * distance_transform.max()`.
   This value is set by whichever connected blob is *largest in the
   whole image*. When a photo mixes one large cluster of touching coins
   with smaller or separate coins elsewhere in the same frame, those
   smaller coins' own distance values never clear 50% of the big
   cluster's max — they're never seeded as their own object, and get
   merged into "unknown"/absorbed into their neighbor instead of counted
   separately. This systematically undercounts images with uneven
   coin-cluster sizes.
3. **These two bugs partially mask each other in the aggregate metric
   above.** Bug 2's tendency to only ever seed the single most dominant
   blob in an image has a side effect: it also suppresses fine surface
   detail (engraved lettering, portraits) from being mistaken for
   separate objects, since only one region can ever be "the max." That
   means this baseline's MAE of 3.83 is, to some extent, an accident of
   two independent bugs' effects partially offsetting one another rather
   than evidence the underlying logic is sound — fixing either bug in
   isolation does not reliably improve overall accuracy, since it can
   remove a failure mode while simultaneously removing a different
   failure mode's accidental suppression.
4. **Morphological kernel size and the pre-threshold blur are both fixed
   pixel constants** (`(3,3)` kernel, `(5,5)` blur) regardless of image
   resolution. This dataset spans a large range, from small catalog
   thumbnails to macro close-ups where a single coin fills most of the
   frame — a fixed pixel-size constant can at best be correctly tuned
   for one end of that range.

---

## Notes for Reviewers

- Weeks 1 and 2 are deliberately implemented with raw NumPy pixel/tile
  loops (no `cv.createCLAHE`, no `cv.GaussianBlur`/`cv.medianBlur` in the
  core logic) per the assignment brief — this is intentional and expected
  to be slower than OpenCV's compiled equivalents.
- Weeks 3–5 build on top of OpenCV's primitives (`Canny`, `HoughCircles`,
  `watershed`, `distanceTransform`, etc.) but still hand-implement the
  parts the assignments specifically target (homography solving +
  bilinear warping in Week 3, Otsu's variance maximization in Week 5).
- Every script's `main()` assumes an input images folder sitting next to
  it (see **Suggested Folder Layout** above) and will print a warning and
  skip any file OpenCV fails to read, rather than crashing the batch.