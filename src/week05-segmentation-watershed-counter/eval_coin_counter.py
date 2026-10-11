import cv2 as cv
import numpy as np
import os 
import csv
import random
import matplotlib.pyplot as plt
import io
from contextlib import redirect_stdout


from watershed import watershed
from otsu import otsu_threshold

CSV_PATH = "coin-counting-eval/coins_count_values.csv"
IMAGES_ROOT = "coin-counting-eval/coins_images"   # base folder that all_coins/china_coins/... live under
FOLDER_COL = "folder"                # column holding the subfolder name
IMAGE_COL = "image_name"             # column holding the filename
COUNT_COL = "coins_count"            # column holding the true count

def load_ground_truth(csv_path, folder_col, image_col, count_col):
    """
    Load ground truth data from a CSV file.

    Args:
        csv_path (str): Path to the CSV file.
        folder_col (str): Name of the column containing folder names.
        image_col (str): Name of the column containing image names.
        count_col (str): Index of the column containing coin counts.

    Must return a list of (folder, filename, true_count) tuples.
    """
    ground_truth = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            folder = row[folder_col].strip()
            fname = row[image_col].strip()
            count = int(float(row[count_col]))
            ground_truth.append((folder, fname, count))
    return ground_truth


def evaluate(images_root, ground_truth):
    """
    Evaluate the predicted coin counts against the ground truth.

    Args:
    images_root (str): Root folder of all coin images.
    ground_truth (list): List of (folder, filename, true_count) tuples.

    Must return a tuple of the list of (folder, filename, true_count, predicted_count) tuples, the list of errors, and the dictinory of errors per folder.
    """
    results = []
    errors = []
    errors_by_folder = {}

    for folder, filename, true_count in ground_truth:
        image_path = os.path.join(images_root, folder, filename)

        image_bgr = cv.imread(image_path)

        if image_bgr is None:
            print(f"  [skip] Could not read {folder}/{filename}")
            continue

        try:
            predicted_count, _ = watershed(image_bgr)
        except Exception as e:
            print(f"  [error] {folder}/{filename}: {e}")
            continue

        
        results.append((folder, filename, true_count, predicted_count))

        error = abs(predicted_count - true_count)
        errors.append(error)

        errors_by_folder.setdefault(folder, []).append(error)
        
        print(f"  {folder:15s} {filename:25s} true={true_count:3d}  pred={predicted_count:3d}  "
              f"{'OK' if predicted_count == true_count else 'diff=' + str(predicted_count - true_count)}")
 
    return results, errors, errors_by_folder

def report(results, errors, errors_by_folder):
    """
    Calculate and display the exact match accuracy, accuracy within +/-1, mean absolute error, and root mean square error
    
    Args:
    results (list): List of (folder, filename, true_count, predicted_count)
    errors (list): List of errors
    errors_by_folder (dict): A dictionary of {folder, errors} pairs

    Must print the results to the console and save them to a CSV file named "results.csv" in the current working directory.
    """
    n = len(results)
    if n == 0:
        print("No images were successfully evaluated.")
        return
    errors = np.array(errors)
    exact_matches = np.sum(errors == 0)
    within_one = np.sum(errors <= 1)

    exact_acc = 100 * exact_matches / n
    within_one_acc = 100 * within_one / n
    mae = np.mean(errors)
    rmse = np.sqrt(np.mean(errors ** 2))

    print("\n" + "=" * 55)
    print(f"Evaluated on {n} images from Count Coins Image Dataset")
    print("=" * 55)
    print(f"Exact-match accuracy   : {exact_acc:.1f}%  ({exact_matches}/{n})")
    print(f"Accuracy within +/-1   : {within_one_acc:.1f}%  ({within_one}/{n})")
    print(f"Mean Absolute Error    : {mae:.2f} coins")
    print(f"RMSE                   : {rmse:.2f} coins")
 
    print("\nPer-folder breakdown:")
    for folder, errs in sorted(errors_by_folder.items()):
        errs = np.array(errs)
        folder_exact = 100.0 * np.sum(errs == 0) / len(errs)
        print(f"  {folder:15s} n={len(errs):3d}  exact-match={folder_exact:5.1f}%  "
              f"MAE={np.mean(errs):.2f}")
    print("=" * 55)

def display_random_predictions(results, images_root, n=10, save_dir=None,
                                seed=None, exact_matches_only=False):
    """
    Display a random sample of images with their predicted and true coin counts, along with the Otsu binary image and the segmented image.
 
    Args:
    results (list): List of (folder, filename, true_count, predicted_count)
    images_root (str): Root folder of all coin images.
    n (int): Number of random images to display.
    save_dir (str): Directory to save the figures. If None, figures will be displayed
    seed (int): Random seed for reproducibility.
    exact_matches_only (bool): If True, only display images where the predicted count matches the true count exactly.

    Must display the images in a 1x3 grid: original image, Otsu binary image, and segmented image.
    If save_dir is provided, save the figures to that directory instead of displaying them.
    """
 
    candidates = results
    if exact_matches_only:
        candidates = [r for r in results if r[3] == r[2]]  # predicted == true
        if not candidates:
            print("No exact matches found in results.")
            return
 
    if not candidates:
        print("Nothing to display.")
        return
 
    rng = random.Random(seed)
    n = min(n, len(candidates))
    subset = rng.sample(candidates, n)
 
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
 
    for folder, fname, true_count, predicted_count in subset:
        image_path = os.path.join(images_root, folder, fname)
        image_bgr = cv.imread(image_path)
 
        if image_bgr is None:
            print(f"  [skip] unreadable: {image_path}")
            continue
 
        image_rgb = cv.cvtColor(image_bgr, cv.COLOR_BGR2RGB)
        image_gray = cv.cvtColor(image_bgr, cv.COLOR_BGR2GRAY)
 
        # Re-run Otsu + Watershed to regenerate the intermediate images
        # for display (results[] only stores the final count) -- see the
        # same note in display_worst_predictions.
        otsu_binary_image = otsu_threshold(image_gray)[1]
        recomputed_count, segmented_image = watershed(image_bgr)
        segmented_image_rgb = cv.cvtColor(segmented_image, cv.COLOR_BGR2RGB)
 
        diff = recomputed_count - true_count
        match_label = "EXACT MATCH" if diff == 0 else f"diff={diff:+d}"
 
        # Flag loudly if re-running the pipeline now disagrees with what
        # evaluate() recorded earlier -- would mean watershed() isn't
        # deterministic, or results[] is stale relative to the current
        # code. Matters especially here since exact_matches_only trusts
        # the ORIGINAL true_count/predicted_count pairing to pick these.
        if exact_matches_only and diff != 0:
            print(f"  [warning] {folder}/{fname} was recorded as an exact "
                  f"match (true={true_count}) but re-running watershed() "
                  f"now gives pred={recomputed_count} -- mismatch with the "
                  f"original evaluate() run.")
 
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        label = f"{folder}/{fname}"
 
        axes[0].imshow(image_rgb)
        axes[0].axis("off")
        axes[0].set_title(f"{label}\nOriginal (true={true_count})")
 
        axes[1].imshow(otsu_binary_image, cmap="gray")
        axes[1].axis("off")
        axes[1].set_title("Otsu's Binary Image")
 
        axes[2].imshow(segmented_image_rgb)
        axes[2].axis("off")
        axes[2].set_title(f"Segmented (pred={recomputed_count}, {match_label})")
 
        plt.tight_layout()
 
        if save_dir:
            prefix = "exact_match" if exact_matches_only else "random"
            out_name = f"{prefix}_{folder}_{fname}.png"
            plt.savefig(os.path.join(save_dir, out_name))
            plt.close(fig)
        else:
            plt.show()
 
    if save_dir:
        print(f"Saved {len(subset)} figures to {save_dir}/")





def performance_metrics(num_runs=5):
    ground_truth = load_ground_truth(CSV_PATH, FOLDER_COL, IMAGE_COL, COUNT_COL)
    total_images = len(ground_truth)
    
    run_times = []
    print(f"[!] Benchmarking started over {num_runs} runs. Terminal outputs are muted...")

    for run in range(num_runs):
        with redirect_stdout(io.StringIO()):
            start_time = cv.getTickCount()
            
            results, errors, errors_by_folder = evaluate(IMAGES_ROOT, ground_truth)
                           
            end_time = cv.getTickCount()
            
        elapsed_time = (end_time - start_time) / cv.getTickFrequency()
        run_times.append(elapsed_time)
        
    mean_run = np.mean(run_times)
    std_run = np.std(run_times)
    mean_image = mean_run / total_images if total_images > 0 else 0

    print(f"\nPerformance Metrics (over {num_runs} runs):")
    print(f"  Average time per run  : {mean_run:.4f} seconds")
    print(f"  Average time per image: {mean_image:.4f} seconds")
    print(f"  Mean Run Time         : {mean_run:.4f} seconds")
    print(f"  Std Dev               : {std_run:.4f} seconds")
    


def main():
    """
    Load the ground truth data, evaluate the predicted coin counts, and report the results.
    """
    print("\n\n")
    print(f"Loading ground truth from {CSV_PATH} ...")
    start_time = cv.getTickCount()
    ground_truth = load_ground_truth(CSV_PATH, FOLDER_COL, IMAGE_COL, COUNT_COL)

    print(f"Evaluating {len(ground_truth)} images from Count Coins Image Dataset ...")
    results, errors, errors_by_folder = evaluate(IMAGES_ROOT, ground_truth)

    print(f"Reporting results ...")
    report(results, errors, errors_by_folder)
    end_time = cv.getTickCount()
    elapsed_time = (end_time - start_time) / cv.getTickFrequency()
    print(f"Evaluation completed in {elapsed_time:.2f} seconds.")
    average_time_per_image = elapsed_time / len(ground_truth) if len(ground_truth) > 0 else 0
    print(f"Average time per image: {average_time_per_image:.4f} seconds")
    display_random_predictions(
        results, IMAGES_ROOT, n=10, seed=42, exact_matches_only=False, save_dir="random_predictions"
    )

    # Uncomment the following line to run performance metrics evaluation
    # performance_metrics(num_runs=5)    
    
if __name__ == "__main__":
    main()