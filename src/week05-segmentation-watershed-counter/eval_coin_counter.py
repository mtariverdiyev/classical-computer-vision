import cv2 as cv
import numpy as np
import os 
import csv

from watershed import watershed

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
    TODO: Calculate and display the exact match accuracy, accuracy within +/-1, mean absolute error, and root mean square error
    
    Args:
    results (list): List of (folder, filename, true_count, predicted_count)
    errors (list): List of errors
    errors_by_folder (dict): A dictionary of {folder, errors} pairs

    Must print the results to the console and save them to a CSV file named "results.csv" in the current working directory.
    """
def main():
    """
    Load the ground truth data, evaluate the predicted coin counts, and report the results.
    """
    print(f"Loading ground truth from {CSV_PATH} ...")
    ground_truth = load_ground_truth(CSV_PATH, FOLDER_COL, IMAGE_COL, COUNT_COL)
    
    results, errors, errors_by_folder = evaluate(IMAGES_ROOT, ground_truth)


if __name__ == "__main__":
    main()