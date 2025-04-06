import os
import open3d as o3d
import numpy as np
import json

# ✅ Ensure we're loading from the correct PCD folder
PCD_FOLDER = "video_processing/useful_point_clouds"

# Define dataset paths
DATASET_FOLDER = "video_processing/dataset"
TRAIN_FOLDER = os.path.join(DATASET_FOLDER, "train")
VAL_FOLDER = os.path.join(DATASET_FOLDER, "val")

# ✅ Get a list of all actual PCD filenames in useful_point_clouds/
actual_pcd_files = {f for f in os.listdir(PCD_FOLDER) if f.endswith(".pcd")}

def load_pcd(file_path):
    """Load and normalize a PCD file."""
    pcd = o3d.io.read_point_cloud(file_path)
    points = np.asarray(pcd.points)

    # Normalize to fit in [-1, 1]
    centroid = np.mean(points, axis=0)
    points -= centroid
    max_dist = np.max(np.linalg.norm(points, axis=1))
    points /= max_dist

    return points

def save_dataset(label_file, output_folder):
    """Convert PCD files into .npy and save corresponding JSON labels."""
    os.makedirs(output_folder, exist_ok=True)

    # Load labels
    with open(label_file, "r") as f:
        labels = json.load(f)

    missing_files = []

    for filename, label in labels.items():
        if filename in actual_pcd_files:  # ✅ Directly check if filename exists in useful_point_clouds
            file_path = os.path.join(PCD_FOLDER, filename)

            # Process point cloud
            points = load_pcd(file_path)

            # Save as .npy
            npy_filename = filename.replace(".pcd", ".npy")
            np.save(os.path.join(output_folder, npy_filename), points)

            # Save label as JSON
            label_path = os.path.join(output_folder, npy_filename.replace(".npy", ".json"))
            with open(label_path, "w") as f:
                json.dump(label, f, indent=4)

        else:
            missing_files.append(filename)

    print(f"✅ Dataset saved in {output_folder}")

    # Report missing PCD files
    if missing_files:
        print("⚠️ WARNING: The following PCD files were listed in JSON but do not exist in useful_point_clouds:")
        for missing in missing_files:
            print(f" - {missing}")

# ✅ Convert both training and validation sets, ensuring only existing files are processed
if __name__ == "__main__":
    save_dataset("train_labels.json", TRAIN_FOLDER)
    save_dataset("val_labels.json", VAL_FOLDER)
