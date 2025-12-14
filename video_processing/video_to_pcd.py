# File: generate_point_cloud.py

from pathlib import Path
import os
import pandas as pd
import cv2
import numpy as np
import open3d as o3d
from tqdm import tqdm
from sklearn.linear_model import LinearRegression

from web_crop import interactive_crop

# --- Configuration ---
px_per_mm = 3.4
# NEW: Set the minimum threshold to start collecting points from.
# Raising this value will decrease memory usage by collecting fewer, more significant points.
# A value of 0.1 is too low for most systems. Try 0.3 or 0.4 to start.
MINIMUM_THRESHOLD = 0.15

def camera_speed_factor(distance_data: pd.DataFrame):
    """Calculates camera speed via linear regression on distance over time."""
    X = distance_data[['time']].values
    y = distance_data['distance'].values
    model = LinearRegression()
    model.fit(X, y)
    return model.coef_[0]

def process_frame_grey(frame_data):
    """
    Processes a single video frame to find points and the highest threshold they pass.
    Now uses a configurable minimum threshold to control point volume.
    """
    frame, frame_count, m = frame_data
    # The thresholds now start from the configurable minimum
    thresholds = np.round(np.arange(MINIMUM_THRESHOLD, 0.95, 0.05), 2)
    if thresholds.size == 0:
        # This can happen if MINIMUM_THRESHOLD is >= 0.95
        return np.empty((0, 4))

    timestamp = frame_count / 60
    distance = m * timestamp * 1000 * px_per_mm

    # --- Image Processing Pipeline (unchanged) ---
    # blue_channel, green_channel, red_channel = cv2.split(frame)
    # green_channel = cv2.GaussianBlur(green_channel, (5, 5), 1)
    # min_red_blue = np.minimum(red_channel, blue_channel)
    # blurred_image = cv2.GaussianBlur(min_red_blue, (5, 5), 1)
    # green_normalized = green_channel.astype(float) / 255
    # min_red_blue_normalized = blurred_image.astype(float) / 255
    # greyscale_combined = (green_normalized + 2 * min_red_blue_normalized) / 3
    # greyscale_combined = (green_channel + red_channel + blue_channel) / 3
    greyscale_combined = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # grayscale_frame = (greyscale_combined * 255).astype(np.uint8)
    # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    # contrast_enhanced = clahe.apply(greyscale_combined)
    contrast_enhanced = greyscale_combined

    # max_pixel_value = np.max(contrast_enhanced)
    max_pixel_value = 255
    if max_pixel_value == 0:
        return np.empty((0, 4))

    # --- Thresholding Logic ---
    normalized_values = contrast_enhanced.astype(float) / max_pixel_value
    
    # We now check against the first value in our (potentially higher) thresholds array
    min_t_for_frame = thresholds[0]
    ys, xs = np.where(normalized_values >= min_t_for_frame)
    
    if xs.size == 0:
        return np.empty((0, 4))

    candidate_values = normalized_values[ys, xs]
    indices = np.searchsorted(thresholds, candidate_values, side='right') - 1
    
    # Filter out any points that might not have passed any threshold in the new range
    valid_mask = indices >= 0
    ys, xs, indices = ys[valid_mask], xs[valid_mask], indices[valid_mask]
    
    max_passed_thresholds = thresholds[indices]
    z_coords = np.full_like(xs, -distance, dtype=np.float32)
    
    # Use memory-efficient data types
    points_with_thresholds = np.stack((
        xs.astype(np.uint16),
        ys.astype(np.uint16),
        z_coords,
        max_passed_thresholds.astype(np.float16)
    ), axis=-1)
    
    return points_with_thresholds

# The rest of the file (create_and_launch_cropper and if __name__ == '__main__') remains unchanged.
def create_and_launch_cropper(video_path, dst_dir, distance_data):
    """
    Creates a point cloud from a video, normalizes it, and launches the interactive cropper.
    """
    m = camera_speed_factor(distance_data)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    all_points_list = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    with tqdm(total=total_frames, desc="Processing Frames") as pbar:
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            points_array = process_frame_grey((frame, frame_count, m))
            if points_array.shape[0] > 0:
                all_points_list.append(points_array)
            
            frame_count += 1
            pbar.update()

    cap.release()

    if all_points_list:
        print("Aggregating point data...")
        all_points_data = np.vstack(all_points_list)
        print(f"Total points generated across all thresholds: {all_points_data.shape[0]}")
        
        xyz_points = all_points_data[:, :3].astype(np.float32)
        min_bound = xyz_points.min(axis=0)
        xyz_points -= min_bound
        all_points_data[:, :3] = xyz_points

        video_name = Path(video_path).stem
        if dst_dir is None:
            file_path_template = str(Path(video_path).parent / video_name)
        else:
            dst_dir = Path(dst_dir)
            dst_dir.mkdir(exist_ok=True)
            file_path_template = str(dst_dir / video_name)

        interactive_crop(all_points_data, file_path_template, crop_step=5)
    else:
        print("No points were generated from the video. Check frame processing logic.")

if __name__ == '__main__':
    distance_data_path = "/Users/grantyang/Downloads/tangle014 255 distance data 2025-11-12 19-26-44.csv"
    video_path = "/Users/grantyang/Downloads/tangle016r 255 2025-11-15 05-55-05.mp4"
    
    distance_data = pd.read_csv(distance_data_path)
    
    create_and_launch_cropper(
        video_path=os.path.expanduser(video_path),
        dst_dir=os.path.expanduser("video_processing/box_point_clouds"),
        distance_data=distance_data
    )