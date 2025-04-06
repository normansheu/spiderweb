from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import List, Optional, Tuple
import os
import pandas as pd
import skimage.io as skio

import cv2
import numpy as np
import open3d as o3d
import skimage as sk
from tqdm import tqdm
from sklearn.linear_model import LinearRegression
from scipy.interpolate import interp1d

import matplotlib.pyplot as plt
import json
import boto3
from dataclasses import dataclass

# axes arrow points towards 0 time
cut_front_frames = 0
cut_back_frames = 0

@dataclass(frozen=True)
class BOX_FOUR_INCH:
    left_border = 530
    right_border = 1380
    top_border = 0
    bottom_border = 560

    box_depth = 850

@dataclass(frozen=True)
class BOX_THREE_INCH:
    left_border = 750
    right_border = 1380
    top_border = 0
    bottom_border = 440

    box_depth = 630


pixel_threshold = 0.55

px_per_mm = 4.86


s3_client = boto3.client('s3')
bucket_name = 'scanned-objects'
crop_file = "video_processing/crop_data/@013 255 2024-10-05 03-18-53 crop.json" # IGNORE


def camera_speed_factor(distance_data: pd.DataFrame):
    X = distance_data[['Time']].values
    y = distance_data['Distance'].values

    model = LinearRegression()
    model.fit(X, y)

    return model.coef_[0]

def process_frame_grey(frame_data, prev_roll, show_brightness=False, box=BOX_FOUR_INCH):
    frame, frame_count, m = frame_data
    if show_brightness:
        return
    original_frame = frame
    print(f"Processing frame {frame_count}")

    timestamp = frame_count / 60
    distance = m * timestamp * 1000 * px_per_mm

    blue_channel, green_channel, red_channel = cv2.split(frame)

    green_channel = cv2.GaussianBlur(green_channel, (5, 5), 1)

    min_red_blue = np.minimum(red_channel, blue_channel)

    blurred_image = cv2.GaussianBlur(min_red_blue, (5, 5), 1)

    green_normalized = green_channel.astype(float) / 255
    min_red_blue_normalized = blurred_image.astype(float) / 255

    greyscale_combined = (green_normalized + 2 * min_red_blue_normalized) / 3

    grayscale_frame = (greyscale_combined * 255).astype(np.uint8)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    contrast_enhanced = clahe.apply(grayscale_frame)

    max_pixel_value = np.max(contrast_enhanced)
    threshold_value = max_pixel_value * pixel_threshold
    _, binary_image = cv2.threshold(contrast_enhanced, threshold_value, 255, cv2.THRESH_BINARY)

    imagelen = len(frame)
    while True:
        if imagelen <= 0:
            imagelen = prev_roll
            break
        last_row = binary_image[imagelen-1]
        if np.sum(last_row == 255) / len(last_row) > 0.10:
            break
        binary_image[0] = 0
        imagelen -= 1
    binary_image = np.roll(binary_image, len(frame) - imagelen, axis=0)
    binary_image = binary_image[::-1, :]
    binary_image = binary_image[20:, :]
    back_boundary = False
    binary_image = binary_image[box.top_border:box.bottom_border, box.left_border:box.right_border]
    if binary_image.sum() / binary_image.size > 0.98:
        back_boundary = True
    ys, xs = np.where(binary_image == 255)
    points = [(x, y, -distance) for x, y in zip(xs, ys)]
    return points, imagelen, (back_boundary, -distance)

def create_and_visualize_point_cloud(video_path: str, dst_dir: Optional[str], distance_data, show_brightness=False,
                                     upload_s3=False, box=BOX_FOUR_INCH):
    m = camera_speed_factor(distance_data)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    all_points = []
    distances = []
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_count = 0
    ignore_first_frames = cut_front_frames
    ignore_last_frames = cut_back_frames
    if show_brightness:
        with open(crop_file, "r") as f:
            crop_bounds = json.load(f)
        x_min = crop_bounds["x_min"]
        y_min = crop_bounds["y_min"]
        x_max = crop_bounds["x_max"]
        y_max = crop_bounds["y_max"]
        while True:
            ret, frame = cap.read()
            
            if not ret:
                break
            
            if frame_count < ignore_first_frames or frame_count >= total_frames - ignore_last_frames:
                frame_count += 1
                continue 
            ymin = y_min
            ymax = y_max
            cropped_frame = frame[-ymax:-ymin, x_min:x_max, :]
            try:
                while np.mean(cropped_frame[-1]) < 60:
                    ymin += 1
                    ymax += 1
                    cropped_frame = frame[-ymax:-ymin, x_min:x_max, :]
            except Exception:
                cropped_frame = frame[-y_max:-y_min, x_min:x_max, :]

            all_points.append(sk.exposure.rescale_intensity(np.mean(cropped_frame, axis=2), in_range='image', out_range=(0, 256)).astype(np.uint8))
            print(frame_count)
            frame_count += 1
    
    else:
        back_boundaries = []
        prev_roll = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count < ignore_first_frames or frame_count >= total_frames - ignore_last_frames:
                frame_count += 1
                continue  

            points, new_roll, boundary = process_frame_grey((frame, frame_count, m), prev_roll, box=box)
            if boundary[0]:
                back_boundaries.append(boundary[1])
            if new_roll != 0:
                prev_roll = new_roll

            all_points.extend(points)
            frame_count += 1
        print(back_boundaries)
    if show_brightness:
        timestamps = np.arange(frame_count) / 60
        distances = m * timestamps * 1000 * px_per_mm
        z_min, z_max = distances.min(), distances.max()
        z_resolution = int(z_max - z_min)
        print(all_points[0].shape)
        frame_height, frame_width = all_points[0].shape
        box = np.zeros((z_resolution, frame_height, frame_width))
        z_positions = np.linspace(z_min, z_max, z_resolution)
        for y in tqdm(range(frame_height)):
            for x in range(frame_width):
                pixel_values = [frame[y, x] for frame in all_points]
                interp_func = interp1d(distances, pixel_values, bounds_error=False, fill_value=0)
                box[:, y, x] = interp_func(z_positions)
        
        brightness_array = np.transpose(box, (1, 2, 0))
        print(brightness_array.shape)
        brightness_array = brightness_array[:, :, -crop_bounds["z_max"]:-crop_bounds["z_min"]]
        print(brightness_array.shape)
        print("box created")
        video_name = Path(video_path).stem
        if dst_dir is None: 
            file_name = str(Path(video_path).parent / f"{video_name} brightness.npy")
        else:
            dst_dir = Path(dst_dir)
            dst_dir.mkdir(exist_ok=True)
            file_name = str(dst_dir / f"{video_name} brightness.npy")
        np.save(file_name, brightness_array)
        return

    cap.release()

    if all_points:
        if len(back_boundaries) == 0:
            back_boundaries.append(min([point[2] for point in all_points]))
        z_boundary_min = min(back_boundaries) + 150
        z_boundary_max = z_boundary_min + box.box_depth
        all_points = [point for point in all_points if z_boundary_min <= point[2] < z_boundary_max]

        points_np = np.array(all_points)
        if points_np.ndim == 2 and points_np.shape[1] == 3:
            print(f"Number of points: {points_np.shape[0]}")
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points_np)
            pcd = normalize_pcd(pcd)
            video_name = Path(video_path).stem
            if dst_dir is None: 
                file_name = str(Path(video_path).parent / f"{video_name}.pcd")
            else:
                dst_dir = Path(dst_dir)
                dst_dir.mkdir(exist_ok=True)
                file_name = str(dst_dir / f"{video_name}.pcd")
            o3d.io.write_point_cloud(file_name, pcd)
            print(f"Saved point cloud to {dst_dir}.")
            if upload_s3:
                try:
                    object_name = file_name.split("/")[-1]
                    s3_client.upload_file(file_name, bucket_name, object_name)
                    print(f"Uploaded {object_name} to {bucket_name}...")
                except Exception as e:
                    print(e)
                    print("Unable to upload...")
            axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1000.0, origin=[0, 0, 0])
            o3d.visualization.draw_geometries([axes, pcd])
        else:
            print("Error: Points array is not in the expected N by 3 shape.")
    else:
        print("No points were added to the point cloud. Check the frame processing logic.")

def voxelize(pcd, voxel_size=1):
    points = np.asarray(pcd.points)
    min_bound = points.min(axis=0)
    max_bound = points.max(axis=0)

    grid_shape = ((max_bound - min_bound) / voxel_size).astype(int) + 1
    voxel_grid = np.zeros(grid_shape, dtype=int)
    voxel_indices = ((points - min_bound) / voxel_size).astype(int)
    voxel_grid[voxel_indices[:, 0], voxel_indices[:, 1], voxel_indices[:, 2]] = 1
    return voxel_grid

def voxel_to_pcd(voxel_grid: np.ndarray):
    """Convert 3D voxel grid to point cloud object. ASSUMES VOXEL SIZE OF 1.

    Args:
        voxel_grid (np.ndarray): 3D voxel grid.

    Returns:
        o3d.geometry.PointCloud: pcd object.
    """
    voxel_indices = np.argwhere(voxel_grid == 1)
    points = voxel_indices.astype(float)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    return pcd

def normalize_pcd(pcd):
    points = np.asarray(pcd.points)
    min_bound = points.min(axis=0)
    points_normalized = points - min_bound
    pcd_normalized = o3d.geometry.PointCloud()
    pcd_normalized.points = o3d.utility.Vector3dVector(points_normalized)
    return pcd_normalized





if __name__ == '__main__':
    distance_data = pd.read_csv("video_processing/distance_records/sparse3 255 distance data 2024-11-30 11-29-33.csv")
    video_path = "video_processing/spider_videos/sparse3 255 2024-11-30 11-29-33.mp4"
    create_and_visualize_point_cloud(video_path=os.path.expanduser(video_path),
                                    dst_dir=os.path.expanduser("video_processing/point_clouds"), distance_data=distance_data,
                                    show_brightness=False,
                                    upload_s3=False,
                                    box=BOX_FOUR_INCH)
    


    