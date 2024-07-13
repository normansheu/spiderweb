"""Utility functions handling videos"""
from typing import Tuple

import cv2
import numpy as np
from skimage.morphology import skeletonize


def get_fps(video: cv2.VideoCapture):
    """Frame per second"""
    return int(video.get(cv2.CAP_PROP_FPS))


def get_num_frames(video: cv2.VideoCapture) -> int:
    """Total number of frames"""
    return int(video.get(cv2.CAP_PROP_FRAME_COUNT))


def get_width(video: cv2.VideoCapture) -> int:
    """Width"""
    return int(video.get(cv2.CAP_PROP_FRAME_WIDTH))


def get_height(video: cv2.VideoCapture) -> int:
    """Height"""
    return int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))


def crop_and_resize(img: np.ndarray, x_min: float, x_max: float, y_min: float, y_max: float) -> np.ndarray:
    # crop and resize an image
    orig_h, orig_w = img.shape[0], img.shape[1]
    crop_img = img[y_min: y_max, x_min: x_max]
    new_img = cv2.resize(crop_img, (orig_w, orig_h))
    return new_img


def detect_edge(img: np.ndarray) -> Tuple[float, float, float, float]:
    """Detect edge. Take horizontal/vertical sum of (green) pixel values and find the positions with largest/smallest values.

    Args:
        img (np.ndarray): image in numpy array. Values represent pixel values, from 0 to 255 (BRG).

    Returns:
        Tuple[float, float, float, float]: x_min, x_max, y_min, y_max
    """
    height, width, _ = img.shape
    N = 10
    adjust_h = int(height * 0.02)
    adjust_v = int(width * 0.02)

    # horizontal lines
    h_thres = 0.0
    h_sum = np.sum(img, axis=1)[:, 2]
    sorted_h_ind = np.argsort(h_sum)
    large_h_ind = sorted_h_ind[-N:]
    sorted_large_h_ind = np.sort(large_h_ind)
    
    if sorted_large_h_ind[0] < 0.5 * height and h_sum[sorted_large_h_ind[0]] > h_thres:
        y_min = sorted_large_h_ind[0] + adjust_h
    else:
        y_min = 0
    if sorted_large_h_ind[-1] > 0.5 * height and h_sum[sorted_large_h_ind[-1]] > h_thres:
        y_max = sorted_large_h_ind[-1] - adjust_h
    else:
        y_max = height - 1

    # vertical lines
    v_thres = 0.0
    v_sum = np.sum(img, axis=0)[:, 2]
    sorted_v_ind = np.argsort(v_sum)
    large_v_ind = sorted_v_ind[-N:]
    sorted_large_v_ind = np.sort(large_v_ind)

    if sorted_large_v_ind[0] < 0.5 * width and v_sum[sorted_large_v_ind[0]] > v_thres:
        x_min = sorted_large_v_ind[0] + adjust_v
    else:
        x_min = 0
    if sorted_large_v_ind[-1] > 0.5 * width and v_sum[sorted_large_v_ind[-1]] > v_thres:
        x_max = sorted_large_v_ind[-1] - adjust_v
    else:
        x_max = width - 1
    
    return x_min, x_max, y_min, y_max


def sharpen_edges(img: np.ndarray) -> np.ndarray:
    """Sharpening edges by applying Canny edge detection and skeletonize.

    Returns:
        np.ndarray: New image with sharpened edges. Black and white.
    """
    img[0] = 0
    img[1] = 0
    avg_val = img.sum() / (img.shape[0] * img.shape[1])
    img = cv2.Canny(img, avg_val * 0.9, 255)
    img = skeletonize(img, method="lee")
    return img
