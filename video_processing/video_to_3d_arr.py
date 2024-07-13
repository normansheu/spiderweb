import argparse
from typing import Optional

import cv2
import numpy as np
import pandas as pd

from video_processing.utils import *


class VideoToArray:
    """Convert video into 3d arrays"""

    def __init__(self, video_file: str, dist_file: Optional[str]) -> None:
        self.video = cv2.VideoCapture(video_file)
        if dist_file is not None:
            self.dist = pd.read_csv(dist_file)
        else:
            self.dist = None

    @property
    def num_frames(self):
        return get_num_frames(self.video)

    @property
    def video_width(self):
        return get_width(self.video)

    @property
    def video_height(self):
        return get_height(self.video)

    def video_to_frames(self):
        if self.dist is not None:
            return self.video_to_frames_with_distances()
        
        pass
    def video_to_frames_with_distances(self):
        """Converts video to a series of images with one frame correspond to the distance data.
        """

        # Edge detection
        # Do crop
        pass

    def video_to_array(self):
        """Converts video into 3d numpy array, based on video itself and distance data.
        """

        # 1. 
        pass
