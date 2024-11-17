import numpy as np
import os
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
import time

"""
Randomly flips some values of the binary image to simulate noise and recording error.
Returns a new numpy array.
"""
def add_noise(image_array, probability=0.0005):
    random_probabilities = np.random.random(image_array.shape)
    
    flip_mask = random_probabilities < probability

    print(f"Pixels flipped: {np.sum(flip_mask)}")  

    noisy_array = image_array.copy()
    noisy_array[flip_mask] = 1 - image_array[flip_mask]

    return noisy_array


def display_array(array):
    plt.figure(figsize=(12, 8))
    plt.imshow(array, cmap='gray')  # Use grayscale color map
    plt.title('Binary Array as Image')
    plt.axis('off') 
    plt.show()
    plt.close()



os.chdir(os.path.dirname(os.path.realpath(__file__)))
synthetic_images = "./synthetic_web_frames"
image_num = 1
for file in os.listdir(synthetic_images):
    if file.lower().endswith(".png"):
        file_path = os.path.join(synthetic_images, file)
        
        with Image.open(file_path) as image:
            image = image.convert('L')
            image_array = np.array(image)
            noisy_image_array = add_noise(image_array)
            #display_array(noisy_image_array)

            np.save(f"./noisy_frames/frame_{image_num}", noisy_image_array)
            np.save(f"./ideal_frames/frame_{image_num}", image_array)

    image_num += 1
            


             



            