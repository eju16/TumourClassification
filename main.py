import os
import zipfile
import numpy as np
import tensorflow as tf  # for data preprocessing
import json

import keras
from keras import layers
import numpy as np
from scipy import ndimage
#scans are in json format

#---------------------
#load scan data from json file
def load_scan_from_json(filepath):
    """Load 3D scan data stored as JSON."""
    with open(filepath, 'r') as f:
        data = json.load(f)
    # Convert data to a numpy array
    volume = np.array(data)
    return volume
#-------------------
#processing scan data

#normalisation and resizing
def normalize(volume):
    """Normalize the volume to be between 0 and 1."""
    min, max = -1000, 400
    volume[volume < min] = min
    volume[volume > max] = max
    volume = (volume - min) / (max - min)
    volume = volume.astype("float32")
    return volume

def resize_volume(img):
    """Resize volume to a desired depth, width, and height."""
    desired_depth, desired_width, desired_height = 64, 128, 128
    current_depth, current_width, current_height = img.shape
    depth_factor = desired_depth / current_depth
    width_factor = desired_width / current_width
    height_factor = desired_height / current_height
    # Rotate and resize
    img = ndimage.rotate(img, 90, reshape=False)
    img = ndimage.zoom(img, (depth_factor, width_factor, height_factor), order=1)
    return img

def process_scan_json(filepath):
    """Load, normalize, and resize a scan stored in JSON format."""
    # Load scan from JSON
    volume = load_scan_from_json(filepath)
    # Normalize
    volume = normalize(volume)
    # Resize
    volume = resize_volume(volume)
    return volume

#----------------------------
#apply processing to eahc json scan files
# Directory containing JSON scan files
json_dir = "path/to/json/scans"
processed_scans = []

# Process each JSON file
for filename in os.listdir(json_dir):
    if filename.endswith(".json"):
        filepath = os.path.join(json_dir, filename)
        processed_volume = process_scan_json(filepath)
        processed_scans.append(processed_volume)

# `processed_scans` now holds all the preprocessed 3D arrays

"""
save each processed scan as a .npy file to access them later without reprocessing
"""
output_dir = "processed_scans"
os.makedirs(output_dir, exist_ok=True)

for i, volume in enumerate(processed_scans):
    output_path = os.path.join(output_dir, f"scan_{i}.npy")
    np.save(output_path, volume)
