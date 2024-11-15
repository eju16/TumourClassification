import os
import zipfile
import numpy as np
import tensorflow as tf  # for data preprocessing
#import json

import keras
from keras import layers
import numpy as np
from scipy import ndimage

#------
#for jpg files

import os
import numpy as np
from PIL import Image

def load_scan_from_jpgs(folder_path):
    """Load a set of JPEG images from a folder and stack them into a 3D volume."""
    #sort the files to ensure slices are loaded in the correct order
    slice_files = sorted([f for f in os.listdir(folder_path) if f.endswith(".jpg")])
    
    #load each slice and stack them
    slices = []
    for filename in slice_files:
        img = Image.open(os.path.join(folder_path, filename)).convert("L")  #convert to grayscale, already in grayscale, but to ensure
        img_array = np.array(img)
        slices.append(img_array)
    
    #stack all slices along a new axis to create a 3D volume
    volume = np.stack(slices, axis=-1)
    print(volume)
    return volume

from scipy import ndimage

def normalize(volume):
    """Normalize the volume to a range between 0 and 1 based on Hounsfield units (HU)."""
    min, max = -1000, 400
    volume[volume < min] = min
    volume[volume > max] = max
    volume = (volume - min) / (max - min)
    volume = volume.astype("float32")
    return volume

def resize_volume(img):
    """Resize volume to a specified depth, width, and height."""
    desired_depth, desired_width, desired_height = 64, 128, 128
    current_depth, current_width, current_height = img.shape
    depth_factor = desired_depth / current_depth
    width_factor = desired_width / current_width
    height_factor = desired_height / current_height
    #rotate and resize
    img = ndimage.rotate(img, 90, reshape=False)
    img = ndimage.zoom(img, (depth_factor, width_factor, height_factor), order=1)
    return img

def process_scan(folder_path):
    """Load JPEG slices, stack to form a 3D volume, normalize, and resize."""
    #load and stack JPEG slices
    volume = load_scan_from_jpgs(folder_path)
    #normalize
    volume = normalize(volume)
    #resize
    volume = resize_volume(volume)
    return volume
    

#directory containing folders of JPEG slices for each scan
scans_dir = "/Users/lj/TumourClassification/TumourClassification/BraTS-Africa/95_Glioma"
processed_scans = []



#process each folder in the scans directory
for folder_name in os.listdir(scans_dir):
    #print("folder name",folder_name)
    if folder_name == ".DS_Store":
        pass
    else:
        folder_path = os.path.join(scans_dir, folder_name)
        #print("folder name",folder_name)
        print("folder path",folder_path)
        if os.path.isdir(folder_path):  # Only process directories
            print("if statement")
            processed_volume = process_scan(folder_path)
            print("processed volume: ",processed_volume)
            processed_scans.append(processed_volume)

print("test")
#now processed_scans holds the preprocessed 3D arrays for each scan

#saving processed volumes to be used later, saved as .npy files
output_dir = "processed_scans"
os.makedirs(output_dir, exist_ok=True)

for i, volume in enumerate(processed_scans):
    output_path = os.path.join(output_dir, f"scan_{i}.npy")
    np.save(output_path, volume)

#print("test")