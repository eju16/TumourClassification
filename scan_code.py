#updated

import nibabel as nib
from scipy import ndimage
import os
import numpy as np

def read_nifti_file(filepath):
    """Read and load volume"""
    scan = nib.load(filepath)
    scan = scan.get_fdata()
    return scan

def normalize(volume):
    """Normalize the volume"""
    min = -1000
    max = 400
    volume[volume < min] = min
    volume[volume > max] = max
    volume = (volume - min) / (max - min)
    volume = volume.astype("float32")
    return volume

def resize_volume(img):
    """Resize across z-axis"""
    desired_depth = 64
    desired_width = 128
    desired_height = 128
    current_depth = img.shape[-1]
    current_width = img.shape[0]
    current_height = img.shape[1]
    depth_factor = desired_depth / current_depth
    width_factor = desired_width / current_width
    height_factor = desired_height / current_height
    img = ndimage.rotate(img, 90, reshape=False)
    img = ndimage.zoom(img, (width_factor, height_factor, depth_factor), order=1)
    return img

def process_scan(path):
    """Read and resize volume"""
    volume = read_nifti_file(path)
    volume = normalize(volume)
    volume = resize_volume(volume)
    return volume

def get_nifti_file_paths(folder):
    """Recursively find all .nii or .nii.gz files in the given folder"""
    nifti_files = []
    for root, _, files in os.walk(folder):
        for file in files:
            if file.endswith(('.nii', '.nii.gz')):
                nifti_files.append(os.path.join(root, file))
    return nifti_files

#paths for normal and abnormal scans
# normal_scan_paths = get_nifti_file_paths("BraTS-Africa/51_OtherNeoplasms")
# abnormal_scan_paths = get_nifti_file_paths("BraTS-Africa/95_Glioma")

# print("Non-cancerous scans: " + str(len(normal_scan_paths)))
# print("Cancerous scans: " + str(len(abnormal_scan_paths)))

# print("checkpoint")
#-------------------------------------------------------------------
#process scans
# normal_scans = np.array([process_scan(path) for path in normal_scan_paths])
# abnormal_scans = np.array([process_scan(path) for path in abnormal_scan_paths])

# print(f"Processed {len(normal_scans)} normal scans.")
# print(f"Processed {len(abnormal_scans)} abnormal scans.")

scan = read_nifti_file("BraTS-Africa/95_Glioma/BraTS-SSA-00228-000/BraTS-SSA-00228-000-t2f.nii.gz")
print(scan.shape)
#visualise using matplotlib

###building and training validation datasets
# for path in abnormal_scan_paths:
#     print("in looop")
#     print(path)

# print("---------1-------")
# #reading and processing the scans
# abnormal_scans = np.array([process_scan(path) for path in abnormal_scan_paths])
# normal_scans = np.array([process_scan(path) for path in normal_scan_paths])
# print("--------2-------")
# #for ct scans having presence of brain tumours: assign 1
# #for normal ones assign 0
# abnormal_labels = np.array([1 for _ in range(len(abnormal_scans))])
# normal_labels = np.array([0 for _ in range(len(normal_scans))])
# print("-------3---------")


#the below up to 4 is the main code
#splitting the data ratio 70-30 for training and validation
# x_train = np.concatenate((abnormal_scans[:70], normal_scans[:70]), axis=0)
# y_train = np.concatenate((abnormal_labels[:70], normal_labels[:70]), axis=0)

# x_val = np.concatenate((abnormal_scans[70:], normal_scans[70:]), axis=0)
# y_val = np.concatenate((abnormal_labels[70:], normal_labels[70:]), axis=0)
print("------4----------")

#the below is test code for 3^
#testing with only 51_otherneoplasms


# x_train = np.concatenate((normal_scans[:70]), axis = 0)
# y_train = np.concatenate((normal_labels[:70]), axis = 0)

# x_val = np.concatenate((normal_scans[70:]), axis = 0)
# y_val = np.concatenate((normal_labels[70:]), axis=0)
# print("No. samples in train and vaidation are %d and %d" % x_train.shape[0], x_val.shape[0])
# print("--------5--------")

#need to unzip each .nii.gz folder (done)
#split the array/work in smaller batches to avoid long loading times while writin gthe code