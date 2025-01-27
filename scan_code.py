#updated

import nibabel as nib
from scipy import ndimage
import os
import numpy as np
import random
#import matplotlib as plt
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib

#still need to reduce run time - no need to process entirety of files

#testing
print(f"Matplotlib version: {matplotlib.__version__}")
print(f"Matplotlib module path: {matplotlib.__file__}")

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

# #debugging matplotlib imshow not working
# # Create a dummy image
# print("runnning dummy")
# image = np.random.rand(128, 128, 64, 1)
# slice_30 = np.squeeze(image[:, :, 30])

# # Visualize the slice
# plt.imshow(slice_30, cmap="gray")
# plt.title("Test Image")
# plt.axis("off")
# plt.show()
# print("end of dummy")
# #paths for normal and abnormal scans

normal_scan_paths = get_nifti_file_paths("BraTS-Africa/51_OtherNeoplasms")
abnormal_scan_paths = get_nifti_file_paths("BraTS-Africa/95_Glioma")

print("Non-cancerous scans: " + str(len(normal_scan_paths)))
print("Cancerous scans: " + str(len(abnormal_scan_paths)))

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

print("---------1-------")
# #reading and processing the scans
abnormal_scans = np.array([process_scan(path) for path in abnormal_scan_paths])
normal_scans = np.array([process_scan(path) for path in normal_scan_paths])
print("--------2-------")
#for ct scans having presence of brain tumours: assign 1
#for normal ones assign 0
abnormal_labels = np.array([1 for _ in range(len(abnormal_scans))])
normal_labels = np.array([0 for _ in range(len(normal_scans))])
print("-------3---------")


#the below up to 4 is the main code
#splitting the data ratio 70-30 for training and validation
x_train = np.concatenate((abnormal_scans[:70], normal_scans[:70]), axis=0)
y_train = np.concatenate((abnormal_labels[:70], normal_labels[:70]), axis=0)

x_val = np.concatenate((abnormal_scans[70:], normal_scans[70:]), axis=0)
y_val = np.concatenate((abnormal_labels[70:], normal_labels[70:]), axis=0)
print("------4----------")

#the below is test code for 3^
#testing with only 51_otherneoplasms


# x_train = np.concatenate((normal_scans[:70]), axis = 0)
# y_train = np.concatenate((normal_labels[:70]), axis = 0)

# x_val = np.concatenate((normal_scans[70:]), axis = 0)
# y_val = np.concatenate((normal_labels[70:]), axis=0)
# print("No. samples in train and vaidation are %d and %d" % x_train.shape[0], x_val.shape[0])
print("No. samples in train and validation are %d and %d" % (x_train.shape[0], x_val.shape[0]))

print("--------5--------")

#need to unzip each .nii.gz folder (done)
#split the array/work in smaller batches to avoid long loading times while writin gthe code

#preprocessing 
##scan visualisation

def rotate(volume):
    """Rotate the volume by a few degrees"""

    def scipy_rotate(volume):
        # define some rotation angles
        angles = [-20, -10, -5, 5, 10, 20]
        # pick angles at random
        angle = random.choice(angles)
        # rotate volume
        volume = ndimage.rotate(volume, angle, reshape=False)
        volume[volume < 0] = 0
        volume[volume > 1] = 1
        return volume

    augmented_volume = tf.numpy_function(scipy_rotate, [volume], tf.float32)
    return augmented_volume


def train_preprocessing(volume, label):
    """Process training data by rotating and adding a channel."""
    # Rotate volume
    volume = rotate(volume)
    volume = tf.expand_dims(volume, axis=3)
    return volume, label


def validation_preprocessing(volume, label):
    """Process validation data by only adding a channel."""
    volume = tf.expand_dims(volume, axis=3)
    return volume, label

# Define data loaders.
train_loader = tf.data.Dataset.from_tensor_slices((x_train, y_train))
validation_loader = tf.data.Dataset.from_tensor_slices((x_val, y_val))

batch_size = 2
# Augment the on the fly during training.
train_dataset = (
    train_loader.shuffle(len(x_train))
    .map(train_preprocessing)
    .batch(batch_size)
    .prefetch(2)
)
# Only rescale.
validation_dataset = (
    validation_loader.shuffle(len(x_val))
    .map(validation_preprocessing)
    .batch(batch_size)
    .prefetch(2)
)

# outputting a single slice
data = train_dataset.take(1)
images, labels = list(data)[0]
images = images.numpy()
image = images[0]
print("Dimension of the CT scan is:", image.shape)
plt.imshow(np.squeeze(image[:, :, 30]), cmap="gray")
plt.show()

# outputting multiple slices at once
def plot_slices(num_rows, num_columns, width, height, data):
    """Plot a montage of 20 CT slices"""
    data = np.rot90(np.array(data))
    data = np.transpose(data)
    data = np.reshape(data, (num_rows, num_columns, width, height))
    rows_data, columns_data = data.shape[0], data.shape[1]
    heights = [slc[0].shape[0] for slc in data]
    widths = [slc.shape[1] for slc in data[0]]
    fig_width = 12.0
    fig_height = fig_width * sum(heights) / sum(widths)
    f, axarr = plt.subplots(
        rows_data,
        columns_data,
        figsize=(fig_width, fig_height),
        gridspec_kw={"height_ratios": heights},
    )
    for i in range(rows_data):
        for j in range(columns_data):
            axarr[i, j].imshow(data[i][j], cmap="gray")
            axarr[i, j].axis("off")
    plt.subplots_adjust(wspace=0, hspace=0, left=0, right=1, bottom=0, top=1)
    plt.show()

#this is not showing anything
# Visualize montage of slices.
# 4 rows and 10 columns for 100 slices of the CT scan.
print("runnign plot slices")
plot_slices(4, 10, 128, 128, image[:, :, :40])
print("end of plot slices")

#ctrl+z to end