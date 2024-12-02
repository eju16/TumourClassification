import nibabel as nib

from scipy import ndimage
import os
import numpy as np

def read_nifti_file(filepath):
    """Read and load volume"""
    # if not filepath.endswith(('.nii', '.nii.gz')):
    #     filepath += '.nii'
    #     # scan = nib.load(filepath)
    #read file
    scan = nib.load(filepath)
    #get raw data
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
    #set the desired depth
    desired_depth = 64
    desired_width = 128
    desired_height = 128
    #get current depth
    current_depth = img.shape[-1]
    current_width = img.shape[0]
    current_height = img.shape[1]
    #compute depth factor
    depth = current_depth / desired_depth
    width = current_width / desired_width
    height = current_height / desired_height
    depth_factor = 1 / depth
    width_factor = 1 / width
    height_factor = 1 / height
    #rotate the image
    img = ndimage.rotate(img, 90, reshape=False)
    #resize across z-axis
    img = ndimage.zoom(img, (width_factor, height_factor, depth_factor), order=1)
    return img


def process_scan(path):
    """Read and resize volume"""
    #read scan
    volume = read_nifti_file(path)
    #normalize
    volume = normalize(volume)
    # Resize width, height and depth
    volume = resize_volume(volume)
    return volume

#folder "51-OtherNeoplasms" consist of non cancerous scans,
normal_scan_paths = [
    os.path.join(os.getcwd(), "BraTS-Africa/51_OtherNeoplasms", x)
    for x in os.listdir("BraTS-Africa/51_OtherNeoplasms")
]
#folder "95_Glioma" consist of scans with glioma (cancerous)
abnormal_scan_paths = [
    os.path.join(os.getcwd(), "BraTS-Africa/95_Glioma", x)
    for x in os.listdir("BraTS-Africa/95_Glioma")
]

print("Non cancerous scans: " + str(len(normal_scan_paths)))
print("Cancerous scans: " + str(len(abnormal_scan_paths)))

#-----------------------------------------
###building and training validation datasets
for path in abnormal_scan_paths:
    print("in looop")
    print(path)

print("---------1-------")
#reading and processing the scans
abnormal_scans = np.array([process_scan(path) for path in abnormal_scan_paths])
normal_scans = np.array([process_scan(path) for path in normal_scan_paths])
print("--------2-------")
#for ct scans having presence of brain tumours: assign 1
#for normal ones assign 0
abnormal_labels = np.array([1 for _ in range(len(abnormal_scans))])
normal_labels = np.array([0 for _ in range(len(normal_scans))])
print("-------3---------")
#splitting the data ratio 70-30 for training and validation
# x_train = np.concatenate((abnormal_scans[:70], normal_scans[:70]), axis=0)
# y_train = np.concatenate((abnormal_labels[:70], normal_labels[:70]), axis=0)

# x_val = np.concatenate((abnormal_scans[70:], normal_scans[70:]), axis=0)
# y_val = np.concatenate((abnormal_labels[70:], normal_labels[70:]), axis=0)
print("------4----------")
#testing with only 51_otherneoplasms
x_train = np.concatenate((normal_scans[:70]), axis = 0)
y_train = np.concatenate((normal_labels[:70]), axis = 0)

x_val = np.concatenate((normal_scans[70:]), axis = 0)
y_val = np.concatenate((normal_labels[70:]), axis=0)
print("No. samples in train and vaidation are %d and %d" % x_train.shape[0], x_val.shape[0])
print("--------5--------")
#need to unzip each .nii.gz folder