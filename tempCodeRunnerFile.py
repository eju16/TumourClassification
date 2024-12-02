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