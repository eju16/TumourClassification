#run this to download kaggle dataset onto machine
import os
from keras import layers
import zipfile
from kaggle.api.kaggle_api_extended import KaggleApi

#initalising kaggle api
api = KaggleApi()
api.authenticate()

#define dataset details
dataset_name = "masoudnickparvar/brain-tumor-mri-dataset"
download_dir = os.path.join(os.getcwd(), "BrainTumorMRI")
os.makedirs(download_dir, exist_ok=True)

#downloading the dataset
api.dataset_download_files(dataset_name, path=download_dir, unzip=True)


#extracting zip files (if they weren’t automatically extracted)
for root, dirs, files in os.walk(download_dir):
    for file in files:
        if file.endswith(".zip"):
            file_path = os.path.join(root, file)
            with zipfile.ZipFile(file_path, 'r') as zip_ref:
                zip_ref.extractall(download_dir)
            os.remove(file_path)  # Optionally remove zip file after extraction

print("Download and extraction complete!")
