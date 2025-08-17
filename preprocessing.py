# -*- coding: utf-8 -*-
"""
Created on Sun Aug 17 15:32:44 2025

@author: Oxana
"""

from pathlib import Path
import os
import pydicom
import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
import kagglehub

# Download des Datensatzes
path = kagglehub.dataset_download("salemrezzag/rsna-pneumonia-detection-2")
labels_file = "lesson3-data/stage_2_train_labels.csv"
image_folder = "lesson3-data/rsna-pneumonia-detection-challenge/stage_2_train_images/"

print("Path to dataset files:", path)
print("Labels file:", labels_file)

# Pfade setzen
labels_path = os.path.join(path, labels_file)
print("Path to labels:", labels_path)

dataset_path = os.path.join(path, image_folder)
print("Path to dataset:", dataset_path)

labels = pd.read_csv(labels_path)


SAVE_PATH = Path("Processed/")


labels.head(6)

# Remove duplicate entries
labels = labels.drop_duplicates("patientId")


"""
Dataloader;
The X-Ray images stored in the DICOM format are converted to numpy arrays. 
Overall mean and standard deviation of the pixels of the whole dataset are computed for the purpose of normalization. 
Created numpy images are stored in two separate folders according to their binary label:
0: X-Rays which do not show signs of pneumonia
1: X-Rays which show signs of pneumonia
"""
sums = 0
sums_squared = 0

for c, patient_id in enumerate(tqdm(labels.patientId)):
    dcm_path = os.path.join(dataset_path, f"{patient_id}.dcm")  # Create the path to the dcm file

    print("DCM: ", dcm_path)
    # Read the dicom file with pydicom and standardize the array
    #dcm = pydicom.read_file(dcm_path).pixel_array / 255  
    dcm = pydicom.dcmread(dcm_path).pixel_array / 255
    # Resize the image as 1024x1024 is way to large to be handeled by Deep Learning models at the moment
    # use a shape of 224x224
    # In order to use less space when storing the image we convert it to float16
    dcm_array = cv2.resize(dcm, (224, 224)).astype(np.float16)
    
    # Retrieve the corresponding label
    label = labels.Target.iloc[c]
    
    # 4/5 train split, 1/5 val split
    train_or_val = "train" if c < 24000 else "val" 
        
    current_save_path = SAVE_PATH/train_or_val/str(label) # Define save path and create if necessary
    current_save_path.mkdir(parents=True, exist_ok=True)
    np.save(current_save_path/patient_id, dcm_array)  # Save the array in the corresponding directory
    
    normalizer = dcm_array.shape[0] * dcm_array.shape[1]  # Normalize sum of image
    if train_or_val == "train":  # Only use train data to compute dataset statistics
        sums += np.sum(dcm_array) / normalizer
        sums_squared += (np.power(dcm_array, 2).sum()) / normalizer


mean = sums / 24000
std = np.sqrt(sums_squared / 24000 - (mean**2))

print(f"Mean of Dataset: {mean}, STD: {std}")