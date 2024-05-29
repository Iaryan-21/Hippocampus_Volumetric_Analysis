import os
from os import listdir
from os.path import isfile, join
import numpy as np
from medpy.io import load
import matplotlib.pyplot as plt

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..','model')))
from utils.utils import med_reshape, med_reshape_label

def Hippocampus_Data(root_dir, y_shape, z_shape):
    image_dir = os.path.join(root_dir, 'unzipped_img')
    labels_dir = os.path.join(root_dir, 'unzipped_labels')
    
    print(f"Image Directory: {image_dir}, Labels Directory: {labels_dir}")

    if not os.path.exists(image_dir) or not os.path.exists(labels_dir):
        raise FileNotFoundError(f"Image directory or label directory not found. Image directory: {image_dir}, Label directory: {labels_dir}")

    images = [f for f in listdir(image_dir) if (isfile(join(image_dir, f)) and f[0] != ".")]
    out = []
    for f in images:
        image, _ = load(os.path.join(image_dir, f))
        label, _ = load(os.path.join(labels_dir, f))

        # Debugging: Print the original shapes and unique values
        print(f"Original Image Shape: {image.shape}, Original Label Shape: {label.shape}")
        print(f"Original Label Unique Values: {np.unique(label)}")

        image = image / 255.0  # Normalize image

        # Reshape images and labels correctly
        image = med_reshape(image, new_shape=(1, y_shape, z_shape))  # Ensure single channel
        label = med_reshape_label(label, new_shape=(1, y_shape, z_shape)).astype(int)  # Ensure single channel

        # Debugging: Print the reshaped shapes and unique values
        print(f"Reshaped Image Shape: {image.shape}, Reshaped Label Shape: {label.shape}")
        print(f"Reshaped Label Unique Values: {np.unique(label)}")

        out.append({"image": image, "seg": label, "filename": f})
        print(f"Processed {len(out)} files, total {sum([x['image'].shape[1] for x in out])} slices")
    
    # Convert to numpy array
    data_array = np.array(out)

    # Debugging: Visualize a few samples
    visualize_samples(data_array)

    return data_array

def visualize_samples(data_array, num_samples=3):
    for i in range(num_samples):
        sample = data_array[i]
        image = sample['image'].squeeze()
        label = sample['seg'].squeeze()
        
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.title(f"Image - {sample['filename']}")
        plt.imshow(image, cmap='gray')
        
        plt.subplot(1, 2, 2)
        plt.title(f"Label - {sample['filename']}")
        plt.imshow(label, cmap='gray')
        
        plt.show()

# # Example usage
# root_dir = r'C:\\Users\\aryan\\OneDrive\\Desktop\\Hypocampal Volume Quantification of Alzheimers\\model\\data\\'
# y_shape, z_shape = 64, 64
# data = Hippocampus_Data(root_dir, y_shape, z_shape)
