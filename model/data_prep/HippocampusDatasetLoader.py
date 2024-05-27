import os
from os import listdir
from os.path import isfile, join
import numpy as np
from medpy.io import load
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..','model')))

from utils.utils import med_reshape


def Hippocampus_Data(root_dir, y_shape, x_shape):
    image_dir = os.path.join(os.path.abspath(root_dir), 'unzipped_img')
    labels_dir = os.path.join(os.path.abspath(root_dir), 'unzipped_labels')
    print(image_dir, labels_dir)

    images = [f for f in listdir(image_dir) if (isfile(join(image_dir,f)) and f[0] != ".")]
    out = []
    for f in images:
        image, _ = load(os.path.join(image_dir, f))
        label, _ = load(os.path.join(image_dir, f))

        image = image/255

        image = med_reshape(image, new_shape=(image.shape[0], y_shape, z_shape))
        label = med_reshape(label, new_shape=(label.shape[0], y_shape, z_shape)).astype(int)

        out.append({"image":image, "seg":label, "filename":f})
        print(f"Processed {len(out)} files, total {sum([x['image'].shape[0] for x in out])} slices")
    return np.array(out)

if __name__ == "__main__":
    root_directory = "C:\\Users\\aryan\\OneDrive\\Desktop\\Hypocampal Volume Quantification of Alzheimers"  # Replace with the actual path
    y_shape = 64  
    z_shape = 64  
    data = Hippocampus_Data(root_directory, y_shape, z_shape)
    print(f"Total processed files: {len(data)}")