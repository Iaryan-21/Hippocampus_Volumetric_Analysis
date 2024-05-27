import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl 
import torch
from PIL import Image

mpl.use("agg")

def mpl_image_grid(images):
    n = min(images.shape[0], 16)
    rows = 4
    columns = (n//4) + (1 if (n%4)!=0 else 0) 
    figure = plt.figure(figsize=(2*rows, 2*columns))
    plt.subplots_adjust(0,0,1,1,0.001,0.001)
    for i in range(n):
        plt.subplots(columns, rows, i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        if images.shape[1] == 3:
            vol = images[i].detach().numpy()
            img = [[[(1-vol[0,x,y])*vol[1,x,y], (1-vol[0,x,y])*vol[2,x,y], 0] \
                            for y in range(vol.shape[2])] \
                            for x in range(vol.shape[1])]
            plt.imshow(img)
        else:   
            plt.imshow((images[i, 0]*255).int(), cmap= "gray")

    return figure

def save_numpy_img(arr, path):
    plt.imshow(arr, cmap="gray")
    plt.savefig(path)

def med_reshape(image, new_shape):
    reshaped_image = np.zeros(new_shape)
    reshaped_image[:image.shape[0], :image.shape[1], :image.shape[2]] = image
    return reshaped_image