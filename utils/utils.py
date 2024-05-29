import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl 
import torch
from PIL import Image
# from scipy.ndimage import zoom

from skimage.transform import resize


mpl.use("agg")

def mpl_image_grid(images):
    n = min(images.shape[0], 16) # no more than 16 thumbnails
    rows = 4
    cols = (n // 4) + (1 if (n % 4) != 0 else 0)
    figure = plt.figure(figsize=(2*rows, 2*cols))
    plt.subplots_adjust(0, 0, 1, 1, 0.001, 0.001)
    for i in range(n):
        # Start next subplot.
        plt.subplot(cols, rows, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        if images.shape[1] == 3:
            # this is specifically for 3 softmax'd classes with 0 being bg
            # We are building a probability map from our three classes using 
            # fractional probabilities contained in the mask
            vol = images[i].detach().numpy()
            img = [[[(1-vol[0,x,y])*vol[1,x,y], (1-vol[0,x,y])*vol[2,x,y], 0] \
                            for y in range(vol.shape[2])] \
                            for x in range(vol.shape[1])]
            plt.imshow(img)
        else: # plotting only 1st channel
            plt.imshow((images[i, 0]*255).int(), cmap= "gray")

    return figure

def save_numpy_img(arr, path):
    plt.imshow(arr, cmap="gray")
    plt.savefig(path)

def med_reshape(image, new_shape):
    reshaped_image = resize(image, new_shape, mode='constant', anti_aliasing=True, preserve_range=True)
    return reshaped_image

def med_reshape_label(label, new_shape):
    reshaped_label = resize(label, new_shape, mode='constant', order=0, preserve_range=True, anti_aliasing=False)
    return reshaped_label
    

def log_to_tensorboard(writer, loss, data, target, prediction, counter):
    writer.add_scalar("Loss",\
        loss, counter)
    writer.add_figure("Image Data",\
        mpl_image_grid(data.float().cpu()), global_step=counter)
    writer.add_figure("Mask",\
        mpl_image_grid(target.float().cpu()), global_step=counter)
    # writer.add_figure("Probability map",\
    #     mpl_image_grid(prediction_softmax.cpu()), global_step=counter)
    writer.add_figure("Prediction",\
        mpl_image_grid(torch.argmax(prediction.cpu(), dim=1, keepdim=True)), global_step=counter)

    