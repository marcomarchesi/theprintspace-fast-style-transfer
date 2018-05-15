# erosion/dilation with Python and SciPy

from argparse import ArgumentParser

import scipy.ndimage
import scipy
import numpy as np


def generate_trimap(img):
    img = img[:,:,0]

    num_iterations = int(img.shape[0] * 0.05)

    
    eroded_img = scipy.ndimage.binary_erosion(img, iterations=num_iterations)
    dilated_img = scipy.ndimage.binary_dilation(img, iterations=num_iterations)

    # trimap as difference between dilated and eroded masks
    trimap_img = np.full(shape=(img.shape[0], img.shape[1]), fill_value=127)
    trimap_img = (dilated_img - eroded_img).astype(img.dtype)

    # find the indices where pixels are white and replace with gray=127
    idx = (trimap_img > 0)
    trimap_img[idx] = 127

    # I can make it better
    for i in range(eroded_img.shape[0]):
        for j in range(eroded_img.shape[1]):
            if trimap_img[i][j] != 127 and img[i][j] == 255:
                trimap_img[i][j] = 255

    trimap_img = np.stack((trimap_img,)*3, -1)

    return trimap_img
