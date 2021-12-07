import numpy as np
import skimage.measure
import numpy as np
import cv2
import matplotlib.pyplot as plt

def calculate_gradients(img):
    sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)
    sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=5)
    gradient = np.hypot(sobelx,sobely)
    return gradient

def redies_complexity(gradient, mask):
    grad_sum = np.sum(gradient)
    return grad_sum / np.sum(mask)

def position_uniformity(positions):
    centre = np.mean(positions, axis=0)
    dist_to_centre = np.linalg.norm(positions - centre, axis=1)
    pos_variance = np.var(dist_to_centre)
    return pos_variance

def scaling_uniformity(scales):
    return np.var(scales)

def gradient_uniformity(gradients):
    return np.var(gradients)

def features_variance(features):
    feat_var = np.var(features, axis=0)
    return np.mean(feat_var)

def shannon_entropy(img):
    return skimage.measure.shannon_entropy(img)

def minkowski_fractal_dimensionality(img):
    img = (img < 0.9)

    n, m = img.shape
    n = 2**np.floor(np.log(min(img.shape))/np.log(2))

    box_dims = 2**np.arange(int(np.log(n)/np.log(2)), 1, -1)

    count_by_dim = []
    for side in box_dims:
        count = np.add.reduceat(np.add.reduceat(img, np.arange(0, np.int32(n), side), axis=0), np.arange(0, np.int32(m), side), axis=1)
        count_by_dim.append(len(np.where((count != 0) & (count != side**2))[0]))

    return -np.polyfit(np.log(box_dims), np.log(count_by_dim), 1)[0]

