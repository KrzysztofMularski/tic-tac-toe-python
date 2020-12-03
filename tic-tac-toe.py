import os
from skimage import io
from skimage import filters
from skimage import feature
from skimage import util
from skimage import segmentation as seg
from skimage.measure import label
from skimage.measure import regionprops
from skimage.measure import moments_central
from skimage.measure import moments_normalized
from skimage.measure import moments_hu
from skimage.color import rgb2gray
from skimage.filters.edges import convolve
import skimage as ski

import math
import scipy
from scipy import ndimage as ndi
import numpy as np
import matplotlib.pyplot as plt
import skimage.morphology as mp


from matplotlib import cm
from skimage.transform import hough_line, hough_line_peaks


def negate(img):
    return 1-img

def threshing(img, thresh=50):
    binary = (img*255) > thresh
    binary = np.uint8(binary)
    return binary

def normalization(img, MIN=50, MAX=150):
    norm = (img*255 - MIN) / (MAX - MIN)
    norm[norm > 1] = 1
    norm[norm < 0] = 0
    return norm

def contrast(img, perc = 1.0):
    MIN = np.percentile(img, perc)
    MAX = np.percentile(img, 100-perc)
    #print("contrast:",MIN,MAX,"\n")
    norm = (img - MIN) / (MAX - MIN)
    norm[norm[:,:] > 1] = 1
    norm[norm[:,:] < 0] = 0
    return norm

def watershed(img):
    img = util.img_as_ubyte(img)
    markers = filters.rank.gradient(img, mp.disk(1)) < 20
    markers = ndi.label(markers)[0]
    gradient = filters.rank.gradient(img, mp.disk(5))
    return seg.watershed(gradient, markers)

def gamma(img, gamma=1.8):
    return img ** gamma

def sobel(img):
    return filters.sobel(img)

def median(img):
    img = ski.img_as_ubyte(img)
    img = filters.rank.median(img, np.ones([5,5], dtype=np.uint8))
    return img

def gauss(img, sigma=3):
    return scipy.ndimage.gaussian_filter(img, sigma)

def canny(img, sigma=3):
    return feature.canny(img, sigma)

def erosion(img):
    return mp.erosion(img)

def dilation(img):
    return mp.dilation(img)

def closing(img):
    dilated = dilation(img)
    return erosion(dilated)

def opening(img):
    erosed = mp.erosion(img)
    return dilation(erosed)

def fill(img):
    return scipy.ndimage.morphology.binary_fill_holes(img)

def processing(img):

    img = rgb2gray(img)
    img = contrast(img, 1)
    img = threshing(img, 100)

    img = negate(img)

    return img

def hough(img, ax):
    # Classic straight-line Hough transform
    # Set a precision of 0.5 degree.
    tested_angles = np.linspace(-np.pi / 2, np.pi / 2, 360)
    h, theta, d = hough_line(img, theta=tested_angles)

    # Generating figure 1
    #fig, axes = plt.subplots(1, 3, figsize=(15, 6))

    ax.imshow(img, cmap='gray')
    origin = np.array((0, img.shape[1]))
    for _, angle, dist in zip(*hough_line_peaks(h, theta, d, threshold=0.5*h.max())):
        y0, y1 = (dist - origin * np.cos(angle)) / np.sin(angle)
        ax.plot(origin, (y0, y1), '-r')
    ax.set_xlim(origin)
    ax.set_ylim((img.shape[0], 0))
    ax.set_axis_off()
    ax.set_title('Detected lines')

def hu():
    list_files = os.listdir('ideals/')
    images = []
    for filename in list_files:
        images.append(io.imread('ideals/'+filename))
    
    img = rgb2gray(images[0])
    mu1 = moments_central(img)
    nu1 = moments_normalized(mu1)
    mom1 = moments_hu(nu1)

    img = rgb2gray(images[1])
    mu2 = moments_central(img)
    nu2 = moments_normalized(mu2)
    mom2 = moments_hu(nu2)

    return mom1[6], mom2[6]

if __name__ == '__main__':

    list_files = os.listdir('pictures/')
    images = []
    for filename in list_files:
        images.append(io.imread('pictures/'+filename))
    #images.append(io.imread('planes/'+list_files[1]))

    correct_planes_ids = list(range(5))
    
    #correct_planes_ids.remove(0)

    fig = plt.figure(figsize=(20, 40))
    
    kolko_hu7, krzyzyk_hu7 = hu()

    for i in correct_planes_ids:
        #ax = fig.add_subplot(6, 3, i+1)
        ax = fig.add_subplot(7, 3, i+1)
        plt.axis('off')
        img = processing(images[i])
        #ax.imshow(img, cmap='gray')
        hough(img, ax)

        

    plt.savefig('processed.pdf')
