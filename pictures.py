import os
from skimage import io
from skimage import filters
from skimage import feature
from skimage import util
from skimage import segmentation as seg
from skimage.measure import label
from skimage.measure import regionprops
from skimage.color import rgb2gray
from skimage.filters.edges import convolve
import skimage as ski

import math
import scipy
from scipy import ndimage as ndi
import numpy as np
import matplotlib.pyplot as plt
import skimage.morphology as mp


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

    img = contrast(img, 0.3)
    img = negate(img)
    img = gamma(img, 3)

    img = dilation(img)
    img = dilation(img)
    img = gauss(img, 3)
    img = canny(img, 3)
    img = dilation(img)
    img = dilation(img)
    img = dilation(img)
    img = dilation(img)
    img = dilation(img)
    img = fill(img)
    img = erosion(img)
    img = erosion(img)

    return img

def contour(img, ax):
    contours = ski.measure.find_contours(img, 0.5)
    for contour in contours:
        ax.plot(contour[:, 1], contour[:, 0], linewidth=2)

def centroid(img, ax):
    label_image = label(img)
    centr_x = []
    centr_y = []
    for region in regionprops(label_image):
        if region.area >= 100:
            centr_x.append(region['Centroid'][0])
            centr_y.append(region['Centroid'][1])
    ax.scatter([centr_y],[centr_x], c = 'w', s = 10)

if __name__ == '__main__':

    list_files = os.listdir('planes/')
    images = []
    for filename in list_files:
        images.append(io.imread('planes/'+filename))
    #images.append(io.imread('planes/'+list_files[12]))

    correct_planes_ids = list(range(21))
    #correct_planes_ids = list(range(18))
    #correct_planes_ids = list(range(1))
    
    correct_planes_ids.remove(20)
    correct_planes_ids.remove(19)
    correct_planes_ids.remove(6)


    fig = plt.figure(figsize=(20, 40))
    
    for i in correct_planes_ids:
        #ax = fig.add_subplot(6, 3, i+1)
        ax = fig.add_subplot(7, 3, i+1)
        plt.axis('off')
        img = processing(images[i])
        ax.imshow(images[i], cmap='gray')
        #ax.imshow(img,cmap='gray')
        contour(img, ax)
        centroid(img, ax)

    plt.savefig('planes.pdf')
