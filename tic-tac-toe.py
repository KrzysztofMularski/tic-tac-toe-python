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
from skimage.color import rgba2rgb

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

#def hough(img, ax): #
def hough(img):
    # Classic straight-line Hough transform
    # Set a precision of 0.5 degree.
    tested_angles = np.linspace(-np.pi / 2, np.pi / 2, 30)
    h, theta, d = hough_line(img, theta=tested_angles)

    # Generating figure 1
    #fig, axes = plt.subplots(1, 3, figsize=(15, 6))

    #ax.imshow(img, cmap='gray') #
    axies = []
    origin = np.array((0, img.shape[1]))
    for _, angle, dist in zip(*hough_line_peaks(h, theta, d, threshold=0.5*h.max(), num_peaks=4)):
        y0, y1 = (dist - origin * np.cos(angle)) / np.sin(angle)
        #ax.plot(origin, (y0, y1), '-r')
        axies.append([y0,y1])

    #ax.set_xlim(origin)
    #ax.set_ylim((img.shape[0], 0))
    #ax.set_axis_off()
    #ax.set_title('Detected lines')
    return axies

def hu():
    list_files = os.listdir('ideals/')
    images = []
    for filename in list_files:
        images.append(io.imread('ideals/'+filename))
    
    img = rgb2gray(images[0])
    img = 1-img
    mu1 = moments_central(img)
    nu1 = moments_normalized(mu1)
    mom1 = moments_hu(nu1)

    img = rgb2gray(images[1])
    img = 1-img
    mu2 = moments_central(img)
    nu2 = moments_normalized(mu2)
    mom2 = moments_hu(nu2)

    return mom1, mom2

def chunkhu(chunk):
    mu1 = moments_central(chunk)
    nu1 = moments_normalized(mu1)
    mom1 = moments_hu(nu1)
    return mom1

def circle(huimg, diffo, huo):
    sum = 0
    for i in range(7):
        sum += (abs(huimg[i])-abs(huo[i]))
    if (sum < diffo):
        return True
    else:
        return False

def cross(huimg, diffx, hux):
    sum = 0
    for i in range(7):
        sum += (abs(huimg[i])-abs(hux[i]))
    if (sum < diffx):
        return True
    else:
        return False

def checkfield(img, diffx, diffo, hux, huo):
    moment = chunkhu(img)
    if (circle(moment,diffo,huo)):
        #print("Chunk ", i, " is circle.")
        return 1
    elif (cross(moment,diffx,hux)):
        #print("Chunk ", i, " is cross.")
        return 2
    else:
        #print("Chunk ", i, " is empty.")
        return 0


def is_descending(line, height):
    if line[0] < 0:
        return True
    else:
        return False

def transform_func(ver, width):
    y0, y1 = ver
    ver_height = y0 - y1
    a = ver_height / width
    b = -y0
    return a, b

def change_lines_order(hor_tran, ver_tran):
    hor_tran2 = []
    ver_tran2 = []
    if len(hor_tran) == 2 and len(ver_tran) == 2:
        if hor_tran[1][1] > hor_tran[0][1]:
            hor_tran2.append(hor_tran[1])
            hor_tran2.append(hor_tran[0])
        else:
            hor_tran2 = hor_tran
        if ver_tran[1][1]/ver_tran[1][0] > ver_tran[0][1]/ver_tran[0][0]:
            ver_tran2.append(ver_tran[1])
            ver_tran2.append(ver_tran[0])
        else:
            ver_tran2 = ver_tran
        return hor_tran2, ver_tran2
    return hor_tran, ver_tran

#i: y
#j: x

def at_top(i, j, hline):
    a, b = hline
    if a*j+b < -i:
        return 1
    else:
        return 0

def at_bottom(i, j, hline):
    return 1-at_top(i, j, hline)

def on_left(i, j, vline, height):
    if is_descending(vline, height):
        return at_bottom(i, j, vline)
    else:
        return at_top(i, j, vline)

def on_right(i, j, vline, height):
    if is_descending(vline, height):
        return at_top(i, j, vline)
    else:
        return at_bottom(i, j, vline)

def get_pxl_pos(i, j, hor, ver, height):
    if on_left(i, j, ver[0], height) and on_left(i, j, ver[1], height) and at_top(i, j, hor[0]) and at_top(i, j, hor[1]):
        return 0
    elif on_right(i, j, ver[0], height) and on_left(i, j, ver[1], height) and at_top(i, j, hor[0]) and at_top(i, j, hor[1]):
        return 1
    elif on_right(i, j, ver[0], height) and on_right(i, j, ver[1], height) and at_top(i, j, hor[0]) and at_top(i, j, hor[1]):
        return 2
    elif on_left(i, j, ver[0], height) and on_left(i, j, ver[1], height) and at_bottom(i, j, hor[0]) and at_top(i, j, hor[1]):
        return 3
    elif on_right(i, j, ver[0], height) and on_left(i, j, ver[1], height) and at_bottom(i, j, hor[0]) and at_top(i, j, hor[1]):
        return 4
    elif on_right(i, j, ver[0], height) and on_right(i, j, ver[1], height) and at_bottom(i, j, hor[0]) and at_top(i, j, hor[1]):
        return 5
    elif on_left(i, j, ver[0], height) and on_left(i, j, ver[1], height) and at_bottom(i, j, hor[0]) and at_bottom(i, j, hor[1]):
        return 6
    elif on_right(i, j, ver[0], height) and on_left(i, j, ver[1], height) and at_bottom(i, j, hor[0]) and at_bottom(i, j, hor[1]):
        return 7
    elif on_right(i, j, ver[0], height) and on_right(i, j, ver[1], height) and at_bottom(i, j, hor[0]) and at_bottom(i, j, hor[1]):
        return 8
    
    return 0

def get_pxl_pos2(i,j,d):
    if (i<d) and (j<d):
        return 0
    if (i<d) and (j<2*d):
        return 1
    if (i<d) and (j<3*d):
        return 2
    if (i<2*d) and (j<d):
        return 3
    if (i<2*d) and (j<2*d):
        return 4
    if (i<2*d) and (j<3*d):
        return 5
    if (i<3*d) and (j<d):
        return 6
    if (i<3*d) and (j<2*d):
        return 7
    if (i<3*d) and (j<3*d):
        return 8

def get_crossed_point(line1, line2):
    a1, b1 = line1
    a2, b2 = line2
    x = (b1-b2)/(a2-a1)
    y = a1*x+b1
    return [x, y]
    

def get_points(hor_tran, ver_tran):
    points = []
    points.append(get_crossed_point(hor_tran[0], ver_tran[0]))
    points.append(get_crossed_point(hor_tran[0], ver_tran[1]))
    points.append(get_crossed_point(hor_tran[1], ver_tran[0]))
    points.append(get_crossed_point(hor_tran[1], ver_tran[1]))
    return points

def dimensions(img, points, d,height,width):
    leftup_x = points[0][0]-d
    leftup_y = points[0][1]+d
    rightup_x = points[1][0]+d
    rightup_y = points[1][1]+d
    leftdown_x = points[2][0]-d
    leftdown_y = points[2][1]-d
    rightdown_x = points[3][0]+d
    rightdown_y = points[3][1]-d
    newimg = np.zeros((3*d,3*d))
    l = -1
    m = -1
    for j in range(int(-leftup_y),int(-(leftup_y-3*d)),1):
            l+=1
            m=-1
            for k in range(int(leftup_x), int(leftup_x+3*d)):
                m+=1
                if(j < height) and (k < width):
                    newimg[l][m] = img[j][k]
    return newimg

if __name__ == '__main__':

    list_files = os.listdir('pictures/')
    images = []
    for filename in list_files:
        images.append(io.imread('pictures/'+filename))
    #images.append(io.imread('planes/'+list_files[1]))

    correct_pics_ids = list(range(25))
    
    #correct_pics_ids.remove(0)

    #fig = plt.figure(figsize=(20, 40)) #
    
    kolko_hu7, krzyzyk_hu7 = hu()
    fig_test = plt.figure(figsize=(20, 40))
    for p in correct_pics_ids:

        #ax = fig.add_subplot(7, 3, i+1) #
        #plt.axis('off') #
        imgorg = images[p]
        img = processing(images[p])
        #ax.imshow(img, cmap='gray')
        #axies = hough(img, ax) #
        axies = hough(img)
        hor = []
        ver = []
        height = len(img)
        width = len(img[0])
        for y0, y1 in axies:
            if y0 > 0 and y0 < height and y1 > 0 and y1 < height:
                hor.append([y0,y1])
            else:
                ver.append([y0,y1])

        if len(hor) != 2 or len(ver) != 2:
            print("Obrazek nr: ", p, " nie dziala")
            continue
        hor_tran = []
        ver_tran = []

        for line in hor:
            hor_tran.append(transform_func(line, width))

        for line in ver:
            ver_tran.append(transform_func(line, width))

        hor_tran, ver_tran = change_lines_order(hor_tran, ver_tran)

        d = hor_tran[0][1] - hor_tran[1][1]
        #d *= 1.1
        d = int(d)

        points = get_points(hor_tran, ver_tran)
        imgold = img
        img = dimensions(img, points,d,height,width)
        height = len(img)
        width = len(img[0])
        chunks = np.zeros((9, d, d))
        #chunks = np.zeros((9, height, width))
        for j in range(height):
            for k in range(width):
                #chunks[get_pxl_pos(j, k, hor_tran, ver_tran, height)][j%d][k%d] = img[j][k]
                chunks[get_pxl_pos2(j, k, d)][j%d][k%d] = img[j][k]
        for i in range(9):
            chunks[i] = fill(chunks[i])

        diffx = 0.1
        diffo = 0.3
        ax_test = fig_test.add_subplot(25, 10, p*10+1)
        plt.axis('off')
        ax_test.imshow(imgorg, cmap='gray')
        ax_test.set_title("Original")
        for i in range(9):
            field = checkfield(chunks[i],diffx,diffo,krzyzyk_hu7, kolko_hu7)
            ax_test = fig_test.add_subplot(25, 10, p*10+i+2)
            plt.axis('off')
            ax_test.imshow(chunks[i], cmap='gray')
            if (field ==0):
                ax_test.set_title("Empty")
            elif (field ==1):
                ax_test.set_title("Circle")
            elif (field ==2):
                ax_test.set_title("Cross")
        #ax_test = fig_test.add_subplot(4, 3, 10)
        #ax_test.imshow(imgold, cmap='gray')
        #ax_test = fig_test.add_subplot(4, 3, 11)
        #ax_test.imshow(img, cmap='gray')
        plt.savefig('wyniki1-25.pdf')


    #plt.savefig('processed.pdf')
