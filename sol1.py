import numpy as np
import matplotlib.pyplot as plt
""" import scikit-image as skmg """
import scipy as sp
import imageio as imao
from skimage import color

GRAY_SCALE = 1
COLOR_RANGE = 255

x = np.hstack([np.repeat(np.arange(0,50,2),10)[None,:], np.array([255]*6)[None,:]])
grad = np.tile(x,(256,1))

"""load color image and convert to grayScale if representaion is 1 """
def read_image(path, representation=2):
    img = imao.imread(path)
    if representation == GRAY_SCALE:
        img = color.rgb2gray(img)
    else:
        img = img/COLOR_RANGE
    return img


def imdisplay(path, rerepresentation=2):
    img = read_image(path, rerepresentation)
    # img = -1*(1 - img)  # convert the image
    if rerepresentation == GRAY_SCALE:
        plt.imshow(img, 'Greys')
    else:
        plt.imshow(img)
    plt.show()

img_url = 'jerusalem.jpg'
imdisplay(img_url, 1)

