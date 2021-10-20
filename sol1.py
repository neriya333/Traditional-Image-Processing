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


def load_img(path, representation=2):
    img = imao.imread(path)
    if representation == GRAY_SCALE:
        img = color.rgb2gray(img)
    else:
        img = img/COLOR_RANGE
    return img

img_url = 'jerusalem.jpg'
load_img(img_url)
