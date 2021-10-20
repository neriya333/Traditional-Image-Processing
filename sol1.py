import numpy as np
import matplotlib.pyplot as plt
""" import scikit-image as skmg """
import scipy as sp
import imageio as imao
from skimage import color

GRAY_SCALE = 1

x = np.hstack([np.repeat(np.arange(0,50,2),10)[None,:], np.array([255]*6)[None,:]])
grad = np.tile(x,(256,1))


def load_img(path, representation=2):
    img = imao.imread(path)
    if representation == GRAY_SCALE:
        img = color.rgb2gray(img)
    return img
