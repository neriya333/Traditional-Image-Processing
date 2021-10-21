import numpy as np
import matplotlib.pyplot as plt

""" import scikit-image as skmg """
import scipy as sp
import imageio as imao
from skimage import color

GRAY_SCALE = 1
COLOR_RANGE = 255

RGB2YIQ_MATRIX = np.array([[0.299, 0.587, 0.144],
                           [0.596, -0.275, -0.321],
                           [0.212, -0.523, 0.311]])

"""load color image and convert to grayScale if representaion is 1 """

"""Q1"""


def read_image(path, representation=2):
    img = imao.imread(path)
    if representation == GRAY_SCALE:
        img = color.rgb2gray(img)
    else:
        img = img / COLOR_RANGE
    return img


"""Q2"""


def imdisplay(path, rerepresentation=2):
    img = read_image(path, rerepresentation)
    # img = -1*(1 - img)  # convert the image
    if rerepresentation == GRAY_SCALE:
        plt.imshow(img, 'Greys')
    else:
        plt.imshow(img)
    plt.show()


"""Q3"""


# given NxMx3 for RGB return YIQ - not optimized.
def rgb2yiq(imRGB):
    """
    :param imRGB: NxMx3 where 3 is RGB, with values in[0,1]
    :return: imYIQ: NxMx3 where 3 is YIQ
    """
    dim = imRGB.shape
    imYIQ = np.zeros(dim)

    for i in range(3):
        imYIQ[i] = imRGB[0] * RGB2YIQ_MATRIX[i][0] + \
                   imRGB[1] * RGB2YIQ_MATRIX[i][1] + \
                   imRGB[2] * RGB2YIQ_MATRIX[i][2]

    return imYIQ


def yiq2rgb(imYIQ):
    """
    :param imYIQ: NxMx3 where 3 is YIQ, with values in[-1,1]
    :return: imRGB: NxMx3 where 3 is YIQ
    """
    dim = imYIQ.shape
    imRGB = np.zeros(dim)
    rev_RGB2YIQ_MATRIX = np.linalg.inv(RGB2YIQ_MATRIX)

    for i in range(3):
        imRGB[i] = imYIQ[0] * rev_RGB2YIQ_MATRIX[i][0] + \
                   imYIQ[1] * rev_RGB2YIQ_MATRIX[i][1] + \
                   imYIQ[2] * rev_RGB2YIQ_MATRIX[i][2]

    return imRGB



"""testing\playground field"""

# x = np.hstack([np.repeat(np.arange(0, 50, 2), 10)[None, :], np.array([255] * 6)[None, :]])
# grad = np.array([[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3], [4, 4, 4, 4], [6, 6, 6, 6]])
# gard3 = np.array([grad, grad * 10, grad * 20])/120
#
#
# # img_url = 'jerusalem.jpg'
# # imdisplay(img_url, 1)  # check q2 and 1 without wired cases
#
#
# img = rgb2yiq(gard3)
# img = yiq2rgb(img)
# plt.imshow(img)
# plt.show()
