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

    if dim[-1] != 3: ValueError("Input array must have a shape == (..., 3)), "f"got {dim}")

    imYIQ = np.zeros(dim)

    for i in range(dim[0]):
        for j in range(dim[1]):
            imYIQ[i][j] = np.dot(RGB2YIQ_MATRIX, imRGB[i][j])

    return imYIQ


def yiq2rgb(imYIQ):
    """
    :param imYIQ: NxMx3 where 3 is YIQ, with values in[-1,1]
    :return: imRGB: NxMx3 where 3 is YIQ
    """
    dim = imYIQ.shape
    if dim[-1] != 3: ValueError("Input array must have a shape == (..., 3)), "f"got {dim}")

    imRGB = np.zeros(dim)
    rev_RGB2YIQ_MATRIX = np.linalg.inv(RGB2YIQ_MATRIX)

    for i in range(dim[0]):
        for j in range(dim[1]):
            imRGB[i][j] = np.dot(rev_RGB2YIQ_MATRIX, imYIQ[i][j])

    return imRGB


# def histogram_equalization(im_orig):
#     """
#
#     :param im_orig: grey scale or RGB normalized to [0,1]
#     :return: [im_eq, hist_orig, hist_eq] where
#               im_eq - is the equalized image. grayscale or RGB float64 image with values in [0, 1].
#               hist_orig - is a 256 bin histogram of the original image (array with shape (256,) ).
#               hist_eq - is a 256 bin histogram of the equalized image (array with shape (256,) )
#     """
#     # make YIQ to work Y with insteade of RGB
#     if type(im_orig[0][0]) == tuple:
#         img = rgb2yiq(im_orig)
#
#
#
#     # return [im_wq, hist_orgi, histogram_eq]



"""testing\playground field"""

# # x = np.hstack([np.repeat(np.arange(0, 50, 2), 10)[None, :], np.array([255] * 6)[None, :]])
# grad = np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4], [6, 6, 6]])
# gard3 = np.array([grad, grad * 10, grad * 20])/120
# #
# #
# img_url = 'jerusalem.jpg'
# jer = read_image(img_url)  # check q2 and 1 without wired cases
# #
# #
# # img = rgb2yiq(jer)
# # img2 = color.rgb2yiq(jer)
# # check = img2-img



print("here")