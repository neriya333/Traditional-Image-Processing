import numpy as np
import matplotlib.pyplot as plt

""" import scikit-image as skmg """
import scipy as sp
import imageio as imao
from skimage import color

GRAY_SCALE = 1
COLOR_RANGE = 256

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

    Y = RGB2YIQ_MATRIX[0][0] * imRGB[:, :, 0] + RGB2YIQ_MATRIX[0][1] * imRGB[:, :, 1] + RGB2YIQ_MATRIX[0][2] * imRGB[:,
                                                                                                               :, 2]
    I = RGB2YIQ_MATRIX[1][0] * imRGB[:, :, 0] + RGB2YIQ_MATRIX[1][1] * imRGB[:, :, 1] + RGB2YIQ_MATRIX[1][2] * imRGB[:,
                                                                                                               :, 2]
    Q = RGB2YIQ_MATRIX[2][0] * imRGB[:, :, 0] + RGB2YIQ_MATRIX[2][1] * imRGB[:, :, 1] + RGB2YIQ_MATRIX[2][2] * imRGB[:,
                                                                                                               :, 2]

    return np.stack((Y, I, Q), axis=-1)


def yiq2rgb(imYIQ):
    """
    :param imYIQ: NxMx3 where 3 is YIQ, with values in[-1,1]
    :return: imRGB: NxMx3 where 3 is YIQ
    """

    rev_RGB2YIQ_MATRIX = np.linalg.inv(RGB2YIQ_MATRIX)
    dim = imYIQ.shape

    if dim[-1] != 3: ValueError("Input array must have a shape == (..., 3)), "f"got {dim}")

    R = rev_RGB2YIQ_MATRIX[0][0] * imYIQ[:, :, 0] + rev_RGB2YIQ_MATRIX[0][1] * imYIQ[:, :, 1] + rev_RGB2YIQ_MATRIX[0][
        2] * imYIQ[:, :, 2]
    G = rev_RGB2YIQ_MATRIX[1][0] * imYIQ[:, :, 0] + rev_RGB2YIQ_MATRIX[1][1] * imYIQ[:, :, 1] + rev_RGB2YIQ_MATRIX[1][
        2] * imYIQ[:, :, 2]
    B = rev_RGB2YIQ_MATRIX[2][0] * imYIQ[:, :, 0] + rev_RGB2YIQ_MATRIX[2][1] * imYIQ[:, :, 1] + rev_RGB2YIQ_MATRIX[2][
        2] * imYIQ[:, :, 2]

    return np.stack((R, G, B), axis=-1)


def histogram_equalization(im_orig):
    """

    :param im_orig: grey scale or RGB normalized to [0,1]
    :return: [im_eq, hist_orig, hist_eq] where
              im_eq - is the equalized image. grayscale or RGB float64 image with values in [0, 1].
              hist_orig - is a 256 bin histogram of the original image (array with shape (256,) ).
              hist_eq - is a 256 bin histogram of the equalized image (array with shape (256,) )
    """

    # make YIQ to work Y with instade of RGB
    img = 0
    if type(im_orig[0][0]) == tuple:
        img = rgb2yiq(im_orig)
    else:
        img = im_orig

    # prepare img
    img255 = np.round(img * 255)

    # calc histogram and cumulative
    hist = np.histogram(img255, bins=range(COLOR_RANGE + 1))
    cumulative = np.cumsum(hist[0])

    # Edge case - empty img
    if cumulative[-1] == 0:
        return [im_orig, im_orig, im_orig]

    # find location of first color volume that is not zero
    first_col = np.nonzero(hist[0])[0][0]
    if len(np.nonzero(hist[0])[0]) == 1:
        return [im_orig, im_orig, im_orig]

        # if 1 - there is only one col,
        # (cumulative[COLOR_RANGE-1]-cumulative[first_col]) = div by 0
    colormap = np.round((COLOR_RANGE - 1) * (cumulative - cumulative[first_col]) / (
                  cumulative[COLOR_RANGE - 1] - cumulative[first_col]))

    """ to further understand how the image looks:
    # plot the histogram
    plt.plot(np.linspace(0, COLOR_RANGE - 1, COLOR_RANGE), hist[0])
    plt.show()

    # plot the cumulative
    plt.plot(np.linspace(0, COLOR_RANGE - 1, COLOR_RANGE), cumulative)
    plt.show()
    """


    new_img = np.zeros(COLOR_RANGE)
    # TODO

    # return [im_wq, hist_orgi, histogram_eq]


"""testing\playground field"""

x = np.hstack([np.repeat(np.arange(2, 52, 2), 10)[None, :], np.array([255] * 6)[None, :]])
grad = np.tile(x, (256, 1))  # gard3 = np.array([grad, grad * 10, grad * 20])/120
# #
# #
# img_url = 'jerusalem.jpg'
# jer = read_image(img_url)  # check q2 and 1 without wired cases

plt.imshow(grad)
plt.show()

histogram_equalization(grad / 255)
print("here")
