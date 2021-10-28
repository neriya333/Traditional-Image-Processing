import numpy as np
import matplotlib.pyplot as plt

""" import scikit-image as skmg """
import scipy as sp
import imageio as imao
from skimage import color

GRAY_SCALE = 1
COLOR_RANGE = 256

RGB2YIQ_MATRIX = np.array([[0.299, 0.587, 0.114],
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

    inv_RGB2YIQ_MATRIX = np.linalg.inv(RGB2YIQ_MATRIX)
    dim = imYIQ.shape

    if dim[-1] != 3: ValueError("Input array must have a shape == (..., 3)), "f"got {dim}")

    R = inv_RGB2YIQ_MATRIX[0][0] * imYIQ[:, :, 0] + inv_RGB2YIQ_MATRIX[0][1] * imYIQ[:, :, 1] + inv_RGB2YIQ_MATRIX[0][
        2] * imYIQ[:, :, 2]
    G = inv_RGB2YIQ_MATRIX[1][0] * imYIQ[:, :, 0] + inv_RGB2YIQ_MATRIX[1][1] * imYIQ[:, :, 1] + inv_RGB2YIQ_MATRIX[1][
        2] * imYIQ[:, :, 2]
    B = inv_RGB2YIQ_MATRIX[2][0] * imYIQ[:, :, 0] + inv_RGB2YIQ_MATRIX[2][1] * imYIQ[:, :, 1] + inv_RGB2YIQ_MATRIX[2][
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
    img , YIQ = 0, 0
    img_converted_to_YIQ_flag = False
    if im_orig.shape[-1] == 3 and len(im_orig.shape) == 3:
        YIQ = rgb2yiq(im_orig).T
        img = YIQ[0]
        img_converted_to_YIQ_flag = True
    else:
        img = im_orig

    # prepare img
    img255 = np.int64(np.floor(img * (COLOR_RANGE-1)))

    # calc histogram and cumulative
    hist_orig = np.histogram(img255, bins=range(COLOR_RANGE + 1))
    cumulative = np.cumsum(hist_orig[0])

    # Edge case - empty img
    if cumulative[-1] == 0:
        return [im_orig, im_orig, im_orig]

    # (else) find location of first color volume that is not zero
    first_col = np.nonzero(hist_orig[0])[0][0]
    if len(np.nonzero(hist_orig[0])[0]) == 1:  # if there is only one volume of color, return it as is
        return [im_orig, im_orig, im_orig]

    plt.plot(np.linspace(0, COLOR_RANGE - 1, COLOR_RANGE), hist_orig[0])
    plt.show()

    colormap = np.round((COLOR_RANGE - 1) * (cumulative - cumulative[first_col]) / (
            cumulative[COLOR_RANGE - 1] - cumulative[first_col]))

    # find shape of img, use colormap to transform the flatten img then normalize the img and rebuild it to its dim
    dim = img.shape
    equalized_img = (colormap[img255.flatten()] / (COLOR_RANGE-1)).reshape(*dim)

    new_hist = np.histogram(equalized_img * (COLOR_RANGE-1), bins=range(COLOR_RANGE + 1))

    plt.plot(np.linspace(0, COLOR_RANGE - 1, COLOR_RANGE), new_hist[0])
    plt.show()

    # convert back to RGB
    if img_converted_to_YIQ_flag:
        YIQ[0] = equalized_img
        equalized_img = yiq2rgb(YIQ.T)

    return [equalized_img, hist_orig[0], new_hist[0]]


def quantize(im_orig, n_quant, n_iter):
    """

    :param im_orig: - is the input grayscale or RGB image to be quantized (float64 image with values in [0, 1])
    :param n_quant: - is the number of intensities your output im_quant image should have.
    :param n_iter:r - is the maximum number of iterations of the optimization procedure (may converge earlier.)

    :return:
    """
    z_arr = np.array(n_quant+1)
    p_arr = np.array(COLOR_RANGE)


    hist = np.histogram(im_orig, bins=COLOR_RANGE+1)
    z_arr[0], z_arr[-1] = 0, COLOR_RANGE
    # if z_arr[0]




    print('hi')


    # return [im_quant, error]


"""testing\playground field"""

x = np.hstack([np.repeat(np.arange(2, 52, 2), 10)[None, :], np.array([255] * 6)[None, :]])
grad = np.tile(x, (256, 1))  # gard3 = np.array([grad, grad * 10, grad * 20])/120
grad = np.hstack([grad, grad, grad])


# #
# #
img_url = 'jerusalem.jpg'
jer = read_image(img_url)  # check q2 and 1 without wired cases
#
# plt.imshow(grad)
# plt.show()

""" to further understand how the image looks:
# plot the histogram
plt.plot(np.linspace(0, COLOR_RANGE - 1, COLOR_RANGE), hist[0])
plt.show()

# plot the cumulative
plt.plot(np.linspace(0, COLOR_RANGE - 1, COLOR_RANGE), cumulative)
plt.show()
"""

# plt.imshow(jer)
# plt.show()
#
# endgame = histogram_equalization(jer)
#
# plt.imshow(endgame[0])
# plt.show()








print("here")
