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
    YIQ, img, img_converted_to_YIQ_flag = img_to_grey_or_yiq(im_orig)

    # prepare img

    img255 = np.int64(np.floor(img * (COLOR_RANGE - 1)))

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

    colormap = np.round((COLOR_RANGE - 1) * (cumulative - cumulative[first_col]) / (
            cumulative[COLOR_RANGE - 1] - cumulative[first_col]))

    # find shape of img, use colormap to transform the flatten img then normalize the img and rebuild it to its dim
    equalized_img = apply_colormaping_to_img(colormap, img255)

    new_hist = np.histogram(equalized_img * (COLOR_RANGE - 1), bins=range(COLOR_RANGE + 1))

    # convert back to RGB
    if img_converted_to_YIQ_flag:
        equalized_img = retun_to_RGB_format(YIQ, equalized_img)

    return [equalized_img, hist_orig[0], new_hist[0]]


def apply_colormaping_to_img(colormap, img255):
    dim = img255.shape
    equalized_img = (colormap[img255.flatten()] / (COLOR_RANGE - 1)).reshape(*dim)
    return equalized_img


def retun_to_RGB_format(YIQ, equalized_img):
    YIQ[0] = equalized_img
    return yiq2rgb(YIQ.T)


def img_to_grey_or_yiq(im_orig):
    """
    return a NxM metrics of intensities, grey scale or Y of YIQ, and mark if it made an YIQ convertion
    :param im_orig:
    :return:
    """
    YIQ = 0
    img_converted_to_YIQ_flag = False
    if im_orig.shape[-1] == 3 and len(im_orig.shape) == 3:
        YIQ = rgb2yiq(im_orig).T
        img = YIQ[0]
        img_converted_to_YIQ_flag = True
    else:
        img = im_orig
    return YIQ, img, img_converted_to_YIQ_flag


def quantize(im_orig, n_quant, n_iter):
    """
    :param im_orig: - is the input grayscale or RGB image to be quantized (float64 image with values in [0, 1])
    :param n_quant: - is the number of intensities your output im_quant image should have.
    :param n_iter:r - is the maximum number of iterations of the optimization procedure (may converge earlier.)

    :return:
    """

    YIQ, img, img_converted_to_YIQ_flag = img_to_grey_or_yiq(im_orig)
    hist = np.histogram(img, bins=COLOR_RANGE)

    # place z
    z_ = split_z_evenly_by_distribution_of_colors(hist, n_quant)
    z_[0], z_[-1] = 0, COLOR_RANGE

    # place p
    q_loc = np.zeros(n_quant)
    for i in range(n_quant):
        q_loc[i] = round((z_[i] + z_[i + 1]) / 2)

    error_iter = np.zeros(n_iter)
    length_arr = np.array(range(COLOR_RANGE))

    # iter over q, z and cal the err
    for j in range(n_iter):
        for i, q in enumerate(q_loc):
            numerator = np.dot(hist[0][z_[i]:z_[i + 1]], length_arr[z_[i]:z_[i + 1]])  # takes this part of the vector
            q_loc[i] = np.round(numerator / np.sum(hist[0][z_[i]:z_[i + 1]]))

        for i in range(1, n_quant, 1):
            z_[i] = round((q_loc[i-1] + q_loc[i])/2)

        # err = sum of (delta(p_i),base_volume)^2 * num_pix_this_volume.
        # here we made a vector multiplication using (volume_VEC-p_i)^2 * hist
        q_vec = create_Id_vec_minos_p__vec_per_interval_Zi_to_Zi_plus1(np.array(range(COLOR_RANGE)), q_loc, z_)
        error_iter[j] += np.dot(q_vec**2, hist[0])

    # make 255 colormap
    colormap = np.zeros(COLOR_RANGE)
    for i in range(n_quant):
        colormap[z_[i]:z_[i+1]] = q_loc[i]

    img = apply_colormaping_to_img(colormap, img)

    if img_converted_to_YIQ_flag:
        img = retun_to_RGB_format(YIQ, img)

    return [img, error_iter]


def split_z_evenly_by_distribution_of_colors(hist, n_quant):
    """
    find where splits the histogram to n_quant has mostly equal pixels in each segment
    :param hist:
    :param n_quant:
    :return: np.array - where to place z
    """
    cumu_hist = np.cumsum(hist[0])
    avg_pix_per_Z = int(cumu_hist[-1] / n_quant)
    evenly_split_sum = np.int64(np.linspace(0, cumu_hist[-1], n_quant + 1))
    return np.searchsorted(np.cumsum(hist[0]), evenly_split_sum)


def create_Id_vec_minos_p__vec_per_interval_Zi_to_Zi_plus1(length_arr, q_loc, z_):
    """help compute vec with values :[0..255] minus q_i per interval [z[i], z_[i+1]].
    use to compute Error value"""
    for i in range(len(q_loc)):
        length_arr[z_[i]:z_[i+1]] -= np.int32(q_loc[i])
    length_arr[-1] -= q_loc[-1]
    return length_arr