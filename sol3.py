import imageio
from skimage import color
import matplotlib.pyplot as plt
import scipy.signal
import scipy.linalg as lin
import numpy as np
import os

MAX_COLOR = 255


def relpath(filename):
    return os.path.join(os.path.dirname(__file__), filename)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  Helper functions ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def create_filter(filter_size):
    """
    create 2D filter_size x filter_size array
    :param filter_size: int
    :return:
    """
    temp = lin.pascal(filter_size, kind='lower')
    filter = np.zeros(temp.shape)
    filter[np.int32(filter_size / 2)] = temp[-1]
    return filter / np.sum(temp[-1])


# make shape(even_number,even_number)
def up_sample(img, axis):
    if axis: return np.append(img, np.zeros((np.shape(img)[0], 1)), axis=1)
    return np.append(img, np.zeros((1, np.shape(img)[1])), axis=0)


# helpers of build_..._pyramid
def expand(image, filter):
    """
    expand by 2 the image, using filler as a filler for the blank rows
    :param image:
    :param filter:
    :return:
    """
    result = np.zeros((image.shape[0] * 2, image.shape[1] * 2))

    result[::2, ::2] = image

    result = scipy.signal.convolve2d(result, 2 * filter.T, mode='same')
    result = scipy.signal.convolve2d(result, 2 * filter, mode='same')
    return result

# make th image with shape%2=0, then reduce. expand wont change the dim result in
def reduce(image, filter):
    """
    reduce the size of an image by factor of 2, using filter to blur before the sampling
    :param image:
    :param filter:
    :return:
    """
    if image.shape[0] % 2: image = up_sample(image, 0)
    if image.shape[1] % 2: image = up_sample(image, 1)
    filter = np.outer(filter, filter)
    result = scipy.signal.convolve2d(image, filter, mode='same')
    return result[::2, ::2]




# make shape(even_number,even_number)
def equalize_dim(filter_vec, target_img, result):
    expandedImg = expand(result, filter_vec)
    # fic uneven expention (1->2: 418/2 = 209, 2->3: 209/2 = 105, 3->2: 105*2 = 210)
    if target_img.shape[0] == expandedImg.shape[0] - 1:
        expandedImg = expandedImg[:expandedImg.shape[0] - 1, :]
    if target_img.shape[1] == expandedImg.shape[1] - 1:
        expandedImg = expandedImg[:, :expandedImg.shape[1] - 1]
    return expandedImg


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~   End Of Helpers   ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


# Q1 code
def build_gaussian_pyramid(im, max_levels, filter_size):
    """
    creates a gaussian pyramid

    :param im:– a grayscale image with double values in [0, 1] (e.g. the output of ex1’s read_image with the
                representation set to 1).
    :param max_levels:the maximal number of levels (including the original level) in the resulting pyramid.
    :param filter_size: the size of the Gaussian filter (an odd scalar that represents a squared filter) to be used
           in constructing the pyramid filter (e.g for filter_size = 3 you should get [0.25, 0.5, 0.25]). You may
           assume the filter size will be >=2

    :return: pyr - Both functions should output the resulting pyramid pyr as a standard python array, grey scale, up
                   to max_levels levels
             filter_vec -  The functions should also output filter_vec which is row vector of shape (1, filter_size) used
             for the pyramid construction. This filter should be built using a consequent 1D convolutions of [1
             1] with itself in order to derive a row of the binomial coefficients which is a good approximation to
             the Gaussian profile. The filter_vec should be normalized.
    """
    minimal_len_img = 32

    filter = create_filter(filter_size)
    pyr = list()
    pyr.append(im)
    for i in range(1, max_levels):
        if pyr[-1].shape[0] >= minimal_len_img and pyr[-1].shape[1] >= minimal_len_img:
            pyr.append(reduce(pyr[-1], filter))

    formated_filter = filter[np.int32(filter_size / 2)]

    return pyr, np.array(formated_filter).reshape(1, len(formated_filter))


def build_laplacian_pyramid(im, max_levels, filter_size):
    """
    creates a laplacian pyramid

    :param im:– a grayscale image with double values in [0, 1] (e.g. the output of ex1’s read_image with the
                representation set to 1).
    :param max_levels:the maximal number of levels (including the original level) in the resulting pyramid.
    :param filter_size: the size of the Gaussian filter (an odd scalar that represents a squared filter) to be used
           in constructing the pyramid filter (e.g for filter_size = 3 you should get [0.25, 0.5, 0.25]). You may
           assume the filter size will be >=2

    :return: pyr - Both functions should output the resulting pyramid pyr as a standard python array, grey scale, up
                   to max_levels levels
             filter_vec -  The functions should also output filter_vec which is row vector of shape (1, filter_size) used
             for the pyramid construction. This filter should be built using a consequent 1D convolutions of [1
             1] with itself in order to derive a row of the binomial coefficients which is a good approximation to
             the Gaussian profile. The filter_vec should be normalized.:
    """
    gaussian, filter = build_gaussian_pyramid(im, max_levels, filter_size)
    laplasian_pyr = list()

    for i in range(1, len(gaussian)):  # can't be vectorized as its continuous process
        x = gaussian[i - 1]
        # expandedImg = equalize_dim(create_filter(len(filter)), x, gaussian[i])
        laplasian_pyr.append(x - expand(gaussian[i],create_filter(filter_size))[:x.shape[0],:x.shape[1]])  # expand will result in
        # even dim, bigger or equal to source, if needed odd shape, we will down sample.

    laplasian_pyr.append(gaussian[-1])

    return laplasian_pyr, filter


# Q2 code
def laplacian_to_image(lpyr, filter_vec, coeff):
    """

    :param lpyr: laplacian pyramid
    :param filter_vec: the filer used to calc the laplacian
    :param coeff:python list. The list length is the same as the number of levels in the pyramid lpyr.
                 Before reconstructing the image img you should multiply each level i of the laplacian pyramid by
                 its corresponding coefficient coeff[i].

    :return:
    """
    # idea - take min level, expand, and laplacian, then repeat as if the result is the min level

    result = lpyr[-1] * coeff[-1]
    for i in range(1, len(lpyr)):
        expandedImg = equalize_dim(create_filter(len(filter_vec)), lpyr[-(i + 1)], result)
        result = expandedImg + lpyr[-(i + 1)] * coeff[-(i + 1)]

    return result + lpyr[0]


# Q3 Code
def stretch_img(img):
    img = img - np.min(img)
    if np.max(img):
        img = img / np.max(img)
    return img


def render_pyramid(pyr, levels):
    """

    :param pyr:
    :param levels:
    :return: res: is a single black image in which the pyramid levels of the
                  given pyramid pyr are stacked horizontally (after stretching the values to [0, 1])
    """

    res = stretch_img(pyr[0])

    for i, img in enumerate(pyr[1:]):
        if i == levels - 1:
            break
        # organize the image
        img = stretch_img(img)
        # make the big picture
        res = np.concatenate((res, np.pad(img, ((0, pyr[0].shape[0] - img.shape[0]), (0, 0)), mode='constant')), axis=1)

    return res


def display_pyramid(pyr, levels):
    plt.imshow(render_pyramid(pyr, levels), cmap='gray')



# Q4 Code
def pyramid_blending(im1, im2, mask, max_levels, filter_size_im, filter_size_mask):
    """
    :param im1: im1, im2 – are two input grayscale images to be blended.
    :param im2: im1, im2 – are two input grayscale images to be blended.
    :param mask: mask – is a boolean (i.e. dtype == np.bool) mask containing True and False representing which parts
        of im1 and im2 should appear in the resulting im_blend. Note that a value of True corresponds to 1,
        and False corresponds to 0.
    :param max_levels:max_levels – is the max_levels parameter you should use when generating the Gaussian and Laplacian
        pyramids.
    :param filter_size_im:filter_size_im – is the size of the Gaussian filter (an odd scalar that represents a squared filter) which
        defining the filter used in the construction of the Laplacian pyramids of im1 and im2.
    :param filter_size_mask:filter_size_mask – is the size of the Gaussian filter(an odd scalar that represents a squared filter) which
        defining the filter used in the construction of the Gaussian pyramid of mask.
    :return im_blend - the blended image
    """

    laplacian_im1, filter = build_laplacian_pyramid(im1, max_levels, filter_size_im)
    laplacian_im2, filter = build_laplacian_pyramid(im2, max_levels, filter_size_im)
    gaussian_mask_pyr, filter2 = build_gaussian_pyramid(mask.astype(float), max_levels, filter_size_mask)
    lout = list()

    for level in range(min(max_levels, len(laplacian_im1), len(laplacian_im2))):
        lout.append(
            gaussian_mask_pyr[level] * laplacian_im1[level] + (1 - gaussian_mask_pyr[level]) * laplacian_im2[level])

    res = laplacian_to_image(lout, create_filter(filter_size_im), np.ones(len(lout)))

    # clipping out of bounds values
    res[res < 0] = 0
    res[res > 1] = 1

    return res


# Q 4.1 Code
def blending_example_facade(mask1, im1, im2, result_name):
    mask = read_image(relpath(mask1), 1)
    #  make sure that we are having a binary img
    temp = np.zeros(mask.shape)
    mask = mask == 1

    im1 = read_image(relpath(im1), 1)
    im2 = read_image(relpath(im2), 1)

    res = pyramid_blending(im1, im2, mask, 3, 3, 3)

    imageio.imwrite(relpath(result_name + '.jpg'), np.uint8(res * 255), format='jpg')

    return im1, im2, mask, res


def blending_example1():
    return blending_example_facade(r'externals/mask.jpg', r'externals/katan.jpg', r'externals/nach.jpg',
                                   r'externals/Nachliely_Katan')


def blending_example2():
    return blending_example_facade(r'externals/mask2.jpg', r'externals/cpn_jack.jpg', r'externals/OppTheatrics.jpg',
                                   'externals/JackTheatrics')


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Additional stuff from previous exercises  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""histogram equalization and its helper functions"""


def histogram_equalize(im_orig):
    """
    Apply the histogram_equalization
    :param im_orig: grey scale or RGB normalized to [0,1]
    :return: [im_eq, hist_orig, hist_eq] where
              im_eq - is the equalized image. grayscale or RGB float64 image with values in [0, 1].
              hist_orig - is a 256 bin histogram of the original image (array with shape (256,) ).
              hist_eq - is a 256 bin histogram of the equalized image (array with shape (256,) )
    """

    # working on [0,255] img
    if 0 <= np.max(im_orig) <= 1: img255 = np.int64(np.floor(im_orig * 255))
    else: img255 = im_orig

    # calc histogram and cumulative
    hist_orig = np.histogram(img255, bins=range(257))
    cumulative = np.cumsum(hist_orig[0])

    # Edge case - empty img
    if cumulative[-1] == 0:
        return [im_orig, im_orig, im_orig]

    # (else) find location of first color volume that is not zero
    first_col = np.nonzero(hist_orig[0])[0][0]
    if len(np.nonzero(hist_orig[0])[0]) == 1:  # Edge case - There is only one volume of color
        return [im_orig, im_orig, im_orig]

    colormap = np.round(255 * (cumulative - cumulative[first_col]) / (
            cumulative[255] - cumulative[first_col]))

    # Use colormap to transform the img.
    equalized_img = apply_color_maping_to_img(colormap, img255)

    new_hist = np.histogram(equalized_img * 255, bins=range(257))


    return equalized_img


def apply_color_maping_to_img(colormap, img255):
    """

    :param colormap: array range[0:255] of float64 (or ints) in [0:255]
    :param img255: [0,255] MxN img (Y or greyscale)
    :return:img with dims of img255, each pixel in [0,1] where each img[x]=colormap[x]=y (move volume of 34 to what is
    in cell 34 of colormap)
    """
    dim = img255.shape
    equalized_img = (colormap[(np.int64(img255)).flatten()] / 255).reshape(*dim)
    return equalized_img


def return_to_RGB_format(YIQ, equalized_img):
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


RGB2YIQ_MATRIX = np.array([[0.299, 0.587, 0.114],
                           [0.596, -0.275, -0.321],
                           [0.212, -0.523, 0.311]])


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


# the other way around
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


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ read_image_from_ex1 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def read_image(filename, representation=2):
    """
    :param filename: the filename of an image on disk (could be grayscale or RGB).
    :param representation: representation code, either 1 or 2 defining whether the output should be a grayscale
                image (1) or an RGB image (2)
    :return: This function returns an image, make sure the output image is represented by a matrix of type
             np.float64 with intensities (either grayscale or RGB channel intensities) normalized to the range [0, 1].
    """
    img = imageio.imread(filename, pilmode='RGB')
    # img = imao.imread(filename)
    if representation == 1:
        img = color.rgb2gray(img)
    else:
        img = img / 256
    return np.float64(img)


def imdisplay(filename, representation=2):
    """
    This function to utilize read_image to display an image in a given representation. The function interface is:
    where filename and representation are the same as those defined in read_image’s interface. T
    :param filename:
    :param representation:
    :return:
    """
    img = read_image(filename, representation)
    # img = -1*(1 - img)  # convert the image
    if representation == 1:
        plt.imshow(img, 'gray')
    else:
        plt.imshow(img, 'rgb')
    plt.show()


if __name__ == '__main__':
    blending_example1()
#     im_orig = read_image('externals/monkey.jpg', 1)
#     gpyr, filter_vec = build_laplacian_pyramid(im_orig, 3, 3)
#     display_pyramid(gpyr,3)
#     print('hell')

#     src = r'externals/nach.jpg'
#     img = read_image(src, 1)
#     # res = build_gaussian_pyramid(read_image(src, 1), 3, 3)[0]
#
#     """test of laplacian pyramid look - not ok"""
#     res = build_laplacian_pyramid(read_image(src, 1), 3, 3)[0]
#     plt.imshow(render_pyramid(res,3),cmap='gray')
#     # for i in range(len(res)):
#     #     x = res[i]-np.min(res[i])
#     #     x = 255*x//np.max(x)
#     #     plt.imshow(x,cmap='gray')
#         # x = input("")
#
#     """ test of dimensions when expanding - GOOD"""
#     # img = img[:, :1023]
#     # # build_gaussian_pyramid(img,5,3)
#     # # filter = create_filter(3)
#     # while img.shape[0] > 32:
#     #     print(img.shape)
#     #     img = reduce(img, create_filter(3)[1])
#     #     # plt.imshow(img,'gray')
#     # #
#     # while img.shape[0] <= 1024:
#     #     print(img.shape)
#     #     img = expand(img, create_filter(3)[1])
#     #     # plt.imshow(img, 'gray')
#
#     x = laplacian_to_image(res,create_filter(3),[1,1,1])
#
#     plt.imshow(x,cmap='gray')