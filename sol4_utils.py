import scipy
from scipy.signal import convolve2d
import scipy.signal
import scipy.linalg as lin
import numpy as np

import imageio as imao
from skimage import color


def gaussian_kernel(kernel_size):
    conv_kernel = np.array([1, 1], dtype=np.float64)[:, None]
    conv_kernel = convolve2d(conv_kernel, conv_kernel.T)
    kernel = np.array([1], dtype=np.float64)[:, None]
    for i in range(kernel_size - 1):
        kernel = convolve2d(kernel, conv_kernel, 'full')
    return kernel / kernel.sum()


def blur_spatial(img, kernel_size):
    kernel = gaussian_kernel(kernel_size)
    blur_img = np.zeros_like(img)
    if len(img.shape) == 2:
        blur_img = convolve2d(img, kernel, 'same', 'symm')
    else:
        for i in range(3):
            blur_img[..., i] = convolve2d(img[..., i], kernel, 'same', 'symm')
    return blur_img


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~   Stuff from ex3   ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


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

def reduce(image, filter):
    """
    reduce the size of an image by factor of 2, using filter to blur before the sampling
    :param image:
    :param filter:
    :return:
    """
    filter = np.outer(filter, filter)
    result = scipy.signal.convolve2d(image, filter, mode='same')
    image = result[::2, ::2]
    return image

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

def read_image(filename, representation=2):
    """
    :param filename: the filename of an image on disk (could be grayscale or RGB).
    :param representation: representation code, either 1 or 2 defining whether the output should be a grayscale
                image (1) or an RGB image (2)
    :return: This function returns an image, make sure the output image is represented by a matrix of type
             np.float64 with intensities (either grayscale or RGB channel intensities) normalized to the range [0, 1].
    """
    img = imao.imread(filename)
    if representation == 1:
        img = color.rgb2gray(img)
    else:
        img = img / 256
    return np.float64(img)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~   End Of ex3 helpers   ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

