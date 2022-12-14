# Initial code for ex4.
# You may change this code, but keep the functions' signatures
# You can also split the code to multiple files as long as this file's API is unchanged

import numpy as np
import os
import matplotlib.pyplot as plt
import scipy.signal
from scipy.signal import convolve2d

from scipy.ndimage.morphology import generate_binary_structure
from scipy.ndimage.filters import maximum_filter
from scipy.ndimage import label, center_of_mass
import shutil
from imageio import imwrite
import imageio

import sol4_utils


# harris helper
def createR(dx2_blur, dy2_blur, dxdy_blur):
    """
    M = [[dx^2, dxdy]
         [dydx, dy^2]]
                        now originaly i thought that the right way is to sum each item on the
    I thought that the right thing to do is to run create a matrix of the surrounding
    :param dx2_blur:
    :param dy2_blur:
    :param dxdy_blur:
    :return:
    """
    k = 0.04

    detMatrix = dx2_blur * dy2_blur - dxdy_blur * dxdy_blur
    traceMatrix = dx2_blur + dy2_blur
    traceMatrix2 = (traceMatrix * traceMatrix) * k  # matrix*matrix multiplied by scalar

    return detMatrix - traceMatrix2
    # represent how high is the value of the eigen vector
    # (which in turn represents how much change a given pixel has,compare to its surroundings)


def harris_corner_detector(im):
    """
  Detects harris corners.
  Make sure the returned coordinates are x major!!!
  :param im: A 2D array representing an image.
  :return: An array with shape (N,2), where ret[i,:] are the [x,y] coordinates of the ith corner points.
  """
    if len(im.shape) != 2:
        Exception(ValueError("Entered non Grey-Scale Img"))
        return
    conv_X = np.array(([0, 0, 0], [1, 0, -1], [0, 0, 0]))
    dx, dy = convolve2d(im, conv_X, mode='same'), convolve2d(im, conv_X.T, mode='same')

    dx2_blur = sol4_utils.blur_spatial(dx * dx, 3)
    dy2_blur = sol4_utils.blur_spatial(dy * dy, 3)
    dxdy_blur = sol4_utils.blur_spatial(dx * dy, 3)

    R_matrix = createR(dx2_blur, dy2_blur, dxdy_blur)

    image_mapped_local_maxima = non_maximum_suppression(R_matrix)
    yx = np.nonzero(image_mapped_local_maxima)[:2]  # save indices where value ain't 0
    return np.vstack((yx[1], yx[0]))  # return ((x,y),) for x[i],y[i] in X,Y


def sample_descriptor(im, pos, desc_rad):
    """
    Samples descriptors at the given corners.
    :param im: A 2D array representing an image.(3ed lvl of gaussian)
    :param pos: An array with shape (N,2), where pos[i,:] are the [x,y] coordinates of the ith corner point.
    :param desc_rad: "Radius" of descriptors to compute.
    :return: A 3D array with shape (N,K,K) containing the ith descriptor at desc[i,:,:].
    """
    K = desc_rad * 2 + 1
    # creates a K^2,K^2 indices as x,y such that (0,0),(0,1)..(0,7)
    #                                             .              .
    #                                             .              .
    #                                             .              .
    #                                            (7,0),(7,1)..(7,7)
    # will be the result of [x,y].
    x = np.repeat(np.linspace(-desc_rad, desc_rad + 1, desc_rad * 2 + 1), desc_rad * 2 + 1)  # coordinate of x
    y = np.tile(np.linspace(-desc_rad, desc_rad + 1, desc_rad * 2 + 1), desc_rad * 2 + 1)  # coordinate of y

    # initiate the array
    descriptor_vec = np.zeros((len(pos), K, K))
    # fill vector vector for each coordinate c.
    # note that 'c_x','c_y' must be added to x,y to move the window to the right place
    for i in range(len(pos)):
        map_coordinates = x + pos[i][0], y + pos[i][1]
        temp = np.reshape(scipy.ndimage.map_coordinates(im, map_coordinates, order=1), (K, K))

        if np.linalg.norm != 0:
            temp = (temp - np.mean(temp)) / np.linalg.norm(temp - np.mean(temp))
        else:
            temp *= 0

        descriptor_vec[i] = temp
    return descriptor_vec


def find_features(pyr):
    """
    Detects and extracts feature points from a pyramid.
    :param pyr: Gaussian pyramid of a grayscale image having 3 levels.
    :return: A list containing:
              1) An array with shape (N,2) of [x,y] feature location per row found in the image.
                 These coordinates are provided at the pyramid level pyr[0].
              2) A feature descriptor array with shape (N,K,K)
    """
    m = n = 7
    radius = 3  # 3, central_pixel,3 =>7
    decrease_per_level = number_of_levels = 2
    pos = spread_out_corners(pyr[0], m, n, radius)

    return pos, sample_descriptor(pyr[0][2], pos * (decrease_per_level ** -number_of_levels), radius)


def match_features(desc1, desc2, min_score):
    """
    Return indices of matching descriptors.
    :param desc1: A feature descriptor array with shape (N1,K,K).
    :param desc2: A feature descriptor array with shape (N2,K,K).
    :param min_score: Minimal match score.
    :return: A list containing:
              1) An array with shape (M,) and dtype int of matching indices in desc1.
              2) An array with shape (M,) and dtype int of matching indices in desc2.
    """
    # todo - more efficient (using reshape(x,1), and matrix multiplication)
    # desc1 = T.reshape((:,1,desc1.T.shape[2]))
    # desc2 = T.reshape((:,1,desc2.T.shape[2]))

    D = np.zeros((desc1.shape[0], desc2.shape[0]))
    for d1 in desc1.shape[0]:
        for d2 in desc2.shape[0]:
            D[d1, d2] = desc1[d1,:,:].dot(desc2[d2,:,:])

    res = list()
    loc1,loc2 = list(),list()
    # the same - improve efficiency, make it numpy-ish
    for i, row in enumerate(D):
        if np.max(row) < min_score:
            continue
        p = np.where(np.max(row) == row)[0]  # find inx (col) of max value
        for col_inx_max_val in p[0]:  # for each of the col that had max_value of row in them
            if col_inx_max_val in np.where(np.max(D[:, col_inx_max_val]) == D[:, col_inx_max_val]): # if its max value == row.max_value
                # max value of col3==max value of row 2 => (2,3) holds max value. pick it
                loc1.append(i)
                loc2.append(col_inx_max_val)

    res.append(np.array(loc1))
    res.append(np.array(loc2))
    return res


def apply_homography(pos1, H12):
    """
    Apply homography to inhomogenous points.
    :param pos1: An array with shape (N,2) of [x,y] point coordinates.
    :param H12: A 3x3 homography matrix.
    :return: An array with the same shape as pos1 with [x,y] point coordinates obtained from transforming pos1 using H12.
    """
    multiplied = H12 @ (np.vstack((pos1, np.ones(len(pos1)))))
    return multiplied[:2] / multiplied[2:]  # divide


def ransac_homography(points1, points2, num_iter, inlier_tol, translation_only=False):
    """
    Computes homography between two sets of points using RANSAC.
    :param pos1: An array with shape (N,2) containing N rows of [x,y] coordinates of matched points in image 1.
    :param pos2: An array with shape (N,2) containing N rows of [x,y] coordinates of matched points in image 2.
    :param num_iter: Number of RANSAC iterations to perform.
    :param inlier_tol: inlier tolerance threshold.
    :param translation_only: see estimate rigid transform
    :return: A list containing:
          1) A 3x3 normalized homography matrix.
          2) An Array with shape (S,) where S is the number of inliers,
              containing the indices in pos1/pos2 of the maximal set of inlier matches found.
    """
    i = 0
    max_inliers = 0
    inx_of_matching_inliers = 0
    real_H = 0

    while i < num_iter:
        # rand int
        rand_point1 = np.random.randint(0, points1.shape[0] + 1, 2)
        rand_point2 = np.random.randint(0, points2.shape[0] + 1, 2)

        # look for H
        H = estimate_rigid_transform(points1[rand_point1, :], points2[rand_point2:], translation_only)
        homographed_points1 = apply_homography(points1, H)

        # check how H is working out on the rest of the points
        diff = np.abs(points2 - homographed_points1)
        E = diff[0] * diff[0] + diff[1] * diff[1]  # (x1-x2)^2+(y1-y2)^2 =: [,,...,,,] of (M)SE
        inliers_location = np.where[points1 == points1[E < inlier_tol]]  # for values<tol, save its' indices

        # save the best result yet
        if inliers_location.shape[0] > max_inliers:
            inx_of_matching_inliers = inliers_location
            max_inliers = inliers_location.shape[0]
            real_H = H

        i += 1

    return real_H, inx_of_matching_inliers


def display_matches(im1, im2, points1, points2, inliers):
    """
    Display matching points.
    :param im1: A grayscale image.
    :param im2: A grayscale image.
    :parma pos1: An array shape (N,2), containing N rows of [x,y] coordinates of matched points in im1.
    :param pos2: An array shape (N,2), containing N rows of [x,y] coordinates of matched points in im2.
    :param inliers: An array with shape (S,) of inlier matches.
    """
    im = np.hstack((im1, im2))
    points2[:, 1] = points2[:, 1] + im1.shape[1]

    plot_inliers = np.concatenate(points1[inliers], points2[inliers])
    plot_outliers = np.concatenate(points1[~(points1[:] == points1[inliers])],
                                   points2[~(points2[:] == points2[inliers])])
    plt.imshow(im)
    plt.plot(plot_outliers[0, :], plot_outliers[1, :], color='blue', marker='o')
    plt.plot(plot_inliers[0, :], plot_inliers[1, :], color='yellow', marker='o')


def accumulate_homographies(H_succesive, m):
    """
    Convert a list of succesive homographies to a list of homographies to a common reference frame.
    :param H_successive: A list of M-1 3x3 homography matrices where H_successive[i] is a homography which transforms
           points from coordinate system i to coordinate system i+1.
    :param m: Index of the coordinate system towards which we would like to accumulate the given homographies.
    :return: A list of M 3x3 homography matrices, where H2m[i] transforms points from coordinate system i to coordinate
             system m
    """
    H_succesive[m] = np.eye((3,3))
    for i, mat in enumerate(H_succesive[m+1:]):
        H_succesive[i] = np.linalg.inv(H_succesive[i-1])@np.linalg.inv(H_succesive[i])

    for i, mat in enumerate(H_succesive[m-1:-1:-1]):
        H_succesive[i] = H_succesive[i - 1] @ H_succesive[i]

    return H_succesive


def compute_bounding_box(homography, w, h):
    """
  computes bounding box of warped image under homography, without actually warping the image
  :param homography: homography
  :param w: width of the image
  :param h: height of the image
  :return: 2x2 array, where the first row is [x,y] of the top left corner,
   and the second row is the [x,y] of the bottom right corner
  """
    pass


def warp_channel(image, homography):
    """
  Warps a 2D image with a given homography.
  :param image: a 2D image.
  :param homography: homograhpy.
  :return: A 2d warped image.
  """
    pass


""" ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Supplied Functions ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"""


def warp_image(image, homography):
    """
  Warps an RGB image with a given homography.
  :param image: an RGB image.
  :param homography: homograhpy.
  :return: A warped image.
  """
    return np.dstack([warp_channel(image[..., channel], homography) for channel in range(3)])


def filter_homographies_with_translation(homographies, minimum_right_translation):
    """
  Filters rigid transformations encoded as homographies by the amount of translation from left to right.
  :param homographies: homograhpies to filter.
  :param minimum_right_translation: amount of translation below which the transformation is discarded.
  :return: filtered homographies..
  """
    translation_over_thresh = [0]
    last = homographies[0][0, -1]
    for i in range(1, len(homographies)):
        if homographies[i][0, -1] - last > minimum_right_translation:
            translation_over_thresh.append(i)
            last = homographies[i][0, -1]
    return np.array(translation_over_thresh).astype(np.int)


def estimate_rigid_transform(points1, points2, translation_only=False):
    """
    Computes rigid transforming points1 towards points2, using least squares method.
    points1[i,:] corresponds to poins2[i,:]. In every point, the first coordinate is *x*.
    :param points1: array with shape (N,2). Holds coordinates of corresponding points from image 1.
    :param points2: array with shape (N,2). Holds coordinates of corresponding points from image 2.
    :param translation_only: whether to compute translation only. False (default) to compute rotation as well.
    :return: A 3x3 array with the computed homography.
    """
    centroid1 = points1.mean(axis=0)
    centroid2 = points2.mean(axis=0)

    if translation_only:
        rotation = np.eye(2)
        translation = centroid2 - centroid1

    else:
        centered_points1 = points1 - centroid1
        centered_points2 = points2 - centroid2

        sigma = centered_points2.T @ centered_points1
        U, _, Vt = np.linalg.svd(sigma)

        rotation = U @ Vt
        translation = -rotation @ centroid1 + centroid2

    H = np.eye(3)
    H[:2, :2] = rotation
    H[:2, 2] = translation
    return H


def non_maximum_suppression(image):
    """
  Finds local maximas of an image.
  :param image: A 2D array representing an image.
  :return: A boolean array with the same shape as the input image, where True indicates local maximum.
  """
    # Find local maximas.
    neighborhood = generate_binary_structure(2, 2)
    local_max = maximum_filter(image, footprint=neighborhood) == image
    local_max[image < (image.max() * 0.1)] = False

    # Erode areas to single points.
    lbs, num = label(local_max)
    centers = center_of_mass(local_max, lbs, np.arange(num) + 1)
    centers = np.stack(centers).round().astype(np.int64)
    ret = np.zeros_like(image, dtype=np.bool_)
    ret[centers[:, 0], centers[:, 1]] = True

    return ret


def spread_out_corners(im, m, n, radius):
    """
  Splits the image im to m by n rectangles and uses harris_corner_detector on each.
  :param im: A 2D array representing an image.
  :param m: Vertical number of rectangles.
  :param n: Horizontal number of rectangles.
  :param radius: Minimal distance of corner points from the boundary of the image.
  :return: An array with shape (N,2), where ret[i,:] are the [x,y] coordinates of the ith corner points.
  """
    corners = [np.empty((0, 2), dtype=np.int)]
    x_bound = np.linspace(0, im.shape[1], n + 1, dtype=np.int)
    y_bound = np.linspace(0, im.shape[0], m + 1, dtype=np.int)
    for i in range(n):
        for j in range(m):
            # Use Harris detector on every sub image.
            sub_im = im[y_bound[j]:y_bound[j + 1], x_bound[i]:x_bound[i + 1]]
            sub_corners = harris_corner_detector(sub_im)
            sub_corners += np.array([x_bound[i], y_bound[j]])[np.newaxis, :]
            corners.append(sub_corners)
    corners = np.vstack(corners)
    legit = ((corners[:, 0] > radius) & (corners[:, 0] < im.shape[1] - radius) &
             (corners[:, 1] > radius) & (corners[:, 1] < im.shape[0] - radius))
    ret = corners[legit, :]
    return ret


class PanoramicVideoGenerator:
    """
  Generates panorama from a set of images.
  """

    def __init__(self, data_dir, file_prefix, num_images, bonus=False):
        """
    The naming convention for a sequence of images is file_prefixN.jpg,
    where N is a running number 001, 002, 003...
    :param data_dir: path to input images.
    :param file_prefix: see above.
    :param num_images: number of images to produce the panoramas with.
    """
        self.bonus = bonus
        self.file_prefix = file_prefix
        self.files = [os.path.join(data_dir, '%s%03d.jpg' % (file_prefix, i + 1)) for i in range(num_images)]
        self.files = list(filter(os.path.exists, self.files))
        self.panoramas = None
        self.homographies = None
        print('found %d images' % len(self.files))

    def align_images(self, translation_only=False):
        """
    compute homographies between all images to a common coordinate system
    :param translation_only: see estimte_rigid_transform
    """
        # Extract feature point locations and descriptors.
        points_and_descriptors = []
        for file in self.files:
            image = sol4_utils.read_image(file, 1)
            self.h, self.w = image.shape
            pyramid, _ = sol4_utils.build_gaussian_pyramid(image, 3, 7)
            points_and_descriptors.append(find_features(pyramid))

        # Compute homographies between successive pairs of images.
        Hs = []
        for i in range(len(points_and_descriptors) - 1):
            points1, points2 = points_and_descriptors[i][0], points_and_descriptors[i + 1][0]
            desc1, desc2 = points_and_descriptors[i][1], points_and_descriptors[i + 1][1]

            # Find matching feature points.
            ind1, ind2 = match_features(desc1, desc2, .7)
            points1, points2 = points1[ind1, :], points2[ind2, :]

            # Compute homography using RANSAC.
            H12, inliers = ransac_homography(points1, points2, 100, 6, translation_only)

            # Uncomment for debugging: display inliers and outliers among matching points.
            # In the submitted code this function should be commented out!
            # display_matches(self.images[i], self.images[i+1], points1 , points2, inliers)

            Hs.append(H12)

        # Compute composite homographies from the central coordinate system.
        accumulated_homographies = accumulate_homographies(Hs, (len(Hs) - 1) // 2)
        self.homographies = np.stack(accumulated_homographies)
        self.frames_for_panoramas = filter_homographies_with_translation(self.homographies, minimum_right_translation=5)
        self.homographies = self.homographies[self.frames_for_panoramas]

        def generate_panoramic_images(self, number_of_panoramas):
            """
      combine slices from input images to panoramas.
      :param number_of_panoramas: how many different slices to take from each input image
      """
            if self.bonus:
                self.generate_panoramic_images_bonus(number_of_panoramas)
            else:
                self.generate_panoramic_images_normal(number_of_panoramas)

    def generate_panoramic_images_normal(self, number_of_panoramas):
        """
    combine slices from input images to panoramas.
    :param number_of_panoramas: how many different slices to take from each input image
    """
        assert self.homographies is not None

        # compute bounding boxes of all warped input images in the coordinate system of the middle image (as given by the homographies)
        self.bounding_boxes = np.zeros((self.frames_for_panoramas.size, 2, 2))
        for i in range(self.frames_for_panoramas.size):
            self.bounding_boxes[i] = compute_bounding_box(self.homographies[i], self.w, self.h)

        # change our reference coordinate system to the panoramas
        # all panoramas share the same coordinate system
        global_offset = np.min(self.bounding_boxes, axis=(0, 1))
        self.bounding_boxes -= global_offset

        slice_centers = np.linspace(0, self.w, number_of_panoramas + 2, endpoint=True, dtype=np.int64)[1:-1]
        warped_slice_centers = np.zeros((number_of_panoramas, self.frames_for_panoramas.size))
        # every slice is a different panorama, it indicates the slices of the input images from which the panorama
        # will be concatenated
        for i in range(slice_centers.size):
            slice_center_2d = np.array([slice_centers[i], self.h // 2])[None, :]
            # homography warps the slice center to the coordinate system of the middle image
            warped_centers = [apply_homography(slice_center_2d, h) for h in self.homographies]
            # we are actually only interested in the x coordinate of each slice center in the panoramas' coordinate system
            warped_slice_centers[i] = np.array(warped_centers)[:, :, 0].squeeze() - global_offset[0]

        panorama_size = np.max(self.bounding_boxes, axis=(0, 1)).astype(np.int64) + 1

        # boundary between input images in the panorama
        x_strip_boundary = ((warped_slice_centers[:, :-1] + warped_slice_centers[:, 1:]) / 2)
        x_strip_boundary = np.hstack([np.zeros((number_of_panoramas, 1)),
                                      x_strip_boundary,
                                      np.ones((number_of_panoramas, 1)) * panorama_size[0]])
        x_strip_boundary = x_strip_boundary.round().astype(np.int64)

        self.panoramas = np.zeros((number_of_panoramas, panorama_size[1], panorama_size[0], 3), dtype=np.float64)
        for i, frame_index in enumerate(self.frames_for_panoramas):
            # warp every input image once, and populate all panoramas
            image = sol4_utils.read_image(self.files[frame_index], 2)
            warped_image = warp_image(image, self.homographies[i])
            x_offset, y_offset = self.bounding_boxes[i][0].astype(np.int64)
            y_bottom = y_offset + warped_image.shape[0]

            for panorama_index in range(number_of_panoramas):
                # take strip of warped image and paste to current panorama
                boundaries = x_strip_boundary[panorama_index, i:i + 2]
                image_strip = warped_image[:, boundaries[0] - x_offset: boundaries[1] - x_offset]
                x_end = boundaries[0] + image_strip.shape[1]
                self.panoramas[panorama_index, y_offset:y_bottom, boundaries[0]:x_end] = image_strip

        # crop out areas not recorded from enough angles
        # assert will fail if there is overlap in field of view between the left most image and the right most image
        crop_left = int(self.bounding_boxes[0][1, 0])
        crop_right = int(self.bounding_boxes[-1][0, 0])
        assert crop_left < crop_right, 'for testing your code with a few images do not crop.'
        print(crop_left, crop_right)
        self.panoramas = self.panoramas[:, :, crop_left:crop_right, :]

    def generate_panoramic_images_bonus(self, number_of_panoramas):
        """
    The bonus
    :param number_of_panoramas: how many different slices to take from each input image
    """
        pass

    def save_panoramas_to_video(self):
        assert self.panoramas is not None
        out_folder = 'tmp_folder_for_panoramic_frames/%s' % self.file_prefix
        try:
            shutil.rmtree(out_folder)
        except:
            print('could not remove folder')
            pass
        os.makedirs(out_folder)
        # save individual panorama images to 'tmp_folder_for_panoramic_frames'
        for i, panorama in enumerate(self.panoramas):
            imwrite('%s/panorama%02d.png' % (out_folder, i + 1), panorama)
        if os.path.exists('%s.mp4' % self.file_prefix):
            os.remove('%s.mp4' % self.file_prefix)
        # write output video to current folder
        os.system('ffmpeg -framerate 3 -i %s/panorama%%02d.png %s.mp4' %
                  (out_folder, self.file_prefix))

    def show_panorama(self, panorama_index, figsize=(20, 20)):
        assert self.panoramas is not None
        plt.figure(figsize=figsize)
        plt.imshow(self.panoramas[panorama_index].clip(0, 1))
        plt.show()


""" ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Supplied Functions ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"""


def test_harrisCornerDetection():
    """look at it and make sure its looking like corners should look like"""

    path = r"C:\Users\neriy\OneDrive\Documents\GitHub\ImageProccessing\ex4-neriya333\ex4-impr-supplementary material\external\oxford1.jpg"
    im = sol4_utils.read_image(path, 1)
    m = n = 7

    result = harris_corner_detector(im)
    x, y = np.zeros(len(result), dtype=np.int64), np.zeros(len(result), dtype=np.int64)
    for i, item in enumerate(result):
        x[i], y[i] = item[0], item[1]
    # input("run this line by line and see that u get image and logically looking corners")
    plt.imshow(im)
    linestyle = tuple((0, tuple((0, 100000))))

    plt.plot(result[0, :], result[1, :], color='red', linestyle=linestyle, marker='.')


def test_spreadOutDetection():
    """look at it and make sure its looking like corners should look like"""

    path = "external\\oxford1.jpg"
    im = sol4_utils.read_image(path, 1)
    m = n = 7

    pyr = sol4_utils.build_gaussian_pyramid(im, 3, 3)
    vectors = find_features(pyr)

    x, y = np.zeros(len(pyr), dtype=np.int64), np.zeros(len(pyr), dtype=np.int64)
    for i, item in enumerate(pyr):
        x[i], y[i] = item[0], item[1]
    # input("run this line by line and see that u get image and logically looking corners")
    plt.imshow(im[2])
    linestyle = tuple((0, tuple((0, 100000))))

    plt.plot(x, y, color='red', linestyle=linestyle, marker='.')


if __name__ == '__main__':
    test_harrisCornerDetection()

    # path1 = "external\\oxford1.jpg"
    # path2 = "external\\oxford2.jpg"
    #
    # im1 = sol4_utils.read_image(path1, 1)
    # im2 = sol4_utils.read_image(path2, 1)
    # m = n = 7
    #
    # # find corners
    # corner1 = spread_out_corners(im1, m, n, 3)
    # corner2 = spread_out_corners(im2, m, n, 3)
    #
    # # get descriptors
    # desc1 = sample_descriptor(sol4_utils.build_gaussian_pyramid(im1, 3, 3)[0][2], corner1, 3)
    # desc2 = sample_descriptor(sol4_utils.build_gaussian_pyramid(im2, 3, 3)[0][2], corner2, 3)
    #
    # matching_features = match_features(desc1,desc2,0.5)


    # path = "external\\oxford1.jpg"
    # im = sol4_utils.read_image(path, 1)
    # im_max = np.argmax(im)
    # corners = spread_out_corners(im, 7, 7, 3)

    # x, y = np.zeros(len(corners), dtype=np.int64), np.zeros(len(corners), dtype=np.int64)
    # for i, item in enumerate(corners):
    #     x[i], y[i] = item[0], item[1]

    # plt.imshow(im)
    # plt.plot(im_max[0], im_max[1], color='red', marker='.')

    # features = sample_descriptor(im, corners, 3)
    # gaussian_pyr = sol4_utils.build_gaussian_pyramid(im, 3, 3)
    # deep_level_multiplier = 1 / 2 ** 2
    # np.multiply(pos, deep_level_multiplier)
    # print(Ix)

    # arr = np.reshape(np.tile(np.repeat(np.linspace(0,7,8),2).T,10),(8,20))
    # # for i in range(5):
    # #     arr[i] = (arr[i]*i)*(arr[i].T*i)
    #
    # for i in arr:
    #     t = np.where(i==np.max(i))
    #     print(i[t],t[0])
    # arr = np.vstack((np.array([[2, 2], [2, 2], [1, 1], [1, 1]]), [1, 2], [3, 4]))
    # # arr2 = np.vstack((np.array([[2, 2, 2, 2], [1, 1, 1, 1]]), [1, 2, 3, 4]))
    # # arr = arr[:2]/arr[2:]
    # # print(arr)
    #
    # arr[:, 0] = arr[:, 0] + 100
    # print(arr)
    # #
    # rand_point1 = np.random.randint(0, arr.shape[0] + 1, 2)
    # rand_point2 = np.random.randint(0, arr.shape[0] + 1, 2)
    #
    # print(rand_point1)
    # print(rand_point2)
    # print('Returned tuple of arrays :', result)
    # print('List of Indices of maximum element :', result[0]) 
