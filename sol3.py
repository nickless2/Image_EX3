import numpy as np
from scipy.misc import imread as imread
from skimage.color import rgb2gray
from scipy.signal import convolve as sig_convolve
from scipy.ndimage.filters import convolve
import matplotlib.pyplot as plt
import os


def read_image(filename, representation):

    im = imread(filename)
    # check if we want to convert RGB pic to greyscale
    if representation == 1 and im.shape.__len__() == 3:
        im = rgb2gray(im)
        im = im.astype(np.float32)

    else:
        im = im.astype(np.float32)
        im /= 255

    return im


def calc_kernel(kernel_size):

    dim1_kernel = np.array([1, 1]).astype(np.float32)
    dim1_result_kernel = dim1_kernel

    for i in range(kernel_size - 2):
        dim1_result_kernel = sig_convolve(dim1_result_kernel, dim1_kernel)

    norm_final_kernel = dim1_result_kernel / np.sum(dim1_result_kernel)

    return np.array([norm_final_kernel])


def reduce(im, filter_vec):

    # blur
    row_convolution = convolve(im, filter_vec)
    final_convolution = convolve(row_convolution, filter_vec.transpose())

    # sub-sample every second pixel
    final_convolution = final_convolution[::2, ::2]

    return final_convolution


def expand(im, filter_vec):

    # zero padding
    extended_im = np.zeros((im.shape[0] * 2, im.shape[1] * 2))
    extended_im[::2, ::2] = im

    # blur
    extended_im = convolve(extended_im, 2 * filter_vec)
    extended_im = convolve(extended_im, 2 * filter_vec.transpose())

    return extended_im


def build_gaussian_pyramid(im, max_levels, filter_size):

    # minimum image resolution
    MIN_IM_SIZE = 16

    filter_vec = calc_kernel(filter_size)

    # insert "im" as first item in regular python list
    pyr = [im]

    # perform reduce for "max_levels"
    for i in range(1, max_levels):
        reduced_im = reduce(pyr[i-1], filter_vec)
        if reduced_im.shape[0] < MIN_IM_SIZE or reduced_im.shape[1] < \
                MIN_IM_SIZE:
            break
        pyr.append(reduced_im)

    return pyr, filter_vec


def build_laplacian_pyramid(im, max_levels, filter_size):

    gauss_pyr, filter_vec = build_gaussian_pyramid(im, max_levels, filter_size)
    laplac_pyr = []

    for i in range(len(gauss_pyr) - 1):
        laplac_pyr.append(gauss_pyr[i] - expand(gauss_pyr[i+1], filter_vec))

    laplac_pyr.append(gauss_pyr[-1])

    return laplac_pyr, filter_vec


def laplacian_to_image(lpyr, filter_vec, coeff):

    im = lpyr[-1]

    for i in range(len(lpyr) - 2, -1, -1):
        im = expand(im, filter_vec)
        im = im + lpyr[i] * coeff[i]

    return im


def render_pyramid(pyr, levels):

    height, width = pyr[0].shape
    # get black image width
    for i in range(1, min(levels, len(pyr))):
        width += pyr[i].shape[0]

    res = np.zeros((height, width))
    width_boundary = 0

    for i in range(min(levels, len(pyr))):
        # stretch to [0,1]
        pyr[i] -= np.min(pyr[i])
        pyr[i] = np.true_divide(pyr[i], np.max(pyr[i]))
        # insert image to black image
        height, width = pyr[i].shape
        res[0:height, width_boundary:width_boundary + width] = pyr[i]

        width_boundary += width

    return res


def display_pyramid(pyr, levels):

    im = render_pyramid(pyr, levels)
    plt.imshow(im, cmap=plt.cm.gray)
    plt.show()


def pyramid_blending(im1, im2, mask, max_levels, filter_size_im,
                     filter_size_mask):

    # create respective pyramids
    im1_pyr, im1_filter = build_laplacian_pyramid(im1, max_levels,
                                                  filter_size_im)
    im2_pyr, im2_filter = build_laplacian_pyramid(im2, max_levels,
                                                  filter_size_im)
    mask_pyr, mask_filter = build_gaussian_pyramid(mask.astype(np.float32),
                                                   max_levels, filter_size_mask)

    # construct laplacian pyramid from blending
    out_im_pyr = []
    for i in range(len(mask_pyr)):
        out_im_pyr.append(mask_pyr[i] * im1_pyr[i] + (1 - mask_pyr[i]) *
                          im2_pyr[i])

    # construct image from laplacian pyramid
    out_im = laplacian_to_image(out_im_pyr, im1_filter, np.ones(len(out_im_pyr)))
    out_im = np.clip(out_im, 0, 1)

    return out_im


def relpath(filename):

    return os.path.join(os.path.dirname(__file__), filename)


def rgb_per_color(im1, im2, mask, max_levels, filter_size_im, filter_size_mask):

    red = pyramid_blending(im1[:, :, 0], im2[:, :, 0], mask[:, :, 0],
                           max_levels, filter_size_im, filter_size_mask)
    green = pyramid_blending(im1[:, :, 1], im2[:, :, 1], mask[:, :, 1],
                             max_levels, filter_size_im, filter_size_mask)
    blue = pyramid_blending(im1[:, :, 2], im2[:, :, 2], mask[:, :, 2],
                            max_levels, filter_size_im, filter_size_mask)

    return np.dstack((red, green, blue))


def show(im1, im2, mask, blend):

    plt.subplot(221)
    plt.imshow(im1)
    plt.subplot(222)
    plt.imshow(im2)
    plt.subplot(223)
    plt.imshow(mask)
    plt.subplot(224)
    plt.imshow(blend)
    plt.show()


def blending_example1():

    im_obser = read_image(relpath('obser.jpg'), 2)
    im_ship = read_image(relpath('ship.jpg'), 2)
    im_mask = read_image(relpath('mask.jpg'), 2)

    im_blend = rgb_per_color(im_obser, im_ship, im_mask, 5, 5, 5)

    show(im_obser, im_ship, im_mask, im_blend)


def blending_example2():

    im_eyes = read_image(relpath('eyes.jpg'), 2)
    im_space = read_image(relpath('space2.jpg'), 2)
    im_mask = read_image(relpath('mask2.jpg'), 2)

    im_blend = rgb_per_color(im_eyes, im_space, im_mask, 5, 5, 5)

    show(im_eyes, im_space, im_mask, im_blend)
