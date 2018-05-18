import cv2 
import numpy as np
import matplotlib.pyplot as plt

from imageio import imread, imwrite
from skimage import color
from skimage.transform import resize, rescale, downscale_local_mean
from scipy.ndimage.filters import gaussian_filter

ref_img_name = 'imgs/reference1.jpg'
ali_img_name = 'imgs/aligned1.jpg'
tmp_img_name = 'imgs/tmp.jpg'

sig = 1.6
n_octaves = 4
n_dogs = 2
k = 2 ** (1 / n_dogs)

def summarize_img(img, name='img'):
    print('** Summary of image: {} **'.format(name))
    print('Shape: {}'.format(img.shape))
    print('Dtype: {}'.format(img.dtype))
    print('Min: {}'.format(np.min(img)))
    print('Max: {}'.format(np.max(img)))
    print('Mean: {}'.format(np.mean(img)))
    print('Std: {}'.format(np.std(img)))
    print('\n')

def stretch_lims(x, lims=[0, 1]):
    x = (x - np.min(x)) / (np.max(x) - np.min(x))
    return x * (lims[1] - lims[0]) + lims[0]

def to_uint8(img):
    return stretch_lims(img, lims=[0, 255]).astype('uint8')

def build_dog_pyramid(img, sigma=1, n_dogs=2, n_octaves=2):
    img_shape = (img.shape[0], img.shape[1])
    img = rescale(img, 2)
    pyramid = np.ndarray(
        [img_shape[0], img_shape[1], n_dogs * n_octaves],
        dtype=img.dtype)
    lvl_idx = 0
    for i in range(n_octaves):
        print("octave: {}".format(i))
        for j in range(n_dogs):
            print("k: {}".format(k ** (j + 1)))
            gauss_img1 = gaussian_filter(img, sigma=sigma * k ** j)
            gauss_img2 = gaussian_filter(img, sigma=sigma * k ** (j + 1))
            dog_img = gauss_img2 - gauss_img1
            pyramid[:, :, lvl_idx] = resize(dog_img, img_shape)
            lvl_idx = lvl_idx + 1

        summarize_img(img)
        img = downscale_local_mean(img, (2, 2))
        # img = resize(img, [x // 2 for x in img.shape[0:2]], interp='bilinear')
        # img = rescale(img, 0.5, anti_aliasing=True)

    return pyramid

# load image
ref_img = imread(ref_img_name)
ref_gray = color.rgb2gray(ref_img)
summarize_img(ref_gray, name='ref_gray')

# build dog pyramid
pyramid = build_dog_pyramid(ref_gray, sigma=sig, n_dogs=n_dogs, n_octaves=n_octaves)
print('# octaves: {}'.format(n_octaves))
print('# dogs: {}'.format(n_dogs))

def display_pyramid(pyramid, n_rows, n_cols):
    for i in range(pyramid.shape[-1]):
        plt.subplot(n_rows, n_cols, i + 1)
        plt.imshow(pyramid[:, :, i])
        sig1 = sig * k**i
        sig2 = sig * k**(i + 1)
        plt.title('G(x, y, {:.2f}) - G(x, y, {:.2f})'.format(sig2, sig1))
    plt.show()

display_pyramid(pyramid, n_dogs, n_octaves)

# save output
out_img = pyramid[:, :, 0]
summarize_img(out_img, 'out_img')
imwrite(tmp_img_name, to_uint8(out_img))