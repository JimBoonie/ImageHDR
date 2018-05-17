import cv2 
import numpy as np

from imageio import imread, imwrite
from skimage import color
from scipy.ndimage.filters import gaussian_filter
from scipy.misc import imresize

ref_img_name = 'imgs/reference1.jpg'
ali_img_name = 'imgs/aligned1.jpg'
tmp_img_name = 'imgs/tmp.jpg'

sig = 3
n_octaves = 2
n_dogs = 4
k = 2 ** (1 / n_dogs)

def to_uint8(img):
    scaled = (img - np.min(img)) / (np.max(img) - np.min(img)) * 255
    return scaled.astype('uint8')

def build_dog_pyramid(img, sigma=1, n_dogs=2, n_octaves=2):
    pyramid = []
    for i in range(n_octaves):
        level = []
        print("octave: {}".format(i))
        for j in range(n_dogs):
            print("k: {}".format(k ** (j + 1)))
            gauss_img1 = gaussian_filter(img, sigma=sigma * k ** j)
            gauss_img2 = gaussian_filter(img, sigma=sigma * k ** (j + 1))
            dog_img = gauss_img2 - gauss_img1
            level.append(dog_img)

        pyramid.append(level)
        img = imresize(img, 0.5, interp='bilinear')

    return pyramid

# load image
ref_img = imread(ref_img_name)
ref_gray = color.rgb2gray(ref_img)

# build dog pyramid
pyramid = build_dog_pyramid(ref_img, sigma=sig, n_dogs=n_dogs, n_octaves=n_octaves)
print("# octaves: {}".format(len(pyramid)))
print("# dogs: {}".format(len(pyramid[0])))

# save output
out_img = pyramid[0][0]
imwrite(tmp_img_name, to_uint8(out_img))