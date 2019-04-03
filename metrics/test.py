import numpy
import sys
import scipy.misc
import os.path
import tensorflow as tf
import argparse

import psnr
import ssim


def log10(x):

    numerator = tf.log(x)

    denominator = tf.log(tf.constant(10, dtype=numerator.dtype))

    return numerator / denominator

def pppsnr(im1, im2):

    img_arr1 = numpy.array(im1).astype('float32')

    img_arr2 = numpy.array(im2).astype('float32')

    mse = tf.reduce_mean(tf.squared_difference(img_arr1, img_arr2))

    psnr = tf.constant(255**2, dtype=tf.float32)/mse

    result = tf.constant(10, dtype=tf.float32)*log10(psnr)

    with tf.Session():

        result = result.eval()

    return result

def parse_args():
    parser = argparse.ArgumentParser(description='metrics arguments')
    parser.add_argument('--a', type=str, default='refine',
                        help='image_a')
    parser.add_argument('--b', type=str, default='/home/opt603/lst/code/SRN-Deblur/testing_res',
                        help='image_b')
    args = parser.parse_args()
    return args

args = parse_args()
imgName = args.a
pred_imgName = args.b
real = scipy.misc.imread(imgName, flatten=True).astype(numpy.float32)
pred = scipy.misc.imread(pred_imgName, flatten=True).astype(numpy.float32)

width, height = pred.shape[1], pred.shape[0]

print('Resolution %d x %d' % (width, height))

ssim_value = ssim.ssim_exact(real/255, pred/255)
psnr_value = psnr.psnr(real, pred)

print('psnr:%.5f ssim:%.5f' % (psnr_value, ssim_value))
