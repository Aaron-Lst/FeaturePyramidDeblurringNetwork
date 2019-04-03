import numpy
import sys
import scipy.misc
import os.path

import argparse

import psnr
import ssim

def parse_args():
    parser = argparse.ArgumentParser(description='metrics arguments')
    parser.add_argument('--model_flag', type=str, default='refine',
                        help='model type: [refine | deeper | rpn]')
    parser.add_argument('--output_path', type=str, default='/home/opt603/lst/code/SRN-Deblur/testing_res',
                        help='output path for testing images')
    parser.add_argument('--exp_num', type=str, default='1')
    args = parser.parse_args()
    return args

args = parse_args()
pred_imgs_path = os.path.join(args.output_path, args.model_flag + '_' + args.exp_num)
imgs_list_file = '/home/opt603/lst/code/SRN-Deblur/test_datalist.txt'

imgs_datalist = open(imgs_list_file, 'rt').read().splitlines()
imgs_datalist = list(map(lambda x: x.split(' '), imgs_datalist))
imgsName = [x[1] for x in imgs_datalist]

psnr_values = []
ssim_values = []

for imgName in imgsName:
    real = scipy.misc.imread(imgName, flatten=True).astype(numpy.float32)
    split_name = imgName.split('/')
    pred_imgName = os.path.join(
        pred_imgs_path, split_name[-3], split_name[-2], split_name[-1])
    pred = scipy.misc.imread(pred_imgName, flatten=True).astype(numpy.float32)

    width, height = pred.shape[1], pred.shape[0]

    print ('Resolution %d x %d' % (width, height))

    ssim_value = ssim.ssim_exact(real/255, pred/255)
    psnr_value = psnr.psnr(real, pred)

    print('psnr:%.5f ssim:%.5f' % (psnr_value, ssim_value))

    ssim_values.append(ssim_value)
    psnr_values.append(psnr_value)

final_psnr_value = numpy.mean(psnr_values)
final_ssim_value = numpy.mean(ssim_values)

print('final psnr:%.5f final ssim:%.5f' % (final_psnr_value, final_ssim_value))
