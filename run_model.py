import os
import argparse
import tensorflow as tf
# import models.model_gray as model
# import models.model_color as model
import models.model as model


def parse_args():
    parser = argparse.ArgumentParser(description='deblur arguments')
    parser.add_argument('--phase', type=str, default='test',
                        help='determine whether train or test')
    parser.add_argument('--datalist', type=str,
                        default='./datalist_gopro.txt', help='training datalist')
    parser.add_argument('--model', type=str, default='color',
                        help='model type: [gray | color]')
    parser.add_argument('--model_flag', type=str, default='fpn',
                        help='model type')
    parser.add_argument(
        '--batch_size', help='training batch size', type=int, default=128)
    parser.add_argument(
        '--epoch', help='training epoch number', type=int, default=2000)
    parser.add_argument('--lr', type=float, default=1e-4,
                        dest='learning_rate', help='initial learning rate')
    parser.add_argument('--gpu', dest='gpu_id', type=str,
                        default='0', help='use gpu or cpu')
    parser.add_argument('--height', type=int, default=720,
                        help='height for the tensorflow placeholder, should be multiples of 16')
    parser.add_argument('--width', type=int, default=1280,
                        help='width for the tensorflow placeholder, should be multiple of 16 for 3 scales')
    parser.add_argument('--load_step', type=str, default='300000',
                        help='load checkpoint')
    parser.add_argument('--output_path', type=str, default='./testing_res',
                        help='output path for testing images')
    parser.add_argument('--exp_num', type=str, default='1')
    parser.add_argument('--is_training', type=bool, default=False)
    parser.add_argument('--pretrain', type=bool, default=True)
    args = parser.parse_args()
    return args


def main(_):
    args = parse_args()

    # set gpu/cpu mode
    if len(args.gpu_id) >= 0:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''

    # set up deblur models
    deblur = model.DEBLUR(args)
    if args.phase == 'test':
        deblur.test(args.height, args.width, args.output_path)
    elif args.phase == 'train':
        deblur.train()
    else:
        print('phase should be set to either test or train')


if __name__ == '__main__':
    tf.app.run()
