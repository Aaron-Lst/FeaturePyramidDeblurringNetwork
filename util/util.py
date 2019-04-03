import sys
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
import scipy

if sys.version_info.major == 3:
    xrange = range


def im2uint8(x):
    if x.__class__ == tf.Tensor:
        return tf.cast(tf.clip_by_value(x, 0.0, 1.0) * 255.0, tf.uint8)
    else:
        t = np.clip(x, 0.0, 1.0) * 255.0
        return t.astype(np.uint8)


def ResnetBlock(x, dim, ksize, scope='rb'):
    with tf.variable_scope(scope):
        net = slim.conv2d(x, dim, [ksize, ksize], scope='conv1')
        net = slim.conv2d(net, dim, [ksize, ksize],
                          activation_fn=None, scope='conv2')
        return net + x


def Wide_ResnetBlock(x, dim, ksize, scope='wr'):
    net = []
    with tf.variable_scope(scope):
        for i in xrange(1, 2, ksize+2):
            for j in xrange(int(dim/16)):
                temp = slim.conv2d(
                    x, 4, [1, 1], scope='bottleneck_%d_%d' % (i, j))
                temp = slim.conv2d(
                    temp, 4, [i, i], scope='reception_%d_%d' % (i, j))
                net = tf.concat([temp, net], axis=3,
                                name='concat_%d_%d' % (i, j))
        net = slim.conv2d(net, dim, [1, 1], scope='conv1')
        return net + x


def Bottleneck_layers(x, dim, ksize, scope='bo'):
    with tf.variable_scope(scope):
        net = []


def DenseBlock(x, dim, ksize, scope='de'):
    with tf.variable_scope(scope):
        net = []


def batch_norm_params(is_training=True,
                      batch_norm_decay=0.997,
                      batch_norm_epsilon=1e-5,
                      batch_norm_scale=True):
    batch_norm_params = {
        'is_training': is_training,
        'decay': batch_norm_decay,
        'epsilon': batch_norm_epsilon,
        'scale': batch_norm_scale,
        'updates_collections': tf.GraphKeys.UPDATE_OPS, }
    return batch_norm_params


def build_vgg19(input, reuse=False):
    vgg_path=scipy.io.loadmat('./VGG_Model/imagenet-vgg-verydeep-19.mat')
    with tf.variable_scope("vgg19_feature"):
        if reuse:
            tf.get_variable_scope().reuse_variables()
        net = {}
        vgg_layers = vgg_path['layers'][0]
        net['input'] = input - np.array([123.6800, 116.7790, 103.9390]).reshape((1, 1, 1, 3))
        net['conv1_1'] = build_net('conv', net['input'], get_weight_bias(vgg_layers, 0), name='vgg_conv1_1')
        net['conv1_2'] = build_net('conv', net['conv1_1'], get_weight_bias(vgg_layers, 2), name='vgg_conv1_2')
        net['pool1'] = build_net('pool', net['conv1_2'])
        net['conv2_1'] = build_net('conv', net['pool1'], get_weight_bias(vgg_layers, 5), name='vgg_conv2_1')
        net['conv2_2'] = build_net('conv', net['conv2_1'], get_weight_bias(vgg_layers, 7), name='vgg_conv2_2')
        net['pool2'] = build_net('pool', net['conv2_2'])
        net['conv3_1'] = build_net('conv', net['pool2'], get_weight_bias(vgg_layers, 10), name='vgg_conv3_1')
        net['conv3_2'] = build_net('conv', net['conv3_1'], get_weight_bias(vgg_layers, 12), name='vgg_conv3_2')
        net['conv3_3'] = build_net('conv', net['conv3_2'], get_weight_bias(vgg_layers, 14), name='vgg_conv3_3')
        net['conv3_4'] = build_net('conv', net['conv3_3'], get_weight_bias(vgg_layers, 16), name='vgg_conv3_4')
        net['pool3'] = build_net('pool', net['conv3_4'])
        net['conv4_1'] = build_net('conv', net['pool3'], get_weight_bias(vgg_layers, 19), name='vgg_conv4_1')
        net['conv4_2'] = build_net('conv', net['conv4_1'], get_weight_bias(vgg_layers, 21), name='vgg_conv4_2')
        net['conv4_3'] = build_net('conv', net['conv4_2'], get_weight_bias(vgg_layers, 23), name='vgg_conv4_3')
        net['conv4_4'] = build_net('conv', net['conv4_3'], get_weight_bias(vgg_layers, 25), name='vgg_conv4_4')
        net['pool4'] = build_net('pool', net['conv4_4'])
        net['conv5_1'] = build_net('conv', net['pool4'], get_weight_bias(vgg_layers, 28), name='vgg_conv5_1')
        net['conv5_2'] = build_net('conv', net['conv5_1'], get_weight_bias(vgg_layers, 30), name='vgg_conv5_2')

        return net
def get_weight_bias(vgg_layers,i):

    weights=vgg_layers[i][0][0][2][0][0]
    weights=tf.constant(weights)
    bias=vgg_layers[i][0][0][2][0][1]
    bias=tf.constant(np.reshape(bias,(bias.size)))
    
    return weights,bias

def build_net(ntype,nin,nwb=None,name=None):

    if ntype=='conv':

        return tf.nn.relu(tf.nn.conv2d(nin,nwb[0],strides=[1,1,1,1],padding='SAME',name=name)+nwb[1])

    elif ntype=='pool':

        return tf.nn.avg_pool(nin,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')