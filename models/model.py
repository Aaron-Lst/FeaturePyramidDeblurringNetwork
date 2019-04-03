from __future__ import print_function
import os
import time
import random
import datetime
import scipy.misc
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from datetime import datetime
from util.util import *
import util.nets.pyramid_network as pyramid_network
import util.nets.nets_factory as network
import pdb
class DEBLUR(object):
    def __init__(self, args):
        self.args = args
        self.scale = 1
        self.chns = 3 if self.args.model == 'color' else 1  # input / output channels

        # if args.phase == 'train':
        self.crop_size = 256
        self.data_list = open(args.datalist, 'rt').read().splitlines()
        self.data_list = list(map(lambda x: x.split(' '), self.data_list))
        random.shuffle(self.data_list)
        self.model_flag = self.args.model_flag
        self.exp_num = self.args.exp_num
        self.save_flag = self.model_flag + '_' + self.exp_num
        self.train_dir = os.path.join(
            './checkpoints', self.save_flag)
        if not os.path.exists(self.train_dir):
            os.makedirs(self.train_dir)

        self.batch_size = args.batch_size
        self.epoch = args.epoch
        self.data_size = (len(self.data_list)) // self.batch_size
        self.max_steps = int(self.epoch * self.data_size)
        self.learning_rate = args.learning_rate
        self.load_step = args.load_step
        self.loss_threshold = 0.004
        self.is_training = args.is_training
        self.pretrain = args.pretrain

    def input_producer(self, batch_size=10):
        def read_data():
            img_a = tf.image.decode_image(tf.read_file(tf.string_join(['/home/opt603/sda5/GOPRO_Large/train/', self.data_queue[0]])),
                                          channels=3)
            img_b = tf.image.decode_image(tf.read_file(tf.string_join(['/home/opt603/sda5/GOPRO_Large/train/', self.data_queue[1]])),
                                          channels=3)
            img_a, img_b = preprocessing([img_a, img_b])
            return img_a, img_b

        def preprocessing(imgs):
            imgs = [tf.cast(img, tf.float32) / 255.0 for img in imgs]
            if self.args.model is not 'color':
                imgs = [tf.image.rgb_to_grayscale(img) for img in imgs]
            img_crop = tf.unstack(tf.random_crop(tf.stack(imgs, axis=0), [2, self.crop_size, self.crop_size, self.chns]),
                                  axis=0)
            return img_crop

        with tf.variable_scope('input'):
            List_all = tf.convert_to_tensor(self.data_list, dtype=tf.string)
            gt_list = List_all[:, 0]
            in_list = List_all[:, 1]

            self.data_queue = tf.train.slice_input_producer(
                [in_list, gt_list], capacity=20)
            image_in, image_gt = read_data()
            batch_in, batch_gt = tf.train.batch(
                [image_in, image_gt], batch_size=batch_size, num_threads=8, capacity=20)

        return batch_in, batch_gt

    def model_fpn(self, inputs, name):
        _, h, w, _ = inputs.get_shape().as_list()
        # inp_pred = inputs
        scale = self.scale
        hi = int(round(h * scale))
        wi = int(round(w * scale))
        inp_blur = tf.image.resize_images(
            inputs, [hi, wi], method=0)
        # inp_pred = tf.stop_gradient(tf.image.resize_images(inp_pred, [hi, wi], method=0))
        # inp_all = tf.concat([inp_blur, inp_pred], axis=3, name='inp')
        refine = {}
        _, end_points, pyramid_map = network.get_network('resnet50', inp_blur,
            weight_decay=0.00005, is_training=self.is_training)
        #pdb.set_trace()
        pyramid_map = pyramid_network.build_pyramid(pyramid_map, end_points)

        with slim.arg_scope([slim.conv2d, slim.conv2d_transpose],
                    activation_fn=None, padding='SAME', normalizer_fn=None,
                    weights_initializer=tf.truncated_normal_initializer(stddev=0.001),
                    weights_regularizer=slim.l2_regularizer(0.00001),
                    biases_initializer=tf.constant_initializer(0.0)):
            up_shape = tf.shape(inp_blur)
            s = tf.image.resize_bilinear(pyramid_map['P1'], [up_shape[1], up_shape[2]], name='C0/upscale')
            s_ = slim.conv2d(inp_blur, 256, [1,1], stride=1, scope='C0')
        
            s = tf.add(s, s_, name='C0/addition')
            pyramid_map['P0'] = slim.conv2d(s, 256, [3,3], stride=1, scope='C0/fusion')
            refine['P0'] = slim.conv2d(pyramid_map['P0'], self.chns, [5, 5], activation_fn=None, reuse=False, scope='pred_')	
            for i in range(5,0,-1):
                p = 'P%d'%i
                refine[p] = slim.conv2d(pyramid_map[p], self.chns, [5, 5], activation_fn=None, reuse=True, scope='pred_')
        
        return refine

    def build_model(self):
        img_in, img_gt = self.input_producer(self.batch_size)

        tf.summary.image('img_in', im2uint8(img_in))
        tf.summary.image('img_gt', im2uint8(img_gt))
        print('img_in, img_gt', img_in.get_shape(), img_gt.get_shape())

        # generator
        pred = self.model_fpn(
            img_in, name=self.model_flag)
        # calculate multi-scale loss
        self.loss_total = 0
        l2_loss = 0
        for i in range(0,5,1):
            _pred = pred['P%d'%i]
            _, hi, wi, _ = _pred.get_shape().as_list()
            gt_i = tf.image.resize_images(img_gt, [hi, wi], method=0)
            l2_loss = tf.reduce_mean((gt_i - _pred) ** 2)* (1-0.2*i)
            tf.summary.scalar('refine_loss_%d'%i, l2_loss)
            tf.summary.image('refine_%d'%i, im2uint8(_pred))
            self.loss_total += l2_loss
        # losses
        tf.summary.scalar('loss_total', self.loss_total)

        # training vars
        all_vars = tf.trainable_variables()
        self.all_vars = all_vars
        for var in all_vars:
            print(var.name)


    def train(self):
        global_step = tf.Variable(
            initial_value=0, dtype=tf.int32, trainable=False)
        self.global_step = global_step

        # build model
        self.build_model()

        # learning rate decay
        self.lr = tf.train.polynomial_decay(self.learning_rate, global_step, self.max_steps, end_learning_rate=1e-6,
                                            power=0.3)
        tf.summary.scalar('learning_rate', self.lr)

        # training operators
        train_op = tf.train.AdamOptimizer(self.lr)
        train_op = train_op.minimize(self.loss_total, self.global_step, self.all_vars)

        # session and thread
        gpu_options = tf.GPUOptions(allow_growth=True)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        self.sess = sess
        sess.run(tf.global_variables_initializer())

        #restore pretrain weight
        if self.pretrain:
            checkpoint_path = '/home/opt603/sda6/pretrain_weight/resnet_v1_50.ckpt'
            model_variables = slim.get_model_variables()
            restore_variables = [var for var in model_variables if
                                 (var.name.startswith('resnet_v1_50')
                                  and not var.name.startswith('{}/logits'.format('resnet_v1_50')))]
            for var in restore_variables:
                print(var.name)
            restorer = tf.train.Saver(restore_variables)
            restorer.restore(sess, checkpoint_path)
            print("model restore from pretrained mode, path is :", checkpoint_path)

        self.saver = tf.train.Saver(
            max_to_keep=50, keep_checkpoint_every_n_hours=1)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        # training summary
        summary_op = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter(
            self.train_dir, sess.graph, flush_secs=30)

        for step in xrange(sess.run(global_step), self.max_steps + 1):

            start_time = time.time()

            # update network
            _, loss_total_val = sess.run([train_op, self.loss_total])

            duration = time.time() - start_time
            # print loss_value
            assert not np.isnan(
                loss_total_val), 'Model diverged with loss = NaN'

            if step % 5 == 0:
                num_examples_per_step = self.batch_size
                examples_per_sec = num_examples_per_step / duration
                sec_per_batch = float(duration)

                format_str = (self.save_flag+' '+'%s: step %d, loss = (%.5f;)(%.1f data/s; %.3f s/bch)')
                print(format_str % (datetime.now().strftime('%Y-%m-%d %H:%M:%S'), step, loss_total_val, examples_per_sec, sec_per_batch))

            if step % 20 == 0:
                # summary_str = sess.run(summary_op, feed_dict={inputs:batch_input, gt:batch_gt})
                summary_str = sess.run(summary_op)
                summary_writer.add_summary(summary_str, global_step=step)

            # Save the model checkpoint periodically.
            if step % 1000 == 0 or step == self.max_steps :
                checkpoint_path = os.path.join(self.train_dir, 'checkpoints')
                self.save(sess, checkpoint_path, step)

    def save(self, sess, checkpoint_dir, step):
        model_name = "deblur.model"
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        self.saver.save(sess, os.path.join(
            checkpoint_dir, model_name), global_step=step)

    def load(self, sess, checkpoint_dir, step=None):
        print(" [*] Reading checkpoints...")
        model_name = "deblur.model"
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)

        if step is not None:
            ckpt_name = model_name + '-' + str(step)
            self.saver.restore(sess, os.path.join(
                checkpoint_dir, 'checkpoints', ckpt_name))
            print(" [*] Reading intermediate checkpoints... Success")
            return str(step)
        elif ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            ckpt_iter = ckpt_name.split('-')[1]
            self.saver.restore(sess, os.path.join(checkpoint_dir, ckpt_name))
            print(" [*] Reading updated checkpoints... Success")
            return ckpt_iter
        else:
            print(" [*] Reading checkpoints... ERROR")
            return False

    def test(self, height, width, output_path):
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        test_datalist = open('test_datalist.txt', 'rt').read().splitlines()
        test_datalist = list(map(lambda x: x.split(' '), test_datalist))

        imgsName = [x[0] for x in test_datalist]

        H, W = height, width
        inp_chns = 3 if self.args.model == 'color' else 1
        self.batch_size = 1 if self.args.model == 'color' else 3
        inputs = tf.placeholder(
            shape=[self.batch_size, H, W, inp_chns], dtype=tf.float32)
        outputs, _ = self.generator(
            inputs, reuse=False, scope=self.model_flag)

        sess = tf.Session(config=tf.ConfigProto(
            gpu_options=tf.GPUOptions(allow_growth=True)))

        self.saver = tf.train.Saver()
        self.load(sess, self.train_dir, step=self.load_step)

        for imgName in imgsName:
            blur = scipy.misc.imread(imgName)
            split_name = imgName.split('/')
            path_temp = os.path.join(
                output_path, self.model_flag + '_' + self.exp_num, split_name[-3], 'sharp')
            if not os.path.exists(path_temp):
                os.makedirs(path_temp)
            h, w, c = blur.shape
            # make sure the width is larger than the height
            rot = False
            if h > w:
                blur = np.transpose(blur, [1, 0, 2])
                rot = True
            h = int(blur.shape[0])
            w = int(blur.shape[1])
            resize = False
            if h > H or w > W:
                scale = min(1.0 * H / h, 1.0 * W / w)
                new_h = int(h * scale)
                new_w = int(w * scale)
                blur = scipy.misc.imresize(blur, [new_h, new_w], 'bicubic')
                resize = True
                blurPad = np.pad(
                    blur, ((0, H - new_h), (0, W - new_w), (0, 0)), 'edge')
            else:
                blurPad = np.pad(
                    blur, ((0, H - h), (0, W - w), (0, 0)), 'edge')
            blurPad = np.expand_dims(blurPad, 0)
            if self.args.model is not 'color':
                blurPad = np.transpose(blurPad, (3, 1, 2, 0))

            start = time.time()
            deblur = sess.run(outputs, feed_dict={inputs: blurPad / 255.0})
            duration = time.time() - start
            print('Saving results: %s ... %4.3fs' % (imgName, duration))
            res = deblur
            if self.args.model is not 'color':
                res = np.transpose(res, (3, 1, 2, 0))
            res = im2uint8(res[0, :, :, :])
            # crop the image into original size
            if resize:
                res = res[:new_h, :new_w, :]
                res = scipy.misc.imresize(res, [h, w], 'bicubic')
            else:
                res = res[:h, :w, :]

            if rot:
                res = np.transpose(res, [1, 0, 2])
            scipy.misc.imsave(os.path.join(path_temp, split_name[-1]), res)
