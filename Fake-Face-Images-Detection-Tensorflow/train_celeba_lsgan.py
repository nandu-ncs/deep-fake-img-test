from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import glob
import utils
import traceback
import numpy as np
import tensorflow as tf
import models_64x64 as models


""" param """
epoch = 50
batch_size = 64
lr = 0.0002
z_dim = 100
gpu_id = 3

''' data '''
# you should prepare your own data in ./data/img_align_celeba
# celeba original size is [218, 178, 3]


def preprocess_fn(img):
    crop_size = 108
    re_size = 64
    img = tf.compat.v1.image.crop_to_bounding_box(img, (218 - crop_size) // 2, (178 - crop_size) // 2, crop_size, crop_size)
    img = tf.compat.v1.to_float(tf.compat.v1.image.resize_images(img, [re_size, re_size], method=tf.compat.v1.image.ResizeMethod.BICUBIC)) / 127.5 - 1
    return img

img_paths = glob.glob('./data/img_align_celeba/img_align_celeba/')
data_pool = utils.DiskImageData('./data/img_align_celeba/img_align_celeba/', batch_size, shape=[218, 178, 3], preprocess_fn=preprocess_fn)


""" graphs """
with tf.compat.v1.device('/gpu:%d' % gpu_id):
    ''' models '''
    generator = models.generator
    discriminator = models.discriminator

    ''' graph '''
    # inputs
    real = tf.compat.v1.placeholder(tf.compat.v1.float32, shape=[None, 64, 64, 3])
    z = tf.compat.v1.placeholder(tf.compat.v1.float32, shape=[None, z_dim])

    # generate
    fake = generator(z, reuse=False)

    # dicriminate
    r_logit = discriminator(real, reuse=False)
    f_logit = discriminator(fake)

    # losses
    d_r_loss = tf.compat.v1.losses.mean_squared_error(tf.compat.v1.ones_like(r_logit), r_logit)
    d_f_loss = tf.compat.v1.losses.mean_squared_error(tf.compat.v1.zeros_like(f_logit), f_logit)
    d_loss = (d_r_loss + d_f_loss) / 2.0
    g_loss = tf.compat.v1.losses.mean_squared_error(tf.compat.v1.ones_like(f_logit), f_logit)

    # otpims
    d_var = utils.trainable_variables('discriminator')
    g_var = utils.trainable_variables('generator')
    d_step = tf.compat.v1.train.AdamOptimizer(learning_rate=lr, beta1=0.5).minimize(d_loss, var_list=d_var)
    g_step = tf.compat.v1.train.AdamOptimizer(learning_rate=lr, beta1=0.5).minimize(g_loss, var_list=g_var)

    # summaries
    d_summary = utils.summary({d_loss: 'd_loss'})
    g_summary = utils.summary({g_loss: 'g_loss'})

    # sample
    f_sample = generator(z, training=False)


""" train """
''' init '''
# session
sess = utils.session()
# iteration counter
it_cnt, update_cnt = utils.counter()
# saver
saver = tf.compat.v1.train.Saver(max_to_keep=5)
# summary writer
summary_writer = tf.compat.v1.summary.FileWriter('./summaries/celeba_lsgan', sess.graph)

''' initialization '''
ckpt_dir = './checkpoints/celeba_lsgan'
utils.mkdir(ckpt_dir + '/')
if not utils.load_checkpoint(ckpt_dir, sess):
    sess.run(tf.compat.v1.global_variables_initializer())

''' train '''
try:
    z_ipt_sample = np.random.normal(size=[100, z_dim])

    batch_epoch = len(data_pool) // (batch_size)
    max_it = epoch * batch_epoch
    for it in range(sess.run(it_cnt), max_it):
        sess.run(update_cnt)

        # which epoch
        epoch = it // batch_epoch
        it_epoch = it % batch_epoch + 1

        # batch data
        real_ipt = data_pool.batch()
        z_ipt = np.random.normal(size=[batch_size, z_dim])

        # train D
        d_summary_opt, _ = sess.run([d_summary, d_step], feed_dict={real: real_ipt, z: z_ipt})
        summary_writer.add_summary(d_summary_opt, it)

        # train G
        sess.run([g_step], feed_dict={z: z_ipt})
        g_summary_opt, _ = sess.run([g_summary, g_step], feed_dict={z: z_ipt})
        summary_writer.add_summary(g_summary_opt, it)

        # display
        if it % 1 == 0:
            print("Epoch: (%3d) (%5d/%5d)" % (epoch, it_epoch, batch_epoch))

        # save
        if (it + 1) % 1000 == 0:
            save_path = saver.save(sess, '%s/Epoch_(%d)_(%dof%d).ckpt' % (ckpt_dir, epoch, it_epoch, batch_epoch))
            print('Model saved in file: % s' % save_path)

        # sample
        if (it + 1) % 100 == 0:
            f_sample_opt = sess.run(f_sample, feed_dict={z: z_ipt_sample})

            save_dir = './sample_images_while_training/celeba_lsgan'
            utils.mkdir(save_dir + '/')
            utils.imwrite(utils.immerge(f_sample_opt, 10, 10), '%s/Epoch_(%d)_(%dof%d).jpg' % (save_dir, epoch, it_epoch, batch_epoch))

except Exception, e:
    traceback.print_exc()
finally:
    print(" [*] Close main session!")
    sess.close()
