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
batch_size = 100
lr = 0.0002
z_dim = 100
gpu_id = 3

''' data '''
# you should prepare your own data in ./data/img_align_celeba
# celeba original size is [218, 178, 3]

""" graphs """
with tf.compat.v1.device('/gpu:%d' % gpu_id):
    ''' models '''
    generator = models.generator

    ''' graph '''
    # inputs
    z = tf.compat.v1.placeholder(tf.compat.v1.float32, shape=[None, z_dim])

    # generate
    gene = generator(z, training=False, reuse=False)


""" train """
''' init '''
# session
sess = utils.session()
# saver
saver = tf.compat.v1.train.Saver()

''' initialization '''
ckpt_dir = './checkpoints/celeba_wgan/Epoch_(48)_(616of633).ckpt'
saver.restore(sess, ckpt_dir)
sess.run(tf.compat.v1.global_variables_initializer())
saver.restore(sess, ckpt_dir)

try:
    for it in range(2000):
        z_ipt = np.random.normal(size=[batch_size, z_dim])
        img = sess.run(gene, feed_dict={z: z_ipt})

        save_dir = './results/celeba_wgan'
        utils.mkdir(save_dir + '/')
        utils.batchimwrite(img, '%s/img%d' % (save_dir,it))
        
        if it%1000==0:
			print("Batch %d images are done!!"%(it))

except Exception, e:
    traceback.print_exc()
finally:
    print(" [*] Close main session!")
    sess.close()
