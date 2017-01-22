# EE488C Special Topics in EE <Deep Learning and AlphaGo>, Fall 2016
# 2016/11/08

##
## We load a pre-trained AlexNet, read an image, and perform image
## classification that outputs 5 most probable categories and their
## probabilities.
##
##
##

import numpy as np
from numpy import random
import math
import tensorflow as tf
from PIL import Image
from caffe_classes import class_names
import sys

# -- Open image

# open a pictures, if the mode is not RGB, then we convert it to RGB.
im1 = Image.open(sys.argv[1])
if im1.mode != 'RGB':
    im1 = im1.convert('RGB')

# resize it to 256x256 and converts it to float32.
im1 = im1.resize((256, 256), Image.ANTIALIAS)
im1 = np.asarray(im1, dtype='float32')
im1 /= 255.

# -- Cropping

# Data augmentation, from 1 picture to 10 augmented pictures.
# Do 5 crops according to entries and the corresponding flipped pictures.
crop_entry = [[0, 0], [0, 29], [29, 0], [29, 29], [14, 14]]
im1_crop = np.empty((10, 227, 227, 3), dtype=np.float32)
for k in range(5):
    im1_crop[k, :, :, :] = im1[crop_entry[k][0]:crop_entry[k][0] + 227,
                               crop_entry[k][1]:crop_entry[k][1] + 227, :]
im1_crop[5:10, :, :, :] = im1_crop[0:5, :, ::-1, :]

im1_crop = im1_crop[:, :, :, [2, 1, 0]]
im1_crop = 255. * im1_crop

# -- Subtract mean

# Substract the mean using the RGB mean computed from ILSVRC 2012 dataset.
mean_file = np.load('ilsvrc_2012_mean.npy').mean(1).mean(1)
mean_file = np.expand_dims(mean_file, axis=0)
mean_file = np.expand_dims(mean_file, axis=0)
mean_file = np.expand_dims(mean_file, axis=0)
mean_repeat = np.repeat(mean_file, 10, axis=0)
mean_repeat = np.repeat(mean_repeat, 227, axis=1)
mean_repeat = np.repeat(mean_repeat, 227, axis=2)
im1_crop = im1_crop - mean_repeat

# Save the picture.
np.save('im1_cropped.npy', im1_crop)

# Constructing AlexNet
net_data = np.load("bvlc_alexnet.npy").item()
for x in net_data:
    exec ("%s = %s" % (str(x) + "W", "tf.Variable(net_data[x][0])"))
    exec ("%s = %s" % (str(x) + "b", "tf.Variable(net_data[x][1])"))


# Convolution function
def conv(input, kernel, biases, k_h, k_w, c_o, s_h, s_w, padding="VALID", group=1):
    input_groups, kernel_groups = tf.split(3, group, input), tf.split(3, group, kernel)
    convolve = lambda i, k: tf.nn.conv2d(i, k, [1, s_h, s_w, 1], padding=padding)
    output_groups = [convolve(i, k) for i, k in zip(input_groups, kernel_groups)]
    conv = tf.concat(3, output_groups)
    return tf.reshape(tf.nn.bias_add(conv, biases), [-1] + conv.get_shape().as_list()[1:])


x = tf.placeholder(tf.float32, shape=(None, 227, 227, 3))

# CONV1) 96 filters. 11 x 11 kernel. 4 x 4 strides.
# -- ReLU
# -- Output normalization
# -- Maxpooling. 3 x 3 kernel, 2 x 2 strides.
conv1 = tf.nn.relu(conv(x, conv1W, conv1b, 11, 11, 96, 4, 4, padding="VALID", group=1))
lrn1 = tf.nn.local_response_normalization(conv1, depth_radius=2, alpha=2e-5, beta=0.75, bias=1.0)
maxpool1 = tf.nn.max_pool(lrn1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID')

# CONV2) 256 filters. 5 x 5 kernel. 1 x 1 strides.
# -- ReLU
# -- Output normalization
# -- Maxpooling. 3 x 3 kernel, 2 x 2 strides.
conv2 = tf.nn.relu(conv(maxpool1, conv2W, conv2b, 5, 5, 256, 1, 1, padding="SAME", group=2))
lrn2 = tf.nn.local_response_normalization(conv2, depth_radius=2, alpha=2e-5, beta=0.75, bias=1.0)
maxpool2 = tf.nn.max_pool(lrn2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID')

# CONV3) 384 filters. 3 x 3 kernel. 1 x 1 strides.
# -- ReLU
conv3 = tf.nn.relu(conv(maxpool2, conv3W, conv3b, 3, 3, 384, 1, 1, padding="SAME", group=1))

# CONV4) 384 filters. 3 x 3 kernel. 1 x 1 strides.
# -- ReLU
conv4 = tf.nn.relu(conv(conv3, conv4W, conv4b, 3, 3, 384, 1, 1, padding="SAME", group=2))

# CONV5) 256 filters. 5 x 5 kernel. 1 x 1 strides.
# -- ReLU
# -- Maxpooling. 3 x 3 kernel, 2 x 2 strides.
conv5 = tf.nn.relu(conv(conv4, conv5W, conv5b, 3, 3, 256, 1, 1, padding="SAME", group=2))
maxpool5 = tf.nn.max_pool(conv5, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID')

# FC6)
fc6 = tf.nn.relu_layer(tf.reshape(maxpool5, [-1, int(np.prod(maxpool5.get_shape()[1:]))]), fc6W, fc6b)

# FC7)
fc7 = tf.nn.relu_layer(fc6, fc7W, fc7b)

# FC8)
fc8 = tf.nn.xw_plus_b(fc7, fc8W, fc8b)

# Softmax layer
y_softmax = tf.nn.softmax(fc8)
y_ = tf.reduce_mean(y_softmax, 0)

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
output = sess.run(y_, feed_dict={x: im1_crop})

top_5 = sess.run(tf.nn.top_k(y_, 5), feed_dict={x: im1_crop})
for k in range(5):
    print("%5.5f\t%s" % (top_5[0][k], class_names[top_5[1][k]]))
