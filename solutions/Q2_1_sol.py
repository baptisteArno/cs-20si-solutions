# -*- coding: utf-8 -*-

from __future__ import print_function

import time
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# Get Data
MNIST = input_data.read_data_sets("MNIST_data/", one_hot=True)

# Parameters
learning_rate = 0.01
batch_size = 100
n_epochs = 20

# Placeholders
X = tf.placeholder(tf.float32, [None, 784])
Y_ = tf.placeholder(tf.float32, [None, 10])

# Definitions for the model
def weight_variable(shape):
	initial = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(initial)
def bias_variable(shape):
	initial = tf.constant(0.1, shape=shape)
	return tf.Variable(initial)

def conv2d(x, W):
	return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
def max_pool_2x2(x):
	return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

# The Model

#First Conv layer
X_reshaped = tf.reshape(X, [-1, 28, 28, 1])
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(X_reshaped, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

#Second Conv Layer
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

#Connected Layer
W_c1 = weight_variable([7*7*64, 1024])
b_c1 = weight_variable([1024])
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_c1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_c1) + b_c1)

#Dropout
keep_prob = tf.placeholder(tf.float32)
h_c1_drop = tf.nn.dropout(h_c1, keep_prob)

#Final decision layer
W_c2 = weight_variable([1024, 10])
b_c2 = weight_variable([10])
Y = tf.matmul(h_c1_drop, W_c2) + b_c2

# Loss function
entropy = tf.nn.softmax_cross_entropy_with_logits(labels=Y_, logits=Y)
loss = tf.reduce_mean(entropy) # computes the mean over examples in the batch

# Training (minimizing loss)
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

acc = tf.equal(tf.argmax(Y_, 1), tf.argmax(Y, 1))
acc = tf.reduce_mean(tf.cast(acc, tf.float32))

tf.summary.scalar('loss', loss)
tf.summary.scalar('accuracy', acc)
merged_summary = tf.summary.merge_all()

init = tf.global_variables_initializer()

with tf.Session() as sess:
	sess.run(init)
	writer = tf.summary.FileWriter("./graphs", sess.graph)
	for i in range(1800):
		X_batch, Y_batch = MNIST.train.next_batch(batch_size)
		_, _, summary = sess.run([optimizer, loss, merged_summary], feed_dict={X: X_batch, Y_: Y_batch, keep_prob: 1.0})
		writer.add_summary(summary, i)
		if i%100 == 0:
			 print ("Accuracy:", acc.eval(feed_dict={X: X_batch, Y_: Y_batch, keep_prob: 1.0}))
	print ("Accuracy:", acc.eval(feed_dict={X: MNIST.test.images, Y_: MNIST.test.labels, keep_prob: 1.0}))