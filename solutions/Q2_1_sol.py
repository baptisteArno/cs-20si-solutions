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
n_epochs = 25

# Placeholders
X = tf.placeholder(tf.float32, [None, 784])
Y_ = tf.placeholder(tf.float32, [None, 10])

# Variables
W = tf.Variable(tf.random_normal(shape=[784, 10], stddev=0.01), name="Weigths")
b = tf.Variable(tf.zeros([1,10]), name="Bias")

# The function
Y = tf.nn.softmax(tf.add(tf.matmul(X,W), b))

# Loss function
entropy = tf.nn.softmax_cross_entropy_with_logits(labels=Y_, logits=Y)
loss = tf.reduce_mean(entropy) # computes the mean over examples in the batch

# Training (minimizing loss)
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

acc = tf.equal(tf.argmax(Y_, 1), tf.argmax(Y, 1))
acc = tf.reduce_mean(tf.cast(acc, tf.float32))

tf.summary.scalar('loss', loss)
tf.summary.scalar('accuracy', acc)
merged_summary = tf.summary.merge_all()

init = tf.global_variables_initializer()

with tf.Session() as sess:
	sess.run(init)
	writer = tf.summary.FileWriter("./graphs", sess.graph)
	pos = 1
	for epoch in range(n_epochs):
		avg_loss = 0;
		n_batches = int(MNIST.train.num_examples/batch_size) 
		for i in range(n_batches):
			X_batch, Y_batch = MNIST.train.next_batch(batch_size)
			_, l, summary = sess.run([optimizer, loss, merged_summary], feed_dict={X: X_batch, Y_: Y_batch})
			writer.add_summary(summary, pos)
			avg_loss = l / n_batches
		print('Epoch :', epoch, 'AvgLoss =', avg_loss)
	print ("Accuracy:", acc.eval(feed_dict={X: MNIST.test.images, Y_: MNIST.test.labels}))