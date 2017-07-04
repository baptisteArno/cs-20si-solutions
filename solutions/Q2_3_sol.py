# -*- coding: utf-8 -*-

from __future__ import print_function

import sys
import time
import numpy as np
import tensorflow as tf
import pandas as pd

DATA_PATH = '../ressources/assignment_1/heart.txt'

#Read and format data
data_x = pd.read_csv(DATA_PATH, sep=" ", usecols=range(9)).as_matrix()
data_y = pd.read_csv(DATA_PATH, sep=" ", usecols=["chd"]).as_matrix()

train_x, test_x = data_x[:308, :], data_x[308:, :]
train_y, test_y = data_y[:308, :], data_y[308:, :]

X = tf.placeholder(tf.float32, [None, 9])
Y_ = tf.placeholder(tf.float32, [None, 1])

W = tf.Variable(tf.truncated_normal([9, 1]))
b = tf.Variable(tf.zeros([1]))

# Parameters
learning_rate = 0.001

#The model
Y = tf.matmul(X,W) + b

# Loss function
entropy = tf.nn.softmax_cross_entropy_with_logits(labels=Y_, logits=Y)
loss = tf.reduce_mean(entropy) # computes the mean over examples in the batch

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
	for i in range(5):
		_, l, summary = sess.run([optimizer, loss, merged_summary], feed_dict={X: train_x, Y_: train_y})
		writer.add_summary(summary, i)
		if i%100 == 0:
			print (i)
	print ("Accuracy:", acc.eval(feed_dict={X: test_x, Y_: test_y}))