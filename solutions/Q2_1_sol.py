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
W = tf.Variable(tf.random_normal(shape=[784, 10], stddev=0.01), name="weigths")
b = tf.Variable(tf.zeros([1,10]), name="bias")

# The function
Y = tf.nn.softmax(tf.add(tf.matmul(X,W), b))

# Loss function
loss = tf.reduce_mean(entropy)

# Training (minimizing loss)
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

init = tf.global_variables_initializer()

with tf.Session() as sess:
	sess.run(init)
	n_batches = int(MNIST.train.num_examples/batch_size)
	for i in range(n_epochs):
		for _ in range(n_batches):
			X_batch, Y_batch = MNIST.train.next_batch(batch_size)
			sess.run([optimizer, loss], feed_dict={X: X_batch, Y_: Y_batch})

	correct_prediction = tf.equal(tf.argmax(Y,1), tf.argmax(Y_,1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
	print(sess.run(accuracy, feed_dict={X: MNIST.test.images, Y_: MNIST.test.labels}))
	writer = tf.summary.FileWriter('./graphs', sess.graph)

writer.close()