
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import Counter
import os
import zipfile
import random
import math

import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector
import numpy as np
from six.moves import urllib

DATA_FOLDER = '/home/barnaud/dev/ML/text8_data/'
FILE_NAME = 'text8.zip'
DOWNLOAD_URL = 'http://mattmahoney.net/dc/'
VOCAB_SIZE = 50000
BATCH_SIZE = 128
EMBED_SIZE = 128 # dimension of the word embedding vectors
SKIP_WINDOW = 1 # the context window
NUM_SAMPLED = 64    # Number of negative examples to sample.
LEARNING_RATE = 1.0
NUM_TRAIN_STEPS = 100000
WEIGHTS_FLD = 'processed/'
SKIP_STEP = 2000 # how many steps to skip before reporting the loss

def download_data(file_name):
	path = DATA_FOLDER + file_name
	if os.path.exists(path):
		print("Data already downloaded. Skipping...")
		return path
	file_name, _ = urllib.request.urlretrieve(DOWNLOAD_URL + file_name, path)
	file_stat = os.stat(path)
	if file_stat.st_size > 0:
		print("Downloaded bytes.")
	else:
		raise Exception('File' + file_name + 'might be corrupted. You should download it manually.')
	return path

def read_data(path):
	with zipfile.ZipFile(path) as f:
		words = tf.compat.as_str(f.read(f.namelist()[0])).split()
	return words

def build_vocab(words, vocab_size):
	dictionary = dict()
	count = [('UNK', -1)]
	count.extend(Counter(words).most_common(vocab_size-1))
	index = 0
	with open(WEIGHTS_FLD + 'vocab_1000.tsv', "w") as f:
		for word, _ in count:
			dictionary[word] = index
			if index < 1000:
				f.write(word + "\n")
			index += 1
		index_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
	return dictionary, index_dictionary

def convert_words_to_index(words, dictionary):
	""" Replace each word in the dataset with its index in the dictionary """
	return [dictionary[word] if word in dictionary else 0 for word in words]

def generate_sample(index_words, context_window_size):
	for index, center in enumerate(index_words):
		context = random.randint(1, context_window_size)
		for target in index_words[max(0, index - context): index]:
			yield center, target
		# get a random target after the center wrod
		for target in index_words[index + 1: index + context + 1]:
			yield center, target

def get_batch(iterator, batch_size):
	""" Group a numerical stream into batches and yield them as Numpy arrays. """
	while True:
		center_batch = np.zeros(batch_size, dtype=np.int32)
		target_batch = np.zeros([batch_size, 1])
		for index in range(batch_size):
			center_batch[index], target_batch[index] = next(iterator)
		yield center_batch, target_batch


class SkipGramModel:

	def __init__(self, vocab_size, embed_size, batch_size, num_sampled, learning_rate):
		self.vocab_size = vocab_size
		self.embed_size = embed_size
		self.batch_size = batch_size
		self.num_sampled = num_sampled
		self.lr = learning_rate
		self.global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')


	def _create_placeholders(self):
		with tf.name_scope('data'):
			self.center_words = tf.placeholder(tf.int32, shape=[self.batch_size], name='center_words')
			self.target_words = tf.placeholder(tf.int32, shape=[self.batch_size, 1], name='target_words')

	def _create_embedding(self):
		with tf.name_scope('embed'):
			self.embed_matrix = tf.Variable(tf.random_uniform([self.vocab_size, self.embed_size], -1.0, 1.0), name='embed_matrix')


	def _create_loss(self):
		with tf.name_scope('loss'):
			embed = tf.nn.embedding_lookup(self.embed_matrix, self.center_words, name='embed')

			nce_weight = tf.Variable(tf.truncated_normal([VOCAB_SIZE, EMBED_SIZE], stddev = 1.0 / math.sqrt(EMBED_SIZE)), name='nce_weights')
			nce_bias = tf.Variable(tf.zeros([VOCAB_SIZE]), name='nce_bias')

			self.loss = tf.reduce_mean(tf.nn.nce_loss(weights=nce_weight,biases=nce_bias,labels=self.target_words, inputs=embed, 
												  num_sampled=self.num_sampled, num_classes=self.vocab_size), name='loss')

	def _create_optimizer(self):
		self.optimizer = tf.train.GradientDescentOptimizer(self.lr).minimize(self.loss, global_step=self.global_step)

	def _create_summaries(self):
		with tf.name_scope("summaries"):
			tf.summary.scalar("loss", self.loss)
			tf.summary.histogram("histogram loss", self.loss)
			self.summary_op = tf.summary.merge_all()

	def build_graph(self):
		self._create_placeholders()
		self._create_embedding()
		self._create_loss()
		self._create_optimizer()
		self._create_summaries()


def train_model(model, batch_gen, num_train_steps):
	saver = tf.train.Saver()

	initial_step = 0

	try:
		os.mkdir('checkpoints')
	except OSError, e:
		if e.errno != 17:
			raise
		pass
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())

		#Handle checkpoints
		ckpt = tf.train.get_checkpoint_state(os.path.dirname('checkpoints/checkpoint'))
		# if that checkpoint exists, restore from checkpoint
		if ckpt and ckpt.model_checkpoint_path:
			saver.restore(sess, ckpt.model_checkpoint_path)
			print("Model already trained.")
		else:
			total_loss = 0.0
			writer = tf.summary.FileWriter('improved_graph/lr' + str(LEARNING_RATE), sess.graph)
			initial_step = model.global_step.eval()
			for i in range(initial_step, initial_step + num_train_steps):
				centers, targets = next(batch_gen)
				feed_dict={model.center_words: centers, model.target_words: targets}
				loss_batch, _, summary= sess.run([model.loss, model.optimizer, model.summary_op], feed_dict=feed_dict)
				writer.add_summary(summary, global_step=i)
				total_loss += loss_batch
				if(i + 1) % SKIP_STEP == 0:
					print('Average loss at step {}: {:5.1f}'.format(i, total_loss/SKIP_STEP))
					total_loss = 0.0
					saver.save(sess, 'checkpoints/skip-gram', i)
			# code to visualize the embeddings. uncomment the below to visualize embeddings
	        # run "'tensorboard --logdir='processed'" to see the embeddings
			final_embed_matrix = sess.run(model.embed_matrix)
	        
	        # # it has to variable. constants don't work here. you can't reuse model.embed_matrix
			embedding_var = tf.Variable(final_embed_matrix[:1000], name='embedding')
			sess.run(embedding_var.initializer)

			config = projector.ProjectorConfig()
			summary_writer = tf.summary.FileWriter('processed')

	        # # add embedding to the config file
			embedding = config.embeddings.add()
			embedding.tensor_name = embedding_var.name
	        
	        # # link this tensor to its metadata file, in this case the first 500 words of vocab
			embedding.metadata_path = 'processed/vocab_1000.tsv'

	        # # saves a configuration file that TensorBoard will read during startup.
			projector.visualize_embeddings(summary_writer, config)
			saver_embed = tf.train.Saver([embedding_var])
			saver_embed.save(sess, 'processed/model3.ckpt', 1)

def main():
	file_path = download_data(FILE_NAME)
	words = read_data(file_path)
	dictionary, _ = build_vocab(words, VOCAB_SIZE)
	index_words = convert_words_to_index(words, dictionary)
	del words
	single_gen = generate_sample(index_words, SKIP_WINDOW)
	batch_generated = get_batch(single_gen, BATCH_SIZE)
	model = SkipGramModel(VOCAB_SIZE, EMBED_SIZE, BATCH_SIZE, NUM_SAMPLED, LEARNING_RATE)
	model.build_graph()
	train_model(model, batch_generated, NUM_TRAIN_STEPS)

if __name__ == "__main__":
	main()
