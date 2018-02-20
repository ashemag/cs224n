#!/usr/bin/env python
# -*- coding: utf-8 -*-
import random
import tensorflow as tf

class Model:
	def __init__(self, name):
		self.name = name
		self.session = tf.Session()

	# Generates a list of batches for training all with a size |batch_size|
	# (except for possibly the last batch)
	@staticmethod
	def generate_batches(x, y, batch_size):
		data = zip(x, y)
		random.shuffle(data)
		return [ tuple(zip(*data[i:i+batch_size])) for i in range(len(data) / batch_size + 1) ]

	# Generates a random sample from (x, y) of size |sample_size|
	@staticmethod
	def random_sample(x, y, sample_size):
		return zip(*random.sample(zip(x, y), min(sample_size, len(x))))

	def train(self, x_train, y_train, x_dev=None, y_dev=None, num_epochs=0, batch_size=0, lr=0, verbose=False):
		raise NotImplementedError('Class ({0}) must implement predict().'.format(self.name))

	def predict(self, x):
		raise NotImplementedError('Class ({0}) must implement predict().'.format(self.name))


# Debugging / Testing code
if __name__ == "__main__":
	pass
