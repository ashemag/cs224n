#!/usr/bin/env python
# -*- coding: utf-8 -*-
import math
import random
import tensorflow as tf
import numpy as np 

class Model:
	def __init__(self, name):
		print "Initialized Model..."
		self.name = name
		self.session = tf.Session()

	# Generates a list of batches for training all with a size |batch_size|
	# (except for possibly the last batch)
	@staticmethod
	def generate_batches(x, y, batch_size):
		data = list(zip(x, y))
		random.shuffle(data)
		num_batches = int(math.ceil(len(x)/batch_size))
		for i in range(num_batches):
			yield tuple(zip(*data[batch_size*i:batch_size*(i+1)]))

	# Computes AUROC scores 
	def compute_auroc_scores(self, x_data, y_data):
		self.session.run(tf.local_variables_initializer())
		scores = [self.auroc_scores[klass][0] for klass in range(self.n_classes)]
		update_ops = [self.auroc_scores[klass][1] for klass in range(self.n_classes)]

		words, capitals = zip(*x_data)
		self.session.run(update_ops, feed_dict = {
			self.words_placeholder: words,
			self.capitals_placeholder: capitals,
			self.labels_placeholder: y_data
		})

		return self.session.run(scores)

	# Computes AUROC scores 2
	def get_scores(self, y, preds): 
		y, preds = np.array(y), np.array(preds)
		column = []
		#add each column score 
		for i in range(self.n_classes):
			try:
			    roc = roc_auc_score(y, preds)
			    column.append(roc)
			except ValueError:
			    pass				
		score = 0 if column == [] else np.mean(column)
		print "SCORE: ", score 
		return score 
	
	@staticmethod
	def one_hot_embedding_matrix(size):
		return np.concatenate((np.eye(size), np.zeros((1, size)))).astype(np.float32)

	#generate embeddings
	def generate_embeddings(self): 
		E_words = tf.get_variable('E_words', shape=(self.vocab_size, self.embedding_size), initializer=tf.contrib.layers.xavier_initializer()) 
		E_words = tf.concat([E_words, np.zeros((1, self.embedding_size)).astype(np.float32)], axis=0)
		E_capitals = tf.constant(self.one_hot_embedding_matrix(self.capitalization_size), name='E_capitals')
		words = tf.nn.embedding_lookup(E_words, self.words_placeholder)
		capitals = tf.nn.embedding_lookup(E_capitals, self.capitals_placeholder)
		return words, capitals 

	# Generates a random sample from (x, y) of size |sample_size|
	@staticmethod
	def random_sample(x, y, sample_size):
		return zip(*random.sample(zip(x, y), min(sample_size, len(x))))

	def train(self, x_train, y_train, x_dev=None, y_dev=None, num_epochs=0, batch_size=0, lr=0, verbose=False):
		raise NotImplementedError('Class ({0}) must implement predict().'.format(self.name))

	def predict(self, x):
		raise NotImplementedError('Class ({0}) must implement predict().'.format(self.name))

