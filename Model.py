#!/usr/bin/env python
# -*- coding: utf-8 -*-
import math
import random
import tensorflow as tf
import numpy as np 

from sklearn.metrics import roc_auc_score

class Model:
	def __init__(self, name):
		print("Initialized Model...")
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
		preds = self.predict(x_data)
		y = np.array(y_data)
		return [ roc_auc_score(y[:, i], preds[:, i]) for i in range(y.shape[1]) ]
	
	@staticmethod
	def one_hot_embedding_matrix(size):
		return np.concatenate((np.eye(size), np.zeros((1, size)))).astype(np.float32)

	@staticmethod
	def create_twitter_embeddings(vocab): 
		embeddings_index = {}
		'''
		print("=== Processing Glove Twitter embeddings ===")
		with open('glove.twitter.27B/glove.twitter.27B.200d.txt') as f:
		 	for line in f:
		 		values = line.split(' ')
		 		word = values[0]
		 		embeddings_index[word] = np.asarray(values[1:], dtype='float64')
		print("=== Glove Twitter embeddings processed ===")
		'''
		
		embeddings = []
		for word in vocab:
			embed = embeddings_index.get(word)
			if embed is not None: 
				embeddings.append(embed)
			else: 
				embeddings.append(np.random.rand(200))
		return np.array(embeddings).astype(np.float32) 
		
		# embedding_dim = 200 
		# embed_comments = [] # Contains the 'average' embedding for each comment
		# for comment in comments:
		# 	avg_embed = np.zeros(embedding_dim) 
		# 	for word in comment.words:
		# 		embed = embeddings_index.get(word)
		# 		if embed is not None:
		# 			avg_embed += embed 
		# 	embed_comments.append(avg_embed/len(comment.words))
		# print("=== Average Embedding for each comment found ===")
		# embed_comments =  np.array(embed_comments) #comment size x embedding size 
		# return embed_comments
		



	#generate random embeddings
	def generate_pretrained_embeddings(self, vocab): 
		E_words = tf.get_variable('E_words', initializer=self.create_twitter_embeddings(self.vocab)) 
		E_words = tf.concat([E_words, np.zeros((1, self.embedding_size)).astype(np.float32)], axis=0)
		E_capitals = tf.constant(self.one_hot_embedding_matrix(self.capitalization_size), name='E_capitals')
		words = tf.nn.embedding_lookup(E_words, self.words_placeholder)
		capitals = tf.nn.embedding_lookup(E_capitals, self.capitals_placeholder)
		return words, capitals
		
		'''
		E_inputs = tf.get_variable('E_inputs', initializer=embeddings_initializer) #shape=(self.vocab_size, self.embedding_size)
		E_inputs = tf.cast(E_inputs, tf.float64)
		E_inputs = tf.concat([E_inputs, np.zeros((1, self.embedding_size)).astype(np.float64)], axis=0)#what is this? 
		inputs = tf.nn.embedding_lookup(E_inputs, self.inputs_placeholder)
		inputs = tf.cast(inputs,tf.float32)
		print(inputs.shape)
		return inputs
		'''

	#generate random embeddings
	def generate_random_embeddings(self, embeddings_initializer=tf.contrib.layers.xavier_initializer()): 
		E_words = tf.get_variable('E_words', shape=(self.vocab_size, self.embedding_size), initializer=embeddings_initializer) 
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

