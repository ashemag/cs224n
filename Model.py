#!/usr/bin/env python
# -*- coding: utf-8 -*-
import csv
import math
import random
import tensorflow as tf
import numpy as np 

from sklearn.metrics import roc_auc_score

class Model:
	def __init__(self, name):
		print("Initializing Model...")
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
	
	def compute_accuracies(self, x_data, y_data):
		correct_labels = np.equal(np.round(self.predict(x_data)), y_data)
		return np.mean(correct_labels, axis=0)

	@staticmethod
	def one_hot_embedding_matrix(size):
		return np.concatenate((np.eye(size), np.zeros((1, size)))).astype(np.float32)

	@staticmethod
	def create_twitter_embeddings(vocab, embedding_size): 
		embeddings_index = {}
		print("=== Processing Glove Twitter embeddings ===")
		with open('glove.twitter.27B/glove.twitter.27B.{0}d.txt'.format(embedding_size)) as f:
		 	for line in f:
		 		values = line.split(' ')
		 		word = values[0]
		 		embeddings_index[word] = np.asarray(values[1:], dtype='float64')
		print("=== Glove Twitter embeddings processed ===")
		
		all_embeddings = np.array(list(embeddings_index.values()))
		mean, std = np.mean(all_embeddings), np.std(all_embeddings)

		embeddings = []
		for word in vocab:
			embed = embeddings_index.get(word)
			if embed is not None: 
				embeddings.append(embed)
			else: 
				embeddings.append(np.random.normal(mean, std, embedding_size))
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

	def generate_pretrained_embeddings(self, vocab, embedding_size=200, trainable=False): 
		E_words = tf.get_variable(
			name = 'E_words',
			initializer = self.create_twitter_embeddings(vocab, embedding_size),
			trainable = trainable,
		) 
		E_words = tf.concat([E_words, np.zeros((1, embedding_size)).astype(np.float32)], axis=0)
		
		capitalization_size = 3
		E_capitals = tf.constant(self.one_hot_embedding_matrix(capitalization_size), name='E_capitals')
		
		words = tf.nn.embedding_lookup(E_words, self.words_placeholder)
		capitals = tf.nn.embedding_lookup(E_capitals, self.capitals_placeholder)
		return words, capitals
	
	def generate_one_hot_embeddings(self, vocab):
		E_words = tf.constant(self.one_hot_embedding_matrix(len(vocab)), name='E_words')
		E_words = tf.concat([E_words, np.zeros((1, len(vocab))).astype(np.float32)], axis=0)
		
		capitalization_size = 3
		E_capitals = tf.constant(self.one_hot_embedding_matrix(capitalization_size), name='E_capitals')
		words = tf.nn.embedding_lookup(E_words, self.words_placeholder)
		capitals = tf.nn.embedding_lookup(E_capitals, self.capitals_placeholder)
		return words, capitals

	def generate_random_embeddings(self, vocab, embedding_size=200, trainable=False):
		E_words = tf.get_variable(
			name = 'E_words',
			shape = (len(vocab), embedding_size),
			initializer=tf.contrib.layers.xavier_initializer(),
			trainable = trainable
		) 
		E_words = tf.concat([E_words, np.zeros((1, embedding_size)).astype(np.float32)], axis=0)
		capitalization_size = 3
		E_capitals = tf.constant(self.one_hot_embedding_matrix(capitalization_size), name='E_capitals')
		
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

	def write_predictions_to_file(self, test_dataset, filename):
		x, _ = test_dataset.get_data()
		with open(filename, 'w') as csv_file:
			csv_writer = csv.writer(csv_file)
			csv_writer.writerow(['id','toxic','severe_toxic','obscene','threat','insult','identity_hate'])
			
			batch_size = 1000
			num_batches = int(np.ceil(len(x)/float(batch_size)))
			for i in range(num_batches):
				if i % 10 == 0: print(i)
				predictions = self.predict(x[batch_size*i:batch_size*(i+1)])
				for j in range(len(predictions)):
					row = [test_dataset.comments[batch_size*i+j].example_id]
					row.extend(predictions[j])
					csv_writer.writerow(row)



