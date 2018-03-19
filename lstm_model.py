#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import tensorflow as tf

from feature_extractor import * 
from dataset import *
from Model import *
from util import Progbar
import datetime

class LSTM(Model):
	def __init__(self, vocab, comments, comment_length, n_classes=6, n_features=2):
		Model.__init__(self, 'LSTMModel')
		
		with tf.variable_scope(type(self).__name__):
			self.vocab_size = len(vocab)
			self.vocab = vocab 
			self.comments = comments 
			self.comment_length = comment_length
			self.labels_placeholder = tf.placeholder(tf.float32, [None, n_classes])
			self.hidden_states = 24 
			self.n_classes = n_classes

			self.words_placeholder = tf.placeholder(tf.int32, shape=(None, self.comment_length), name='words')
			self.capitals_placeholder = tf.placeholder(tf.int32, shape=(None, self.comment_length), name='capitals') 
			self.inputs_placeholder = tf.placeholder(tf.int32, shape=(None, self.comment_length), name='inputs') 

			self.embedding_size = 200
			self.capitalization_size = 3

			#Random embeddings: Comment out to avoid duplicate TF variables 
			words, capitals = self.generate_random_embeddings(vocab, trainable=False)
			
			#character level modeling 
			# words, capitals = self.generate_one_hot_embeddings(vocab)

			#Pretrained GloVe embeddings 
			# words, capitals = self.generate_pretrained_embeddings(self.vocab, trainable=False)

			inputs = tf.concat([words, capitals], 2)
			cell = tf.contrib.rnn.LSTMCell(self.hidden_states, state_is_tuple=True, reuse=None)
			output, state = tf.nn.dynamic_rnn(cell, inputs, dtype=tf.float32)

			#We transpose the output to switch batch size with sequence size
			#output = tf.transpose(output, [1, 0, 2])
			#last = tf.gather(output, int(output.get_shape()[0]) - 1)
			last = tf.reduce_mean(output, axis=1)

			weight = tf.Variable(tf.truncated_normal([self.hidden_states, 50]))
			bias = tf.Variable(tf.constant(0.1, shape=[50]))
			layer = tf.nn.elu(tf.matmul(last, weight) + bias)

			self.prediction = tf.layers.dense(
				inputs = layer,
				units = self.n_classes,
				kernel_initializer = tf.contrib.layers.xavier_initializer(),
				bias_initializer = tf.zeros_initializer(),
				activation = tf.nn.sigmoid,
				name = 'y_hat'
			)

			self.lr = tf.placeholder(tf.float64, shape=())
			self.loss = tf.reduce_mean(-self.labels_placeholder*tf.log(tf.clip_by_value(self.prediction,1e-10,1.0)) -(1-self.labels_placeholder)*tf.log(tf.clip_by_value(1-self.prediction,1e-10,1.0)))
			self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

			#self.auroc_scores = [tf.metrics.auc(self.labels_placeholder[:, klass], self.prediction[:, klass]) for klass in range(self.n_classes)]
			self.session.run(tf.global_variables_initializer())
	
	def train_helper(self, epoch, num_epochs, x_train, y_train, batch_size, lr): 
		print('Epoch #{0} out of {1}'.format(epoch+1, num_epochs))
		num_batches = int(np.ceil(len(x_train)/batch_size))
		progbar = Progbar(target=num_batches)

		train_loss = 0 
		for batch, (x_batch, y_batch) in enumerate(Model.generate_batches(x_train, y_train, batch_size)):
			words, capitals = zip(*x_batch)
		
			train_loss, _ = self.session.run((self.loss, self.train_op), {
				self.words_placeholder: words,
				self.capitals_placeholder: capitals,
				self.labels_placeholder: y_batch, 
				self.lr: lr
			})

			#mean_auroc = np.mean(self.compute_auroc_scores(x_batch, y_batch))
			mean_auroc = 0 
			progbar.update(batch + 1, [("Train Loss", train_loss), ("Mean AUROC", mean_auroc)])	
		return train_loss
	
	def train(self, x_train, y_train, x_dev, y_dev, num_epochs=10, batch_size=100, lr=1e-4): 
		start_time = int(round(time.time() * 1000))
		print('Training LSTM Model "{0}" (started at {1})...'.format(self.name, start_time))
		
		train_losses = []
		epochs = []
		for epoch in range(num_epochs):
			train_loss = self.train_helper(epoch, num_epochs, x_train, y_train, batch_size, lr)
			train_losses.append(train_loss)
			epochs.append(epoch)

			words, capitals = zip(*x_dev)
			dev_loss = self.session.run(self.loss, {
				self.words_placeholder: words,
				self.capitals_placeholder: capitals,
				self.labels_placeholder: y_dev, 
			})

			mean_auroc = np.mean(self.compute_auroc_scores(x_dev, y_dev))
			print("Dev Set - Loss: {0:.4f}, Mean AUROC: {1:.4f}\n".format(dev_loss, mean_auroc))
			print('Accuracies: {0}'.format(self.compute_accuracies(x_dev,y_dev)))

		end_time = int(round(time.time() * 1000))
		print('Training LSTM Model took {0} seconds.'.format((end_time-start_time) / 1000.0))
		
		return train_losses, epochs  

	def predict(self, x):
		words, capitals = zip(*x)
		return self.session.run(self.prediction, feed_dict={
			self.words_placeholder: words,
			self.capitals_placeholder: capitals
		})

	def write_predictions_to_file(self, comments, x_dev, y_dev=None, batch_size=100, filename="submissions/lstm_model.csv"): 
		with open(filename, 'w') as csv_file:
			csv_writer = csv.writer(csv_file)
			csv_writer.writerow(['id','toxic','severe_toxic','obscene','threat','insult','identity_hate'])
			num_batches = int(np.ceil(len(x)/float(batch_size)))
			for i in range(num_batches):
				predictions = self.predict(x_dev[batch_size*i:batch_size*(i+1)])
				for j in range(len(predictions)):
					row = [comments[batch_size*i+j].example_id]
					row.extend(predictions[j])
					csv_writer.writerow(row)
	
	def close(self): 
		self.session.close() 

# Debugging / Testing code
if __name__ == "__main__":
	max_comment_length = 100 
	feature_extractor = OneHotFeatureExtractor(max_comment_length)
	
	train_data = DataSet(DataSet.TRAIN_CSV, feature_extractor, verbose=True, use_glove=False, character_level=True) 
	x, y = train_data.get_data()
	
	DEV_SPLIT = 150000
	#DEV_SPLIT = 1000
	#DEV_SPLIT2 = -20000
	x_train, x_dev = x[:DEV_SPLIT], x[DEV_SPLIT:]
	y_train, y_dev = y[:DEV_SPLIT], y[DEV_SPLIT:]
	
	num_epochs = 5

	lstm = LSTM(train_data.vocab, train_data.comments, comment_length=max_comment_length * 5)
	train_losses, epochs = lstm.train(x_train, y_train, x_dev, y_dev, num_epochs = num_epochs)
	
	feature_extractor = OneHotFeatureExtractor(max_comment_length, train_data.vocab)
	del train_data.comments
	test_data = DataSet(DataSet.TEST_CSV, feature_extractor, test=True, verbose=True, use_glove=False, character_level=True) 	
	x, y = test_data.get_data()
	
	lstm.write_predictions_to_file(test_data.comments, x)
	lstm.close() 

