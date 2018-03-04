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
	def __init__(self, vocab_size, n_classes=6, n_features=2, comment_length=100):
		Model.__init__(self, 'LSTMModel')
		
		with tf.variable_scope(type(self).__name__):
			self.vocab_size = vocab_size
			self.comment_length = comment_length
			self.labels_placeholder = tf.placeholder(tf.float32, [None, n_classes])
			self.hidden_states = 24 
			self.n_classes = n_classes

			self.words_placeholder = tf.placeholder(tf.int32, shape=(None, self.comment_length), name='words')
			self.capitals_placeholder = tf.placeholder(tf.int32, shape=(None, self.comment_length), name='capitals') 

			self.embedding_size = 200
			self.capitalization_size = 3
			words, capitals = Model.generate_embeddings(self)

			inputs = tf.concat([words, capitals], 2)
			
			cell = tf.nn.rnn_cell.LSTMCell(self.hidden_states,state_is_tuple=True, reuse = tf.AUTO_REUSE)
			output, state = tf.nn.dynamic_rnn(cell, inputs, dtype=tf.float32)

			#We transpose the output to switch batch size with sequence size
			output = tf.transpose(output, [1, 0, 2])
			last = tf.gather(output, int(output.get_shape()[0]) - 1)

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

			self.auroc_scores = [tf.metrics.auc(self.labels_placeholder[:, klass], self.prediction[:, klass]) for klass in range(self.n_classes)]
			self.session.run(tf.global_variables_initializer())
	
	def train_helper(self, epoch, num_epochs, x_train, y_train, batch_size, lr): 
		print 'Epoch #{0} out of {1}'.format(epoch+1, num_epochs)
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

			mean_auroc = np.mean(self.compute_auroc_scores(x_batch, y_batch))
			progbar.update(batch + 1, [("Train Loss", train_loss, "Mean Score", mean_auroc)])	
		return train_loss
	
	def train(self, x_train, y_train, x_dev, y_dev, num_epochs=10, batch_size=100, lr=1e-4): 
		start_time = int(round(time.time() * 1000))
		print 'Training LSTM Model "{0}" (started at {1})...'.format(self.name, start_time)
		
		train_losses = []
		epochs = []
		for epoch in range(num_epochs):
			train_loss = self.train_helper(epoch, num_epochs, x_train, y_train, batch_size, lr)
			train_losses.append(train_loss)
			epochs.append(epoch)

			mean_auroc = np.mean(Model.compute_auroc_scores(self, x_dev, y_dev))
			print "Dev Set Mean AUROC: {0}\n".format(mean_auroc)


		end_time = int(round(time.time() * 1000))
		
		print 'Training LSTM Model took {0} seconds.'.format((end_time-start_time) / 1000.0)
		return train_losses, epochs  

	def predict(self, comments, x_dev, y_dev=None, batch_size=100, filename="submissions/lstm_model.csv"): 
		with open(filename, 'wb') as csv_file:
			csv_writer = csv.writer(csv_file)
			csv_writer.writerow(['id','toxic','severe_toxic','obscene','threat','insult','identity_hate'])
			num_batches = int(np.ceil(len(x)/float(batch_size)))
			for i in range(num_batches):
				predictions, scores = self.predict_helper(x_dev[batch_size*i:batch_size*(i+1)], y_dev)
				for j in range(len(predictions)):
					row = [comments[batch_size*i+j].example_id]
					row.extend(predictions[j])
					csv_writer.writerow(row)

	def predict_helper(self, x_dev, y_dev=None): 
		words, capitals = zip(*x_dev)
		preds = self.session.run(self.prediction, {
			self.words_placeholder: words,
			self.capitals_placeholder: capitals
		})
		scores = None 
		if y_dev is not None: 
			scores = self.get_scores(y_dev, preds)
		return preds, scores
	
	def close(self): 
		self.session.close() 

# Debugging / Testing code
if __name__ == "__main__":
	max_comment_length = 100 
	feature_extractor = OneHotFeatureExtractor(max_comment_length)
	
	train_data = DataSet(DataSet.TRAIN_CSV, feature_extractor, count=100, verbose=True) 
	x, y = train_data.get_data()
	# DEV_SPLIT = 140000
	DEV_SPLIT = len(y) / 2
	x_train, x_dev = x[:DEV_SPLIT], x[DEV_SPLIT:]
	y_train, y_dev = y[:DEV_SPLIT], y[DEV_SPLIT:]
	
	num_epochs = 1

	lstm = LSTM(len(train_data.vocab))
	train_losses, epochs = lstm.train(x_train, y_train, x_dev, y_dev, num_epochs = num_epochs)
	
	feature_extractor = OneHotFeatureExtractor(100, train_data.vocab)
	del train_data.comments
	test_data = DataSet(DataSet.TEST_CSV, feature_extractor, test=True, verbose=True) 	
	x, y = test_data.get_data()
	lstm.predict(test_data.comments, x)
	lstm.close() 

