#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_auc_score

from feature_extractor import * 
from DataSet import *
from Model import *
from util import Progbar
import datetime

class LSTM(Model):
	def __init__(self, n_classes=6, n_features=2, max_comment_length=100):
		Model.__init__(self, 'LSTMModel')

		self.inputs_placeholder = tf.placeholder(tf.float32, [None, n_features, max_comment_length]) #[Batch Size, Sequence Length, Input Dimension]
		self.labels_placeholder = tf.placeholder(tf.float32, [None, n_classes])
		self.preds = tf.placeholder(tf.float32, [None, n_classes])
		self.hidden_states = 24 
		self.n_classes = n_classes

		self.auroc_scores = [tf.metrics.auc(self.labels_placeholder[:, klass], self.preds[:, klass]) for klass in range(self.n_classes)]
		self.session.run(tf.global_variables_initializer())
	
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

	def train(self, 
			x_train, 
			y_train, 
			x_dev=None, 
			y_dev=None, 
			num_epochs=10, 
			batch_size=100, 
			lr=1e-3,
			verbose=False):

		start_time = int(round(time.time() * 1000))
		print 'Training LSTM Model "{0}" (started at {1})...'.format(self.name, start_time)
		
		cell = tf.nn.rnn_cell.LSTMCell(self.hidden_states,state_is_tuple=True)
		output, state = tf.nn.dynamic_rnn(cell, self.inputs_placeholder, dtype=tf.float32)

		#We transpose the output to switch batch size with sequence size
		output = tf.transpose(output, [1, 0, 2])
		last = tf.gather(output, int(output.get_shape()[0]) - 1)

		weight = tf.Variable(tf.truncated_normal([self.hidden_states, int(self.labels_placeholder.get_shape()[1])]))
		bias = tf.Variable(tf.constant(0.1, shape=[self.labels_placeholder.get_shape()[1]]))
		optimizer = tf.train.AdamOptimizer()

		prediction = tf.nn.softmax(tf.matmul(last, weight) + bias)
		cross_entropy = -tf.reduce_sum(self.labels_placeholder * tf.log(tf.clip_by_value(prediction,1e-10,1.0)))
		minimize = optimizer.minimize(cross_entropy)
		
		mistakes = tf.not_equal(tf.argmax(self.labels_placeholder, 1), tf.argmax(prediction, 1))
		error = tf.reduce_mean(tf.cast(mistakes, tf.float32))
		
		self.session.run(tf.global_variables_initializer())

		for epoch in range(num_epochs):
			print 'Epoch #{0} out of {1}'.format(epoch+1, num_epochs)
			num_batches = int(np.ceil(len(x_train)/batch_size))
			progbar = Progbar(target=num_batches)

			for batch, (x_batch, y_batch) in enumerate(Model.generate_batches(x_train, y_train, batch_size)):
				self.session.run(minimize, {self.inputs_placeholder: x_batch, self.labels_placeholder: y_batch}) 

		if x_dev is not None and y_dev is not None:
			preds = self.session.run(prediction, {self.inputs_placeholder: x_dev})
			self.get_scores(y_dev, preds)
			
		end_time = int(round(time.time() * 1000))
		print 'Training LSTM Model took {0} seconds.'.format((end_time-start_time) / 1000.0)
	
	def close(self): 
		self.session.close() 

# Debugging / Testing code
if __name__ == "__main__":
	if True: 
		max_comment_length = 100 
		feature_extractor = OneHotFeatureExtractor(max_comment_length)
		
		train_data = DataSet(DataSet.TRAIN_CSV, feature_extractor, verbose=True) 
		x, y = train_data.get_data()
		DEV_SPLIT = len(y) / 2
		x_train, x_dev = x[:DEV_SPLIT], x[DEV_SPLIT:]
		y_train, y_dev = y[:DEV_SPLIT], y[DEV_SPLIT:]

		lstm = LSTM()
		lstm.train(x_train, y_train, x_dev, y_dev)
		lstm.close() 
		