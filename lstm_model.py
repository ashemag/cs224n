#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt

from feature_extractor import * 
from DataSet import *
from Model import *
from util import Progbar
import datetime

class LSTM(Model):
	def __init__(self, vocab_size, name, n_classes=6, n_features=2, comment_length=100):
		Model.__init__(self, 'LSTMModel')
		scope = type(self).__name__
		with tf.variable_scope(name):
			self.vocab_size = vocab_size
			self.comment_length = comment_length
			self.inputs_placeholder = tf.placeholder(tf.float32, [None, n_features, max_comment_length]) #[Batch Size, Sequence Length, Input Dimension]
			self.labels_placeholder = tf.placeholder(tf.float32, [None, n_classes])
			self.hidden_states = 24 
			self.n_classes = n_classes
			self.prediction = None 

			# self.words_placeholder = tf.placeholder(tf.int32, shape=(None, self.comment_length), name='words')
			# self.capitals_placeholder = tf.placeholder(tf.int32, shape=(None, self.comment_length), name='capitals') 

			# self.embedding_size = 200
			# self.capitalization_size = 3
			# self.word_vector_size = self.embedding_size + self.capitalization_size
			# self.input_length = self.word_vector_size * self.comment_length

			# self.E_words = tf.get_variable('E_words', shape=(self.vocab_size, self.embedding_size), initializer=tf.contrib.layers.xavier_initializer())
			# self.E_words = tf.concat([self.E_words, np.zeros((1, self.embedding_size)).astype(np.float32)], axis=0)
			# self.E_capitals = tf.constant(self.one_hot_embedding_matrix(self.capitalization_size), name='E_capitals')#tf.get_variable('E_capitals', initializer=self.one_hot_embedding_matrix(self.capitalization_size))
			# words = tf.nn.embedding_lookup(self.E_words, self.words_placeholder)
			# capitals = tf.nn.embedding_lookup(self.E_capitals, self.capitals_placeholder)

			# inputs = tf.reshape(tf.concat([words, capitals], 2), (-1, 1, self.input_length, 1))
			# inputs = tf.reshape(tf.concat([words, capitals], 2), shape=(-1, 1, self.input_length, 1))
			
			cell = tf.nn.rnn_cell.LSTMCell(self.hidden_states,state_is_tuple=True, reuse = tf.AUTO_REUSE)
			output, state = tf.nn.dynamic_rnn(cell, self.inputs_placeholder, dtype=tf.float32)

			#We transpose the output to switch batch size with sequence size
			output = tf.transpose(output, [1, 0, 2])
			last = tf.gather(output, int(output.get_shape()[0]) - 1)

			weight = tf.Variable(tf.truncated_normal([self.hidden_states, int(self.labels_placeholder.get_shape()[1])]))
			bias = tf.Variable(tf.constant(0.1, shape=[self.labels_placeholder.get_shape()[1]]))
			
			self.lr = tf.placeholder(tf.float64, shape=())
			self.prediction = tf.nn.softmax(tf.matmul(last, weight) + bias)
			self.loss = -tf.reduce_sum(self.labels_placeholder * tf.log(tf.clip_by_value(self.prediction,1e-10,1.0)))
			self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

			self.session.run(tf.global_variables_initializer())
			# tf.get_variable_scope().reuse_variables()
	
	def one_hot_embedding_matrix(self, size):
		return np.concatenate((np.eye(size), np.zeros((1, size)))).astype(np.float32)

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

	def train_helper(self, epoch, num_epochs, x_train, y_train, batch_size, lr=1e-3): 
		print 'Epoch #{0} out of {1}'.format(epoch+1, num_epochs)
		num_batches = int(np.ceil(len(x_train)/batch_size))
		train_loss = 0 
		for batch, (x_batch, y_batch) in enumerate(Model.generate_batches(x_train, y_train, batch_size)):
			words, capitals = zip(*x_batch)
			train_loss, _ = self.session.run((self.loss, self.train_op), {
				# self.words_placeholder: words,
				# self.capitals_placeholder: capitals,
				self.inputs_placeholder: x_batch, 
				self.labels_placeholder: y_batch, 
				self.lr: lr
				}) 			
		return train_loss
	
	def train(self, x_train, y_train, num_epochs=10, batch_size=100, lr=1e-3): 
		start_time = int(round(time.time() * 1000))
		print 'Training LSTM Model "{0}" (started at {1})...'.format(self.name, start_time)
		
		train_losses = []
		epochs = []
		for epoch in range(num_epochs):
			train_loss = self.train_helper(epoch, num_epochs, x_train, y_train, batch_size)
			train_losses.append(train_loss)
			epochs.append(epoch)

		end_time = int(round(time.time() * 1000))
		
		print 'Training LSTM Model took {0} seconds.'.format((end_time-start_time) / 1000.0)
		return train_losses, epochs  

	def predict(self, x_dev, y_dev=None): 
		preds = self.session.run(self.prediction, {self.inputs_placeholder: x_dev})
		scores = None 
		if y_dev is not None: 
			scores = self.get_scores(y_dev, preds)
		return preds, scores
	
	def close(self): 
		self.session.close() 
	
	def write_submission(self, test_ids, preds, filename):
 		with open(filename, 'wb') as csvfile: 
			writer = csv.writer(csvfile)
			fieldnames = ["id","toxic","severe_toxic","obscene","threat","insult","identity_hate"]
			writer.writerow(fieldnames)
			for i, comment_id in enumerate(test_ids): 
				preds[i]
				entry = [comment_id] + list(preds[i])
				writer.writerow(entry)

def plot(epochs, train_losses, title='Tuning Training Loss for LSTM'): 
	fig = plt.figure()
	ax = plt.subplot(111)
	plt.plot(epochs, train_losses, color = 'b')
	plt.title(title)
	plt.xlabel('Epoch')
	plt.ylabel('Training Loss')
	fig.savefig('submissions/lstm_training_loss.png')

# Debugging / Testing code
if __name__ == "__main__":
	if True: 
		max_comment_length = 100 
		feature_extractor = OneHotFeatureExtractor(max_comment_length)
		
		train_data = DataSet(DataSet.TRAIN_CSV, feature_extractor, count=100, verbose=True) 
		x, y = train_data.get_data()
		DEV_SPLIT = len(y) / 2
		x_train, x_dev = x[:DEV_SPLIT], x[DEV_SPLIT:]
		y_train, y_dev = y[:DEV_SPLIT], y[DEV_SPLIT:]
		
		num_epochs = 10

		lstm = LSTM(len(train_data.vocab), name=str(num_epochs))
		train_losses, epochs = lstm.train(x_train, y_train, num_epochs = num_epochs)
		preds, scores = lstm.predict(x_dev, y_dev)
		plot(epochs, train_losses)
		
		feature_extractor = OneHotFeatureExtractor(100, train_data.vocab)
		del train_data.comments
		test_data = DataSet(DataSet.TEST_CSV, feature_extractor, test=True, verbose=True) 

		x, y = test_data.get_data()

		preds, scores = lstm.predict(x)
		test_ids = [comment.example_id for comment in test_data.comments]
		lstm.write_submission(test_ids, preds, "submissions/lstm.csv")
		lstm.close() 

		