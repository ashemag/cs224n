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
			#self.inputs_placeholder = tf.placeholder(tf.float32, [None, n_features, max_comment_length]) #[Batch Size, Sequence Length, Input Dimension]
			self.labels_placeholder = tf.placeholder(tf.float32, [None, n_classes])
			self.hidden_states = 24 
			self.n_classes = n_classes
			self.prediction = None 

			self.words_placeholder = tf.placeholder(tf.int32, shape=(None, self.comment_length), name='words')
			self.capitals_placeholder = tf.placeholder(tf.int32, shape=(None, self.comment_length), name='capitals') 

			self.embedding_size = 200
			self.capitalization_size = 3
			self.word_vector_size = self.embedding_size + self.capitalization_size
			self.input_length = self.word_vector_size * self.comment_length

			self.E_words = tf.get_variable('E_words', shape=(self.vocab_size, self.embedding_size), initializer=tf.contrib.layers.xavier_initializer())
			self.E_words = tf.concat([self.E_words, np.zeros((1, self.embedding_size)).astype(np.float32)], axis=0)
			self.E_capitals = tf.constant(self.one_hot_embedding_matrix(self.capitalization_size), name='E_capitals')#tf.get_variable('E_capitals', initializer=self.one_hot_embedding_matrix(self.capitalization_size))
			words = tf.nn.embedding_lookup(self.E_words, self.words_placeholder)
			capitals = tf.nn.embedding_lookup(self.E_capitals, self.capitals_placeholder)

			inputs = tf.concat([words, capitals], 2)
			print(inputs.get_shape())
			# inputs = tf.reshape(tf.concat([words, capitals], 2), shape=(-1, 1, self.input_length, 1))
			
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
			#self.prediction = tf.nn.softmax(tf.matmul(last, weight) + bias)
			#self.loss = -tf.reduce_sum(self.labels_placeholder * tf.log(tf.clip_by_value(self.prediction,1e-10,1.0)))
			self.loss = tf.reduce_mean(-self.labels_placeholder*tf.log(tf.clip_by_value(self.prediction,1e-10,1.0)) -(1-self.labels_placeholder)*tf.log(tf.clip_by_value(1-self.prediction,1e-10,1.0)))
			self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

			self.auroc_scores = [tf.metrics.auc(self.labels_placeholder[:, klass], self.prediction[:, klass]) for klass in range(self.n_classes)]

			self.session.run(tf.global_variables_initializer())
	
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

	def train_helper(self, epoch, num_epochs, x_train, y_train, batch_size, lr): 
		print 'Epoch #{0} out of {1}'.format(epoch+1, num_epochs)
		num_batches = int(np.ceil(len(x_train)/batch_size))
		progbar = Progbar(target=num_batches)

		#train_loss = 0 
		for batch, (x_batch, y_batch) in enumerate(Model.generate_batches(x_train, y_train, batch_size)):
			words, capitals = zip(*x_batch)
			train_loss, _ = self.session.run((self.loss, self.train_op), {
				self.words_placeholder: words,
				self.capitals_placeholder: capitals,
				#self.inputs_placeholder: x_batch, 
				self.labels_placeholder: y_batch, 
				self.lr: lr
			})

			mean_auroc = 0#np.mean(self.compute_auroc_scores(x_batch, y_batch))
			progbar.update(batch + 1, [("Train Loss", train_loss), ("Mean AUROC", mean_auroc)])	
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

			mean_auroc = np.mean(self.compute_auroc_scores(x_dev, y_dev))
			print "Dev Set Mean AUROC: {0}\n".format(mean_auroc)


		end_time = int(round(time.time() * 1000))
		
		print 'Training LSTM Model took {0} seconds.'.format((end_time-start_time) / 1000.0)
		return train_losses, epochs  

	def predict(self, x_dev, y_dev=None): 
		#preds = self.session.run(self.prediction, {self.inputs_placeholder: x_dev})
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
<<<<<<< HEAD
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
=======
	if True: 
		max_comment_length = 100 
		feature_extractor = OneHotFeatureExtractor(max_comment_length)
		
		train_data = DataSet(DataSet.TRAIN_CSV, feature_extractor, count=None, verbose=True) 
		x, y = train_data.get_data()
		DEV_SPLIT = 140000#len(y) / 2
		x_train, x_dev = x[:DEV_SPLIT], x[DEV_SPLIT:]
		y_train, y_dev = y[:DEV_SPLIT], y[DEV_SPLIT:]
		
		num_epochs = 2

		lstm = LSTM(len(train_data.vocab), name=str(num_epochs))
		train_losses, epochs = lstm.train(x_train, y_train, x_dev, y_dev, num_epochs = num_epochs)
		preds, scores = lstm.predict(x_dev, y_dev)
		plot(epochs, train_losses)
		
		feature_extractor = OneHotFeatureExtractor(100, train_data.vocab)
		del train_data.comments
		test_data = DataSet(DataSet.TEST_CSV, feature_extractor, test=True, verbose=True) 
>>>>>>> 336dfe3e9ae6760294cb53e39a35183991c85c21

	x, y = test_data.get_data()

<<<<<<< HEAD
	preds, scores = lstm.predict(x)
	test_ids = [comment.example_id for comment in test_data.comments]
	lstm.write_submission(test_ids, preds, "submissions/lstm.csv")
	lstm.close() 

=======
		preds, scores = lstm.predict(x)
		test_ids = [comment.example_id for comment in test_data.comments]
		lstm.write_submission(test_ids, preds, "submissions/lstm.csv")
		lstm.close()
>>>>>>> 336dfe3e9ae6760294cb53e39a35183991c85c21
		