import datetime
import numpy as np
import os
import random
import tensorflow as tf

from dataset import *
from feature_extractor import *
from Model import *
from util import Progbar

class BiLSTM(Model):
	def __init__(self, vocab, num_classes, comment_length):
		self.graph = tf.Graph()
		with self.graph.as_default():
			Model.__init__(self, 'BiLSTM')

			embedding_size = 200
			capitalization_size = 3
			word_vector_size = embedding_size + capitalization_size
			input_length = word_vector_size * comment_length

			self.words_placeholder = tf.placeholder(tf.int32, shape=(None, comment_length), name='words')
			self.capitals_placeholder = tf.placeholder(tf.int32, shape=(None, comment_length), name='capitals') 

			#Random embeddings: Comment out to avoid duplicate TF variables 
			#words, capitals = self.generate_random_embeddings(vocab, embedding_size, trainable=True)

			#Pretrained embeddings 
			words, capitals = self.generate_pretrained_embeddings(vocab, embedding_size, trainable=True)

			x = tf.concat([words, capitals], 2)
			self.y = tf.placeholder(tf.float32, shape=(None, num_classes))

			self.dropout = tf.placeholder(tf.float32, shape=())
			num_cells = 256
			num_layers = 3

			cell_fw = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.LSTMCell(num_cells), output_keep_prob=1-self.dropout) for _ in range(num_layers)])
			cell_bw = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.LSTMCell(num_cells), output_keep_prob=1-self.dropout) for _ in range(num_layers)])
			outputs, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, inputs=x, dtype=tf.float32)
			
			# The average of outputs across all time steps
			# rnn_output = tf.reduce_mean(tf.concat(outputs, 2), axis=1) 
			# rnn_output = tf.concat((outputs[0][:,-1,:],outputs[1][:,0,:]), 1)
			rnn_output = tf.reduce_max(tf.concat(outputs, 2), axis=1)
			
			dense1 = tf.layers.dense(
				inputs = tf.nn.dropout(
					x = rnn_output,
					keep_prob = 1-self.dropout
				),
				units = 64,
				kernel_initializer = tf.contrib.layers.xavier_initializer(),
				bias_initializer = tf.zeros_initializer(),
				activation = tf.nn.elu,
				name = 'dense1'
			)

			self.y_hat = tf.layers.dense(
				inputs = dense1,
				units = num_classes,
				kernel_initializer = tf.contrib.layers.xavier_initializer(),
				bias_initializer = tf.zeros_initializer(),
				activation = tf.nn.sigmoid,
				name = 'y_hat'
			)
			
			self.lr = tf.placeholder(tf.float32, shape=())
			self.losses = -self.y*self.log_epsilon(self.y_hat) - (1-self.y)*self.log_epsilon(1-self.y_hat)
			self.loss = tf.reduce_mean(self.losses)
			self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

			self.session.run(tf.global_variables_initializer())


	def one_hot_embedding_matrix(self, size):
		return np.concatenate((np.eye(size), np.zeros((1, size)))).astype(np.float32)

	def log_epsilon(self, tensor):
		return tf.log(tensor + 1e-10)

	def save(self, filename):
		directory = os.path.dirname(filename)
		if not os.path.exists(directory):
			os.makedirs(directory)
		
		with self.graph.as_default():
			tf.train.Saver().save(self.session, filename)

	def load(self, filename):
		with self.graph.as_default():
			tf.train.Saver().restore(self.session, filename)

	def train(self, x_train,
					y_train,
					x_dev=None,
					y_dev=None,
					num_epochs=10,
					batch_size=100,
					lr=1e-3,
					verbose=False):

		start_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
		print('Training Model "{0}" (started at {1})...'.format(self.name, start_time))
		
		max_auroc = 0
		for epoch in range(num_epochs):
			print('Epoch #{0} out of {1}'.format(epoch+1, num_epochs))
			num_batches = int(np.ceil(len(x_train)/batch_size))
			progbar = Progbar(target=num_batches)

			for batch, (x_batch, y_batch) in enumerate(Model.generate_batches(x_train, y_train, batch_size)):
				words, capitals = zip(*x_batch)
				train_loss, _ = self.session.run((self.loss, self.train_op), feed_dict={
					self.words_placeholder: words,
					self.capitals_placeholder: capitals,
					self.y: y_batch,
					self.lr: lr,
					self.dropout: 0.5
				})

				mean_auroc = 0#np.mean(self.compute_auroc_scores(x_batch, y_batch))
				progbar.update(batch + 1, [("Train Loss", train_loss), ("Mean AUROC", mean_auroc)])

			if x_dev is not None and y_dev is not None:
				words, capitals = zip(*x_dev)
				dev_loss = self.session.run(self.loss, feed_dict = {
					self.words_placeholder: words,
					self.capitals_placeholder: capitals,
					self.y: y_dev,
					self.dropout: 0.0
				})
				auroc_scores = self.compute_auroc_scores(x_dev, y_dev)
				mean_auroc = np.mean(auroc_scores)

				marker = ""
				if mean_auroc >= max_auroc:
					max_auroc = mean_auroc
					self.save("models/{0}/{0}".format(self.name))
					marker = "*"
				
				print('Dev Loss: {0}'.format(dev_loss))
				print('Accuracies: {0}'.format(self.compute_accuracies(x_dev,y_dev)))
				print('AUROC Scores: {0}'.format(auroc_scores))
				print('Mean AUROC Score: {0} {1}\n'.format(mean_auroc, marker))
	
	def predict(self, x):
		words, capitals = zip(*x)
		return self.session.run(self.y_hat, feed_dict={
			self.words_placeholder: words,
			self.capitals_placeholder: capitals,
			self.dropout: 0.0
		})

# Debugging / Testing code
if __name__ == "__main__": 
	train = False#True
	seed = 13
	np.random.seed(seed)
	random.seed(seed)
	np.set_printoptions(precision=4, suppress=True)

	num_classes = 6
	comment_length = 100

	feature_extractor = OneHotFeatureExtractor(comment_length) 
	train_dataset = DataSet(DataSet.TRAIN_CSV, feature_extractor, use_glove=True, verbose=True)
	model = BiLSTM(train_dataset.vocab, num_classes, comment_length)

	if train:
		x, y = train_dataset.get_data()
		
		index = 140000
		x_train, x_dev = x[:index], x[index:]
		y_train, y_dev = y[:index], y[index:]

		model.train(x_train, y_train, x_dev, y_dev, num_epochs=10)

	else:
		model.load('models/BiLSTM/BiLSTM')
		feature_extractor = OneHotFeatureExtractor(comment_length, train_dataset.vocab)
		test_dataset = DataSet(DataSet.TEST_CSV, feature_extractor, test=True, verbose=True)
		model.write_predictions_to_file(test_dataset, 'submissions/bi_lstm_model.csv')





