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

			self.vocab_size = len(vocab)
			self.vocab = vocab
			self.num_classes = num_classes
			self.comment_length = comment_length

			self.embedding_size = 200
			self.capitalization_size = 3

			self.words_placeholder = tf.placeholder(tf.int32, shape=(None, self.comment_length), name='words')
			self.capitals_placeholder = tf.placeholder(tf.int32, shape=(None, self.comment_length), name='capitals') 

			# Random embeddings: Comment out to avoid duplicate TF variables 
			words, capitals = self.generate_random_embeddings()
			
			# Pretrained embeddings 
			# words, capitals = self.generate_pretrained_embeddings(self.vocab)

			x = tf.concat([words, capitals], 2)
			self.y = tf.placeholder(tf.float32, shape=(None, self.num_classes))
			
			cell_fw = tf.contrib.rnn.LSTMCell(128)
			cell_bw = tf.contrib.rnn.LSTMCell(128)
			outputs, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, inputs=x, dtype=tf.float32)
			
			# The average of outputs across all time steps
			# rnn_output = tf.reduce_mean(tf.concat(outputs, 2), axis=1) 
			rnn_output = tf.concat((outputs[0][:,-1,:],outputs[1][:,0,:]), 1)

			dense1 = tf.layers.dense(
				inputs = rnn_output,
				units = 64,
				kernel_initializer = tf.contrib.layers.xavier_initializer(),
				bias_initializer = tf.zeros_initializer(),
				activation = tf.nn.elu,
				name = 'dense1'
			)

			self.y_hat = tf.layers.dense(
				inputs = dense1,
				units = self.num_classes,
				kernel_initializer = tf.contrib.layers.xavier_initializer(),
				bias_initializer = tf.zeros_initializer(),
				activation = tf.nn.sigmoid,
				name = 'y_hat'
			)
			
			self.lr = tf.placeholder(tf.float32, shape=())
			self.losses = -self.y*self.log_epsilon(self.y_hat) - (1-self.y)*self.log_epsilon(1-self.y_hat)
			self.loss = tf.reduce_mean(self.losses)
			self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)
			self.auroc_scores = [tf.metrics.auc(self.y[:, klass], self.y_hat[:, klass]) for klass in range(self.num_classes)]

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
				})

				mean_auroc = 0#np.mean(self.compute_auroc_scores(x_batch, y_batch))
				progbar.update(batch + 1, [("Train Loss", train_loss), ("Mean AUROC", mean_auroc)])

			if x_dev is not None and y_dev is not None:
				words, capitals = zip(*x_dev)
				dev_loss = self.session.run(self.loss, feed_dict = {
					self.words_placeholder: words,
					self.capitals_placeholder: capitals,
					self.y: y_dev,
				})
				auroc_scores = self.compute_auroc_scores(x_dev, y_dev)
				mean_auroc = np.mean(auroc_scores)

				marker = ""
				if mean_auroc >= max_auroc:
					max_auroc = mean_auroc
					self.save("models/{0}/{0}".format(self.name))
					marker = "*"
				
				print('Dev Loss: {0}'.format(dev_loss))
				print('AUROC Scores: {0}'.format(auroc_scores))
				print('Mean AUROC Score: {0} {1}\n'.format(mean_auroc, marker))

	def compute_auroc_scores(self, x_data, y_data):
		with self.graph.as_default():
			self.session.run(tf.local_variables_initializer())
			scores = [self.auroc_scores[klass][0] for klass in range(self.num_classes)]
			update_ops = [self.auroc_scores[klass][1] for klass in range(self.num_classes)]
			words, capitals = zip(*x_data)
			self.session.run(update_ops, feed_dict = {
				self.words_placeholder: words,
				self.capitals_placeholder: capitals,
				self.y: y_data,
			})
			return self.session.run(scores)
	
	def predict(self, x):
		words, capitals = zip(*x)
		return self.session.run(self.y_hat, feed_dict={
			self.words_placeholder: words,
			self.capitals_placeholder: capitals,
		})

	def get_losses(self, x, y):
		words, capitals = zip(*x)
		losses = self.session.run(self.losses, feed_dict={
			self.words_placeholder: words,
			self.capitals_placeholder: capitals,
			self.y: y,
		})
		return np.mean(losses, axis=1)

# Debugging / Testing code
if __name__ == "__main__":
	train = False
	seed = 13
	np.random.seed(seed)
	random.seed(seed)
	np.set_printoptions(precision=4, suppress=True)

	num_classes = 6
	comment_length = 200

	feature_extractor = OneHotFeatureExtractor(comment_length) 
	train_dataset = DataSet(DataSet.TRAIN_CSV, feature_extractor, count=None, verbose=True)
	model = BiLSTM(train_dataset.vocab, num_classes, comment_length)

	if train:
		x, y = train_dataset.get_data()
		
		index = 140000
		x_train, x_dev = x[:index], x[index:]
		y_train, y_dev = y[:index], y[index:]

		model.train(x_train, y_train, x_dev, y_dev, num_epochs=10)
		#model.load('models/BiLSTM/BiLSTM')
		'''
		predictions, losses = model.predict(x_dev), model.get_losses(x_dev, y_dev)

		results = zip(train_dataset.comments[index:], x_dev, y_dev, predictions, losses)
		results = sorted(results, key=lambda x: x[-1], reverse=True)

		for comment, x, y, y_hat, loss in results[:10]:
			print('Comment: {0}'.format(comment.words))
			print('X:     {0}'.format(x))
			print('Y:     {0}'.format(y))
			print('Y_hat: {0}'.format(y_hat))
			print('Loss: {0}\n'.format(loss))
		'''

	else:
		model.load('models/BiLSTM/BiLSTM')
		feature_extractor = OneHotFeatureExtractor(comment_length, train_dataset.vocab)
		test_dataset = DataSet(DataSet.TEST_CSV, feature_extractor, test=True, verbose=True)
		x, y = test_dataset.get_data()

		with open('submissions/bi_lstm_model.csv', 'w') as csv_file:
			csv_writer = csv.writer(csv_file)
			csv_writer.writerow(['id','toxic','severe_toxic','obscene','threat','insult','identity_hate'])
			
			batch_size = 1000
			num_batches = int(np.ceil(len(x)/float(batch_size)))
			for i in range(num_batches):
				if i % 10 == 0: print(i)
				predictions = model.predict(x[batch_size*i:batch_size*(i+1)])
				for j in range(len(predictions)):
					row = [test_dataset.comments[batch_size*i+j].example_id]
					row.extend(predictions[j])
					csv_writer.writerow(row)





