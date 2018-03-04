import cPickle as pickle
import datetime
import numpy as np
import random
import tensorflow as tf

from dataset import *
from feature_extractor import *
from model import *
from util import Progbar

class CNNModel(Model):
	def __init__(self, vocab_size, num_classes, comment_length):
		self.graph = tf.Graph()
		with self.graph.as_default():
			Model.__init__(self, 'CNNModel')

			self.vocab_size = vocab_size
			self.num_classes = num_classes
			self.comment_length = comment_length

			self.embedding_size = 200
			self.capitalization_size = 3
			self.word_vector_size = self.embedding_size + self.capitalization_size
			self.input_length = self.word_vector_size * self.comment_length

			self.words_placeholder = tf.placeholder(tf.int32, shape=(None, self.comment_length), name='words')
			self.capitals_placeholder = tf.placeholder(tf.int32, shape=(None, self.comment_length), name='capitals') 


			self.E_words = tf.get_variable('E_words', shape=(self.vocab_size, self.embedding_size), initializer=tf.contrib.layers.xavier_initializer())
			self.E_words = tf.concat([self.E_words, np.zeros((1, self.embedding_size)).astype(np.float32)], axis=0)
			self.E_capitals = tf.constant(self.one_hot_embedding_matrix(self.capitalization_size), name='E_capitals')#tf.get_variable('E_capitals', initializer=self.one_hot_embedding_matrix(self.capitalization_size))
			words = tf.nn.embedding_lookup(self.E_words, self.words_placeholder)
			capitals = tf.nn.embedding_lookup(self.E_capitals, self.capitals_placeholder)

			x = tf.reshape(tf.concat([words, capitals], 2), (-1, 1, self.input_length, 1))
			self.y = tf.placeholder(tf.float32, shape=(None, self.num_classes))

			self.conv1 = tf.layers.conv2d(
				inputs = x,
				filters = 64,
				strides = (1, self.word_vector_size),
				kernel_size = (1, 5*(self.embedding_size+3)),
				padding = 'same',
				kernel_initializer = tf.contrib.layers.xavier_initializer(),
				bias_initializer = tf.zeros_initializer(),
				activation = tf.nn.elu,
				name = 'conv1'
			)

			self.dense1 = tf.layers.dense(
				inputs = tf.layers.flatten(self.conv1),
				units = 50,
				kernel_initializer = tf.contrib.layers.xavier_initializer(),
				bias_initializer = tf.zeros_initializer(),
				activation = tf.nn.elu,
				name = 'dense1'
			)

			self.y_hat = tf.layers.dense(
				inputs = self.dense1,
				units = self.num_classes,
				kernel_initializer = tf.contrib.layers.xavier_initializer(),
				bias_initializer = tf.zeros_initializer(),
				activation = tf.nn.sigmoid,
				name = 'y_hat'
			)
			
			self.lr = tf.placeholder(tf.float64, shape=())
			self.loss = tf.reduce_mean(-self.y*self.log_epsilon(self.y_hat) - (1-self.y)*self.log_epsilon(1-self.y_hat))
			self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)
			self.auroc_scores = [tf.metrics.auc(self.y[:, klass], self.y_hat[:, klass]) for klass in range(self.num_classes)]

			self.session.run(tf.global_variables_initializer())


	def one_hot_embedding_matrix(self, size):
		return np.concatenate((np.eye(size), np.zeros((1, size)))).astype(np.float32)

	def log_epsilon(self, tensor):
		return tf.log(tensor + 1e-10)

	def compute_auroc_scores(self, x_data, y_data):
		with self.graph.as_default():
			self.session.run(tf.local_variables_initializer())
			scores = [self.auroc_scores[klass][0] for klass in range(num_classes)]
			update_ops = [self.auroc_scores[klass][1] for klass in range(num_classes)]

			words, capitals = zip(*x_data)
			self.session.run(update_ops, feed_dict = {
				self.words_placeholder: words,
				self.capitals_placeholder: capitals,
				self.y: y_data
			})

			return self.session.run(scores)

	def save(self, filename):
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
		print 'Training Model "{0}" (started at {1})...'.format(self.name, start_time)
		
		max_auroc = 0
		for epoch in range(num_epochs):
			print 'Epoch #{0} out of {1}'.format(epoch+1, num_epochs)
			num_batches = int(np.ceil(len(x_train)/batch_size))
			progbar = Progbar(target=num_batches)

			for batch, (x_batch, y_batch) in enumerate(Model.generate_batches(x_train, y_train, batch_size)):
				words, capitals = zip(*x_batch)
				train_loss, _ = self.session.run((self.loss, self.train_op), feed_dict={
					self.words_placeholder: words,
					self.capitals_placeholder: capitals,
					self.y: y_batch,
					self.lr: lr
				})

				mean_auroc = np.mean(self.compute_auroc_scores(x_batch, y_batch))
				progbar.update(batch + 1, [("Train Loss", train_loss), ("Mean AUROC", mean_auroc)])

			if x_dev is not None and y_dev is not None:
				words, capitals = zip(*x_dev)
				dev_loss = self.session.run(self.loss, feed_dict = {
					self.words_placeholder: words,
					self.capitals_placeholder: capitals,
					self.y: y_dev
				})
				auroc_scores = self.compute_auroc_scores(x_dev, y_dev)
				mean_auroc = np.mean(auroc_scores)

				marker = ""
				if mean_auroc >= max_auroc:
					max_auroc = mean_auroc
					self.save("models/{0}/{0}".format(self.name))
					marker = "*"
				
				print 'Dev Loss: {0}'.format(dev_loss)
				print 'AUROC Scores: {0}'.format(auroc_scores)
				print 'Mean AUROC Score: {0} {1}\n'.format(mean_auroc, marker)

	def predict(self, x):
		words, capitals = zip(*x)
		return self.session.run(self.y_hat, feed_dict={self.words_placeholder: words, self.capitals_placeholder: capitals})


# Debugging / Testing code
if __name__ == "__main__":
	train = True
	seed = 13
	np.random.seed(seed)
	random.seed(seed)
	np.set_printoptions(precision=4, suppress=True)

	num_classes = 6
	comment_length = 100

	feature_extractor = OneHotFeatureExtractor(comment_length) 
	train_dataset = DataSet(DataSet.TRAIN_CSV, feature_extractor, count=None, verbose=True)
	vocab_size = len(train_dataset.vocab)
	cnn = CNNModel(vocab_size, num_classes, comment_length)

	if train:
		x, y = train_dataset.get_data()

		index = 140000
		x_train, x_dev = x[:index], x[index:]
		y_train, y_dev = y[:index], y[index:]
		
		cnn.train(x_train, y_train, x_dev, y_dev, num_epochs=10)

	else:
		cnn.load('models/CNNModel/CNNModel')
		feature_extractor = OneHotFeatureExtractor(comment_length, train_dataset.vocab)
		del train_dataset.comments
		test_dataset = DataSet(DataSet.TEST_CSV, feature_extractor, test=True, verbose=True)
		x, y = test_dataset.get_data()

		with open('submissions/cnn_model.csv', 'wb') as csv_file:
			csv_writer = csv.writer(csv_file)
			csv_writer.writerow(['id','toxic','severe_toxic','obscene','threat','insult','identity_hate'])
			
			batch_size = 1000
			num_batches = int(np.ceil(len(x)/float(batch_size)))
			for i in range(num_batches):
				if i % 10 == 0: print i
				predictions = cnn.predict(x[batch_size*i:batch_size*(i+1)])
				for j in range(len(predictions)):
					row = [test_dataset.comments[batch_size*i+j].example_id]
					row.extend(predictions[j])
					csv_writer.writerow(row)






