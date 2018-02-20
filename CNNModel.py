import cPickle as pickle
import datetime
import numpy as np
import tensorflow as tf

from DataSet import *
from Model import *
from util import Progbar

class CNNModel(Model):
	def __init__(self, vocab_size, num_classes, comment_length):
		Model.__init__(self, 'CNNModel')

		self.vocab_size = vocab_size
		self.num_classes = num_classes
		self.comment_length = comment_length
		self.input_length = (self.vocab_size + 3)*self.comment_length

		self.words_placeholder = tf.placeholder(tf.int32, shape=(None, self.comment_length), name='words')
		self.capitals_placeholder = tf.placeholder(tf.int32, shape=(None, self.comment_length), name='capitals') 

		E_words = tf.get_variable('E_words', initializer=self.one_hot_embedding_matrix(self.vocab_size))
		E_capitals = tf.get_variable('E_capitals', initializer=self.one_hot_embedding_matrix(3))
		words = tf.nn.embedding_lookup(E_words, self.words_placeholder)
		capitals = tf.nn.embedding_lookup(E_capitals, self.capitals_placeholder)

		x = tf.reshape(tf.concat([words, capitals], 2), (-1, 1, self.input_length, 1))
		#x = tf.concat((x, np.ones((1, self.input_length))), 1)
		self.y = tf.placeholder(tf.float32, shape=(None, self.num_classes))

		'''
		W = tf.get_variable('W', shape=(self.input_length, self.num_classes), dtype=tf.float64)
		b = tf.get_variable('b', shape=(self.num_classes, ), dtype=tf.float64)
		self.y_hat = tf.nn.sigmoid(tf.matmul(x, W) + b)
		'''

		conv1 = tf.layers.conv2d(
			inputs = x,
			filters = 32,
			strides = (1, self.vocab_size+3),
			kernel_size = (2, 5*(self.vocab_size+3)),
			padding = 'same',
			kernel_initializer = tf.contrib.layers.xavier_initializer(),
			bias_initializer = tf.zeros_initializer(),
			activation = tf.nn.elu,
			name = 'conv1'
		)

		dense1 = tf.layers.dense(
			inputs = tf.layers.flatten(conv1),
			units = 50,
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
		
		self.lr = tf.placeholder(tf.float64, shape=())
		#self.loss = tf.reduce_mean(tf.squared_difference(self.y, self.y_hat))
		weights = tf.reshape(tf.nn.softmax(tf.div(tf.norm(self.y, axis=1), 0.5)), (-1,1))
		self.loss = tf.losses.mean_squared_error(self.y, self.y_hat, weights)

		self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

		self.session.run(tf.global_variables_initializer())

	def one_hot_embedding_matrix(self, size):
		return np.concatenate((np.eye(size), np.zeros((1, size)))).astype(np.float32)

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
		
		for epoch in range(num_epochs):
			print 'Epoch #{0} out of {1}'.format(epoch+1, num_epochs)
			batches = Model.generate_batches(x_train, y_train, batch_size)
			progbar = Progbar(target=len(batches))

			for batch, (x_batch, y_batch) in enumerate(batches):
				words, capitals = zip(*x_batch)
				train_loss, _ = self.session.run((self.loss, self.train_op), feed_dict={
					self.words_placeholder: words,
					self.capitals_placeholder: capitals,
					self.y: y_batch,
					self.lr: lr
				})
				progbar.update(batch + 1, [("Train Loss", train_loss)])

			if x_dev is not None and y_dev is not None:
				x_batch, y_batch = Model.random_sample(x_dev, y_dev, batch_size)
				words, capitals = zip(*x_batch)
				dev_loss = self.session.run(self.loss, feed_dict = {
					self.words_placeholder: words,
					self.capitals_placeholder: capitals,
					self.y: y_batch
				}) 
				print 'Dev Loss: {0}'.format(dev_loss)

	def predict(self, x):
		words, capitals = zip(*x)
		return self.session.run(self.y_hat, feed_dict={self.words_placeholder: words, self.capitals_placeholder: capitals})


# Debugging / Testing code
if __name__ == "__main__":
	np.set_printoptions(precision=4, suppress=True)
	'''
	feature_extractor = OneHotFeatureExtractor()
	dataset = DataSet(DataSet.TRAIN_CSV, feature_extractor, count=200, verbose=True)
	x, y = dataset.get_data()
	'''

	dataset = pickle.load(open('one-hot-dataset-small.pkl', 'rb'))
	x, y = dataset.get_data()

	vocab_size = len(dataset.vocab)
	num_classes = y[0].shape[0]
	comment_length = 100

	cnn = CNNModel(vocab_size, num_classes, comment_length)
	x_train, x_dev = x[:800], x[800:]
	y_train, y_dev = y[:800], y[800:]

	cnn.train(x_train, y_train, x_dev, y_dev, num_epochs=10)
	print dataset.comments[827].words
	predictions = cnn.predict(x[827:837])
	print predictions
	print np.round(predictions)
