import datetime
import numpy as np
import tensorflow as tf

from DataSet import *
from Model import *
from util import Progbar

class CNNModel(Model):
	def __init__(self, embedding_size, num_classes, max_comment_length):
		Model.__init__(self, 'CNNModel')

		self.embedding_size = embedding_size
		self.num_classes = num_classes
		self.max_comment_length = max_comment_length
		self.input_length = self.embedding_size*self.max_comment_length

		self.x = tf.placeholder(tf.float32, shape=(None, self.input_length))
		self.y = tf.placeholder(tf.float32, shape=(None, self.num_classes))

		W = tf.get_variable('W', shape=(self.input_length, self.num_classes))
		b = tf.get_variable('b', shape=(self.num_classes, ))
		self.y_hat = tf.nn.sigmoid(tf.matmul(self.x, W) + b)
		
		self.lr = tf.placeholder(tf.float32, shape=())
		self.loss = tf.reduce_mean(tf.squared_difference(self.y, self.y_hat))
		self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

		self.session.run(tf.global_variables_initializer())

	def transform_input(self, x):
		x_flat = [ x_value.flatten() for x_value in x ]
		return [ np.pad(x_value, (self.input_length - x_value.shape[0], 0), 'constant') for x_value in x_flat ]

	def train(self, x_train,
					y_train,
					x_dev=None,
					y_dev=None,
					num_epochs=10,
					batch_size=100,
					lr=1e-3,
					verbose=False):
		x_train = self.transform_input(x_train)
		start_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
		print 'Training Model "{0}" (started at {1})...'.format(self.name, start_time)
		
		for epoch in range(num_epochs):
			print 'Epoch #{0} out of {1}'.format(epoch+1, num_epochs)
			batches = Model.generate_batches(x_train, y_train, batch_size)
			progbar = Progbar(target=len(batches))

			for batch, (x_batch, y_batch) in enumerate(batches):
				train_loss, _ = self.session.run((self.loss, self.train_op), feed_dict={ self.x: x_batch, self.y: y_batch, self.lr: lr })
				progbar.update(batch + 1, [("Train Loss", train_loss)])

			if x_dev is not None and y_dev is not None:
				x_batch, y_batch = Model.random_sample(x_dev, y_dev, batch_size)
				dev_loss = self.session.run(self.loss, feed_dict = { self.x: x_batch, self.y: y_batch, self.lr: lr }) 
				print 'Dev Loss: {0}'.format(dev_loss)

	def predict(self, x):
		return self.session.run(self.y_hat, feed_dict={self.x: self.transform_input(x)})


# Debugging / Testing code
if __name__ == "__main__":
	feature_extractor = OneHotFeatureExtractor()
	dataset = DataSet(DataSet.TRAIN_CSV, feature_extractor, count=200, verbose=True)
	x, y = dataset.get_data()

	embedding_size = x[0].shape[1]
	num_classes = y[0].shape[0]
	max_comment_length = max(x_value.shape[0] for x_value in x)

	cnn = CNNModel(embedding_size, num_classes, max_comment_length)
	cnn.train(x, y, num_epochs=10)
	print cnn.predict(x)
