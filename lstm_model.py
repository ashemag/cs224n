import numpy as np
import tensorflow as tf

from feature_extractor import * 
from dataset import *
from model import *
from util import Progbar

class LSTM(Model):
	def __init__(self, train_input, train_output, test_input, test_ids, n_classes=6, n_features=2, max_comment_length = 100):
		self.inputs_placeholder = tf.placeholder(tf.float32, [None, n_features, max_comment_length]) #[Batch Size, Sequence Length, Input Dimension]
		self.labels_placeholder = tf.placeholder(tf.float32, [None, n_classes])
		self.hidden_states = 24 
		self.train_input = train_input
		self.train_output = train_output
		self.test_input = test_input
		self.test_output = test_output
		self.test_ids = test_ids 

	def write_submission(self, preds, filename='submission.csv'):
		with open(filename, 'wb') as csvfile: 
			writer = csv.writer(csvfile)
			fieldnames = ["id","toxic","severe_toxic","obscene","threat","insult","identity_hate"]
			writer.writerow(fieldnames)
			for i, comment_id in enumerate(self.test_ids): 
				preds[i]
				entry = [comment_id] + list(preds[i])
				writer.writerow(entry)


	def run_model(self): 
		data, target = self.inputs_placeholder, self.labels_placeholder
		cell = tf.nn.rnn_cell.LSTMCell(self.hidden_states,state_is_tuple=True)
		output, state = tf.nn.dynamic_rnn(cell, data, dtype=tf.float32)
		
		#We transpose the output to switch batch size with sequence size
		output = tf.transpose(output, [1, 0, 2])
		last = tf.gather(output, int(output.get_shape()[0]) - 1)

		weight = tf.Variable(tf.truncated_normal([self.hidden_states, int(target.get_shape()[1])]))
		bias = tf.Variable(tf.constant(0.1, shape=[target.get_shape()[1]]))

		prediction = tf.nn.softmax(tf.matmul(last, weight) + bias)
		cross_entropy = -tf.reduce_sum(target * tf.log(tf.clip_by_value(prediction,1e-10,1.0)))

		optimizer = tf.train.AdamOptimizer()
		minimize = optimizer.minimize(cross_entropy)

		mistakes = tf.not_equal(tf.argmax(target, 1), tf.argmax(prediction, 1))
		error = tf.reduce_mean(tf.cast(mistakes, tf.float32))

		init_op = tf.initialize_all_variables()
		sess = tf.Session()
		sess.run(init_op)

		batch_size = 1000
		no_of_batches = int(len(self.train_input)/batch_size)
		epoch = 100 #Epoch 5000 error 28.9%
		for i in range(epoch):
		    ptr = 0
		    for j in range(no_of_batches):
		        inp, out = self.train_input[ptr:ptr+batch_size], self.train_output[ptr:ptr+batch_size]
		        ptr+=batch_size
		        sess.run(minimize,{data: inp, target: out})
		    print "Epoch - ",str(i)
		# incorrect = sess.run(error,{data: self.test_input, target: self.test_output})
		# print('Epoch {:2d} error {:3.1f}%'.format(i + 1, 100 * incorrect))
		
		preds = sess.run(prediction, {data: self.test_input})
		sess.close() 
		self.write_submission(preds)

# Debugging / Testing code
if __name__ == "__main__":
	max_comment_length = 100 
	feature_extractor = OneHotFeatureExtractor(max_comment_length)
	
	train_data = DataUtil(True, feature_extractor, verbose=True) 
	train_input, train_output, ids = train_data.get_data()
	
	test_data = DataUtil(False, feature_extractor, verbose=True) 
	test_input, test_output, ids = test_data.get_data()
	print test_data == train_data

	example = LSTM(train_input, train_output, test_input, ids)
	example.run_model() 

