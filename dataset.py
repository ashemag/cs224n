#!/usr/bin/env python
# -*- coding: utf-8 -*-
import collections 
import csv
import numpy as np
import re
import string
import time
import sys, os 

from feature_extractor import *

filename = 'one-hot-dataset-small.pkl'


class Comment: 
	def __init__(self, example_id, words, labels, chars):
		self.example_id = example_id
		self.words = words
		self.labels = labels
		self.chars = chars 

class DataSet:
	CLASSES = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
	TRAIN_CSV = "data/train.csv"
	TEST_CSV = "data/test.csv"
	UNKNOWN_WORD = "<unknown>"
	MIN_WORD_COUNT = 20

	# csv_filename = CSV file to read the comment data from
	# feature_extractor = function that converts a list of words into a list of word embeddings
	def __init__(self, csv_filename, feature_extractor, count=None, test=False, use_glove=False, character_level=False, verbose=False):
		self.test = test
		self.character_level = character_level
		start_time = int(round(time.time() * 1000)) 
		self.comments, self.vocab = self.load_data(csv_filename, count) 
		self.vocab = [] if self.test else DataSet.prune_vocabulary(self.vocab, use_glove)
		end_time = int(round(time.time() * 1000))

		self.feature_extractor = feature_extractor
		self.verbose = verbose
		self.x = None
		self.y = None
		
		if self.verbose:
			print('Loaded {0} comments from "{1}" in {2} seconds.'.format(
				len(self.comments),
				csv_filename,
				(end_time-start_time) / 1000.0
			))
			print('Vocabulary size = {0}'.format(len(self.vocab)))

	# Splits the input |text| into a list of words.
	# TODO: We may want to remove stop words and/or change this parsing in some way.
	@staticmethod
	def split_into_words(text):
		return re.findall(r"[\w'-]+|[.,!?;]", text)

	@staticmethod
	def get_glove_vocab():
		glove_vocab = set()
		with open('glove.twitter.27B/glove.twitter.27B.200d.txt') as f:
		 	for line in f:
		 		glove_vocab.add(line.split(' ')[0])
		return glove_vocab

	# Processes vocabulary by removing some subset of the words
	@staticmethod
	def prune_vocabulary(vocab, use_glove):
		if use_glove:
			# words in glove vocab and comment vocab
			glove_vocab = DataSet.get_glove_vocab().intersection(set(vocab.elements())) if use_glove else set()
		else:
			glove_vocab = set()

		# Only include words that occur >= MIN_WORD_COUNT times
		comment_vocab = set([word for word, count in vocab.items() if count >= DataSet.MIN_WORD_COUNT])
		
		vocab = sorted(list(comment_vocab.union(glove_vocab))) 
		vocab.append(DataSet.UNKNOWN_WORD)

		return { word:index for index, word in enumerate(vocab) }
		
	# Loads all of the comment data from the given |csv_filename|, only reads
	# the first |count| comments from the dataset (for debugging)
	def load_data(self, csv_filename, character_level, count=None):
		comments = []
		vocab = collections.Counter()
		print("Started loading data")
		with open(csv_filename) as csvfile:
			reader = csv.DictReader(csvfile)
			for i, row in enumerate(reader):
				if i == count: break

				words = DataSet.split_into_words(row['comment_text']) #list 
				labels = None if self.test else set([c for c in DataSet.CLASSES if row[c] == '1'])
		
				chars = None 
				if self.character_level: 
					txt = ''
					for word in words: 
						for c in word: 
							txt += c 
						txt += ' '
					chars = list(txt)

				comments.append(Comment(row['id'], words, labels, chars))
				
				if not self.test:
					if self.character_level:
						vocab.update([ch.lower() for ch in chars])
					else:
						vocab.update([word.lower() for word in words])
		print("Finished loading data")
		return comments, vocab

	# Converts a set of |labels| into the appropriate "one-hot" vector (i.e. there will
	# be ones in the indices corresponding to the input |labels|)
	@staticmethod
	def to_label_vector(labels):
		return np.array([ 1 if klass in labels else 0 for klass in DataSet.CLASSES ])

	# Returns the fully preprocessed input (x) and output (y):
	# (x) will be a list of numpy arrays with shape (comment length, embedding size)
	#     - Each element of x is a list of word embeddings for the words in the comment.
	# (y) will be a list of numpy arrays with shape (# of classes, )
	def get_data(self):
		if self.x is None:
			start_time = int(round(time.time() * 1000))
			self.x = [ self.feature_extractor.parse(c.words, self.vocab, self.character_level, c.chars) for c in self.comments ]
			if not self.test:
				self.y = [ DataSet.to_label_vector(c.labels) for c in self.comments ]
			end_time = int(round(time.time() * 1000))

			if self.verbose:
				print('Processing data (int get_data()) took {0} seconds.'.format((end_time-start_time) / 1000.0))

		return self.x, self.y

	# Takes in a |model| and evaluates its performance on this dataset
	# TODO: Implement This (probably want loss, accuracy, precision, recall, F1, etc.). 
	def evaluate_model(self, model):
		#y_hat = [ model.predict(x_value) for x_value in self.x ]
		pass

# Debugging / Testing code
if __name__ == "__main__":
	feature_extractor = OneHotFeatureExtractor(100) 
	data = DataSet(DataSet.TRAIN_CSV, feature_extractor, count=None, use_glove=False, character_level=True, verbose=True)
	x, y = data.get_data()

