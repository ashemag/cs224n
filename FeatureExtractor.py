#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np

from DataSet import *

class FeatureExtractor:
	N_CAPITALIZATION_FEATURES = 3

	def __init__(self):
		pass

	def capitalization_index(self, word):
		if word.islower(): return 0 # |word| is all lower case
		if word.isupper(): return 2 # |word| is all upper case
		return 1 # |word| is mixed case

	def parse(words, vocab):
		raise NotImplementedError("Subclasses of FeatureExtractor must implement parse().")


# Output feature vectors are one-hot word vectors concatenated with capitalization features.
class OneHotFeatureExtractor(FeatureExtractor):
	def __init__(self):
		FeatureExtractor.__init__(self)

	def parse(self, words, vocab):
		capitalization_indices = [ self.capitalization_index(word) for word in words ]
		one_hot_capitalization = np.eye(FeatureExtractor.N_CAPITALIZATION_FEATURES)[capitalization_indices]

		valid_words = [ word.lower() if word.lower() in vocab else Dataset.UNKNOWN_WORD for word in words ]
		word_indices = [ vocab.index(word) for word in valid_words ]		
		one_hot_word_vectors = np.eye(len(vocab))[word_indices]

		return np.concatenate((one_hot_word_vectors, one_hot_capitalization), axis=1)



# Debugging / Testing code
if __name__ == "__main__":
	pass


