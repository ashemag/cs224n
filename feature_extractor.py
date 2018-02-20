#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
from dataset import *

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


class OneHotFeatureExtractor(FeatureExtractor):
	def __init__(self, comment_length):
		FeatureExtractor.__init__(self)
		self.comment_length = comment_length

	def parse(self, words, vocab):
		valid_words = [ word.lower() if word.lower() in vocab else DataSet.UNKNOWN_WORD for word in words ]
		word_features = [ vocab[word] for word in valid_words ]
		capital_features = [ self.capitalization_index(word) for word in words ]
		
		if len(word_features) < self.comment_length:
			padding_length = self.comment_length-len(word_features)
			word_features += [len(vocab)] * padding_length
			capital_features += [3] * padding_length

		return word_features[:self.comment_length], capital_features[:self.comment_length]


# Debugging / Testing code
if __name__ == "__main__":
	pass


