#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np

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
	def __init__(self, comment_length, vocab=None):
		FeatureExtractor.__init__(self)
		self.comment_length = comment_length
		self.vocab = vocab
	
	

	# called for each comment 
	def parse(self, words, vocab, character_level, chars):
		if character_level: 
			max_len = self.comment_length * 5 
			capital_features = [self.capitalization_index(char) for char in chars]  
			word_features = [i for i, char in enumerate(chars)] #position in comment 
			if len(chars) < max_len: 
				padding_length = max_len - len(chars)
				word_features += [len(vocab)] * padding_length #arbitrary 
				capital_features += [3] * padding_length  
				
			word_features, capital_features = word_features[:max_len], capital_features[:max_len]
			return word_features, capital_features
		else: 
			if len(vocab) == 0: vocab = self.vocab

			valid_words = [ word.lower() if word.lower() in vocab else '<unknown>' for word in words ]
			word_features = [ vocab[word] for word in valid_words ] # index of each word in comment in the vocab.
			capital_features = [ self.capitalization_index(word) for word in words ]

			#padding 
			if len(word_features) < self.comment_length:
				padding_length = self.comment_length-len(word_features)
				word_features += [len(vocab)] * padding_length
				capital_features += [3] * padding_length  
			
			#truncates when > comment length 
			return word_features[:self.comment_length], capital_features[:self.comment_length] 

# Debugging / Testing code
if __name__ == "__main__":
	pass


