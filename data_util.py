#!/usr/bin/env python
# -*- coding: utf-8 -*-
import csv 
import string 
import collections 

test_csv = "data/test.csv"
train_csv = "data/train.csv"
WRITE_CORPUS = False 

class Comment: 
	def __init__(self, example_id, comment_text, label_vector, embedding =''):
		self.example_id = example_id
		self.comment_text = comment_text
		self.label_vector = label_vector
		self.embedding = embedding

def load_data(): 
	print "Loading training data..."
	valid_characters = string.printable[0:62]
	data = {}
	corpus = ''
	with open(train_csv) as csvfile:
		reader = csv.DictReader(csvfile)
		for i, row in enumerate(reader):
			example_id, comment_text = row['id'], row['comment_text'] 
			toxic, severe_toxic, obscene = int(row['toxic']), int(row['severe_toxic']), int(row['obscene'])
			threat, insult, identity_hate = int(row['threat']), int(row['insult']), int(row['identity_hate']) 
			data[example_id] = Comment(example_id, comment_text, [toxic, severe_toxic, obscene, threat, insult, identity_hate])
			if WRITE_CORPUS:
				for ch in comment_text: 
					if ch not in valid_characters: 
						comment_text = comment_text.replace(ch, ' ')

				corpus += ' ' + comment_text

	if WRITE_CORPUS: 
		with open("corpus.txt", "w") as f: 
			f.write(corpus)

	print "Done. Read %d comments." % i 
	return data  

#map word comments to word vectors (GloVe)
# for each comment example, for each word -> vector 
#initialize with glove vectors 

#Helper: Get vocabulary from comments and remove terms that appear less than X times 
# Maybe not a good idea for comments, since many words are WOOOOOOO
def prune_vocabulary(filename, threshold = 0): 
	print "Pruning vocabulary..."
	with open(filename, "r") as f: 
		corpus = f.read().lower() 
	ctr = collections.Counter([word for word in corpus.split(' ')])
	ctr[' '] = 0
	n = len(ctr)
	vocab = [v for v in ctr if ctr[v] > threshold]
	print n 
	print "We lose %f percent of vocabulary" % (100 - ((len(vocab) / float(n)) * 100)) 
	return vocab, n 

def embeddings_layer(data): 
	pass 

#driver 
if __name__ == "__main__": 
	#data = load_data() 
	vocab, n = prune_vocabulary('corpus.txt')


