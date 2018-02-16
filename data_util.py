#!/usr/bin/env python
# -*- coding: utf-8 -*-
import csv 

test_csv = "data/test.csv"
train_csv = "data/train.csv"

class Comment: 
	def __init__(self, example_id, comment_text, toxic, severe_toxic, obscene, threat, insult, identity_hate):
		self.example_id = example_id
		self.comment_text = comment_text
		self.toxic = toxic
		self.severe_toxic = severe_toxic
		self.obscene = obscene
		self.threat = threat
		self.insult = insult
		self.identity_hate = identity_hate

def process_csv(): 
	print "=== Processing data ==="
	data = {}
	with open(train_csv) as csvfile:
		reader = csv.DictReader(csvfile)
		for i, row in enumerate(reader):
			example_id, comment_text = row['id'], row['comment_text'] 
			toxic, severe_toxic, obscene = row['toxic'], row['severe_toxic'], row['obscene'] 
			threat, insult, identity_hate = row['threat'], row['insult'], row['identity_hate']
			data[example_id] = Comment(example_id, comment_text, toxic, severe_toxic, obscene, threat, insult, identity_hate)
	return data 