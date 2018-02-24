class DataUtil:
	CLASSES = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
	TRAIN_CSV = "data/train.csv"
	TEST_CSV = "data/test.csv"
	UNKNOWN_WORD = "<unknown_word>"

	# csv_filename = CSV file to read the comment data from
	# feature_extractor = function that converts a list of words into a list of word embeddings
	def __init__(self, is_training, feature_extractor, count=None, verbose=False):
		self.is_training = is_training

		csv_filename = self.TRAIN_CSV if is_training else self.TEST_CSV
		start_time = int(round(time.time() * 1000)) 
		self.comments, self.vocab = self.load_data(csv_filename, count) 
		end_time = int(round(time.time() * 1000))

		self.feature_extractor = feature_extractor
		self.verbose = verbose
		self.x = None
		self.y = None
		
		#train data comments 
		if self.verbose:
			print 'Loaded {0} comments from "{1}" in {2} seconds.'.format(
				len(self.comments),
				csv_filename,
				(end_time-start_time) / 1000.0
			)
			print 'Vocabulary size = {0}'.format(len(self.vocab))
			print "Most common vocabulary words = {0}".format(self.vocab.most_common(5))

	# Splits the input |text| into a list of words.
	# TODO: We may want to remove stop words and/or change this parsing in some way.
	@staticmethod
	def split_into_words(text):
		return re.findall(r"[\w'-]+|[.,!?;]", text)

	# Processes vocabulary by removing some subset of the words
	# TODO: We need to decide on how to go about this. 
	@staticmethod
	def prune_vocabulary(vocab):
		new_vocab = sorted(list(set(vocab.elements())))
		new_vocab.append(DataSet.UNKNOWN_WORD)
		return { word:index for index, word in enumerate(new_vocab) }

	# Loads all of the comment data from the given |csv_filename|, only reads
	# the first |count| comments from the dataset (for debugging)
	def load_data(self, csv_filename, count=None, train=True):
		comments = []
		vocab = collections.Counter()

		with open(csv_filename) as csvfile:
			reader = csv.DictReader(csvfile)
			for i, row in enumerate(reader):
				if i == count: break
				words = DataSet.split_into_words(row['comment_text']) #list 
				labels = []
				if self.is_training:
					labels = set([c for c in DataUtil.CLASSES if row[c] == '1'])
				comments.append(Comment(row['id'], words, labels))
				vocab.update([word.lower() for word in words])
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
		if self.x is None or self.y is None:
			start_time = int(round(time.time() * 1000))
			self.x = [ self.feature_extractor.parse(comment.words, self.vocab) for comment in self.comments ]
			self.y = [ DataUtil.to_label_vector(comment.labels) for comment in self.comments ]		
			ids = [comment.example_id for comment in self.comments]

			end_time = int(round(time.time() * 1000))

			if self.verbose:
				print 'Processing data (int get_data()) took {0} seconds.'.format((end_time-start_time) / 1000.0)

		return self.x, self.y, ids

	# Takes in a |model| and evaluates its performance on this dataset
	# TODO: Implement This (probably want loss, accuracy, precision, recall, F1, etc.). 
	def evaluate_model(self, model):
		#y_hat = [ model.predict(x_value) for x_value in self.x ]
		pass


#driver 
if __name__ == "__main__": 
	#data = load_data() 
	vocab, n = prune_vocabulary('corpus.txt')


