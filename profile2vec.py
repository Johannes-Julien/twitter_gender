# -*- coding: utf-8 -*-

#############################################################################
#
#	Required imports.
#
#############################################################################

import time

from nltk import TweetTokenizer
from nltk.tokenize.treebank import TreebankWordDetokenizer
import emoji

import numpy as np

from sklearn import metrics
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.svm import LinearSVC


class CustomGenderClassifier(object):

	def __init__(self, merged_tweets_of_authors, truths, author_ids, original_tweet_lengths, reduce_dims=False):

		self.reduce_dims = reduce_dims
		self.ngrams_vectorizer = None
		self.svd_reducer = None
		'''
		Build a classifier: Linear Support Vector classification
		- The underlying C implementation of LinearSVC uses a random number generator to select features when fitting the
		model.
		References:
		http://scikit-learn.org/stable/modules/svm.html#tips-on-practical-use
		http://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html
		'''
		self.classifier = LinearSVC(random_state=42)

		self.build_classifier(merged_tweets_of_authors, truths, author_ids, original_tweet_lengths)


	def classify(self, text):

		text_prep = self.preprocess_tweet(text)

		vecs = self.ngrams_vectorizer.transform([text_prep])
		if self.reduce_dims:
			vecs = self.svd_reducer.transform(vecs)

		redictions = self.classifier.predict(vecs)

		return redictions


	def preprocess_tweet(self, tweet):
		"""Pre-process a tweet and/or profile description.
		The following pre-processing operations are done on the text:
		- Replace emojis like: "Python is :thumbs_up:"
		- Replace repeated character sequences of length 3 or greater with sequences of length 3
		- Lowercase
		- Replace all URLs and username mentions with the following tags:
			URL		    <URLURL>
			@Username   <UsernameMention>
		Args:
			tweet: String
		Returns:
			The pre-processed tweet as String
		IMPROVEMENTS TO MAKE:
		- Instead of tokenizing and detokenizing, which is messy, the strings should be directly replaced using regex.
		"""

		replaced_urls = []  # Create an empty list
		replaced_mentions = []  # Create an empty list

		# Replace emojis
		tweet = emoji.demojize(tweet)

		# Tokenize using NLTK
		tokenizer = TweetTokenizer(preserve_case=False, reduce_len=True)
		tokens = tokenizer.tokenize(tweet)

		# Iterate over tokens
		for index, token in enumerate(tokens):
			# Replace URLs
			if token[0:4] == "http":
				replaced_urls.append(token)
				tokens[index] = "<URLURL>"
				# ↳ *tokens[index]* will directly modify *tokens*, whereas any changes to *token* will be lost.

			# Replace mentions (Twitter handles; usernames)
			elif token[0] == "@" and len(token) > 1:
				# ↳ Skip the single '@' tokens
				replaced_mentions.append(token)
				tokens[index] = "<UsernameMention>"

		# Detokenize using NLTK's Treebank Word Detokenizer
		detokenizer = TreebankWordDetokenizer()
		processed_tweet = detokenizer.detokenize(tokens)

		# *replaced_urls* and *replaced_mentions* will contain all of the replaced URLs and Mentions of the input string.
		return processed_tweet


	def extract_features(self, docs_train, docs_test, word_ngram_range=(1, 3), dim_reduce=False):
		"""Extract features
		This function builds a transformer (vectorizer) pipeline,
		fits the transformer to the training set (learns vocabulary and idf),
		transforms the training set and the test set to their TF-IDF matrix representation,
		and builds a classifier.
		"""

		# Build a vectorizer that splits strings into sequences of i to j words
		word_vectorizer = TfidfVectorizer(preprocessor=self.preprocess_tweet,
									  analyzer='word', ngram_range=word_ngram_range,
									  min_df=2, use_idf=True, sublinear_tf=True)
		# Build a vectorizer that splits strings into sequences of 3 to 5 characters
		char_vectorizer = TfidfVectorizer(preprocessor=self.preprocess_tweet,
									 analyzer='char', ngram_range=(3, 5),
									 min_df=2, use_idf=True, sublinear_tf=True)

		# Build a transformer (vectorizer) pipeline using the previous analyzers
		# *FeatureUnion* concatenates results of multiple transformer objects
		self.ngrams_vectorizer = Pipeline([('feats', FeatureUnion([('word_ngram', word_vectorizer),
														 ('char_ngram', char_vectorizer),
														 ])),
								 # ('clff', LinearSVC(random_state=42))
								 ])

		# Fit (learn vocabulary and IDF) and transform (transform documents to the TF-IDF matrix) the training set
		X_train_ngrams_tfidf = self.ngrams_vectorizer.fit_transform(docs_train)
		'''
		↳ Check the following attributes of each of the transformers (analyzers)—*word_vectorizer* and *char_vectorizer*:
		vocabulary_ : dict. A mapping of terms to feature indices.
		stop_words_ : set. Terms that were ignored
		'''
		print("%.2f seconds: Finished fit_transforming the training dataset" % time.process_time())
		print("Training set word & character ngrams .shape = ", X_train_ngrams_tfidf.shape)

		feature_names_ngrams = [word_vectorizer.vocabulary_, char_vectorizer.vocabulary_]

		'''
		Extract the features of the test set (transform test documents to the TF-IDF matrix)
		Only transform is called on the transformer (vectorizer), because it has already been fit to the training set.
		'''
		X_test_ngrams_tfidf = self.ngrams_vectorizer.transform(docs_test)
		print("%.2f seconds: Finished transforming the test dataset" % time.process_time())
		print("Test set word & character ngrams .shape = ", X_test_ngrams_tfidf.shape)

		# • Dimensionality reduction using truncated SVD (aka LSA)
		if dim_reduce:
			# Build a truncated SVD (LSA) transformer object
			self.svd_reducer = TruncatedSVD(n_components=300, random_state=43)
			# Fit the LSI model and perform dimensionality reduction
			X_train_ngrams_tfidf_reduced = self.svd_reducer.fit_transform(X_train_ngrams_tfidf)
			print("@ %.2f seconds: Finished dimensionality reduction (LSA) on the training dataset", time.process_time())
			X_test_ngrams_tfidf_reduced = self.svd_reducer.transform(X_test_ngrams_tfidf)
			print("@ %.2f seconds: Finished dimensionality reduction (LSA) on the test dataset", time.process_time())

			X_train = X_train_ngrams_tfidf_reduced
			X_test = X_test_ngrams_tfidf_reduced
		else:
			X_train = X_train_ngrams_tfidf
			X_test = X_test_ngrams_tfidf

		return X_train, X_test, feature_names_ngrams


	def cross_validate_model(self, X_train, y_train):
		"""Evaluates the classification model by k-fold cross-validation.
		The model is trained and tested k times, and all the scores are reported.
		"""

		# Build a stratified k-fold cross-validator object
		skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

		'''
		Evaluate the score by cross-validation
		This fits the classification model on the training data, according to the cross-validator
		and reports the scores.
		Alternative: sklearn.model_selection.cross_validate
		'''
		scores = cross_val_score(self.classifier, X_train, y_train, scoring='accuracy', cv=skf)

		print("%.2f seconds: Cross-validation finished" % time.process_time())

		# Log the cross-validation scores, the mean score and the 95% confidence interval, according to:
		# http://scikit-learn.org/stable/modules/cross_validation.html#computing-cross-validated-metrics
		# https://en.wikipedia.org/wiki/Standard_error#Assumptions_and_usage
		# print("Scores = %s" % scores)
		# print("Accuracy: %0.2f (±%0.2f)" % (scores.mean()*100, scores.std()*2*100))
		# ↳ https://docs.scipy.org/doc/numpy/reference/generated/numpy.std.html


	def train_and_test_model(self, X_train, y_train, X_test, y_test):
		"""Train the classifier and test it.
		This function trains the classifier on the training set,
		predicts the classes on the test set using the trained model,
		and evaluates the accuracy of the model by comparing it to the truth of the test set.
		"""

		# Fit the classification model on the whole training set (as opposed to cross-validation)
		# print("Y TRAIN: ", y_train[:10])
		# print("x TRAIN: ", X_train[:10])
		self.classifier.fit(X_train, y_train)
		y_train_predicted = self.classifier.predict(X_train)
		print("np.mean Accuracy TRAINING: %s" % np.mean(y_train_predicted == y_train))

		''' Predict the outcome on the test set
			Note that the clf classifier has already been fit on the training data.
		'''
		y_predicted = self.classifier.predict(X_test)

		print("%.2f seconds: Finished training the model and predicting class labels for the test set" % time.process_time())

		# Simple evaluation using numpy.mean
		# print("np.mean Accuracy: %s" % np.mean(y_predicted == y_test))

		# Log the classification report
		# print("Classification report:\n%s" % metrics.classification_report(y_test, y_predicted))

		# The confusion matrix
		# confusion_matrix = metrics.confusion_matrix(y_test, y_predicted)
		# print("Confusion matrix:\n%s" % confusion_matrix)


	def get_train_test(self, merged_tweets_of_authors, truths, author_ids, original_tweet_lengths):
		"""Load the twitter  dataset for the development phase.
		This function loads the twitter training documents and truth,
		then splits the dataset into training and test sets.
		"""

		# Split the dataset into balanced (stratified) training and test sets:
		docs_train, docs_test, y_train, y_test, author_ids_train, author_ids_test,\
		original_tweet_lengths_train, original_tweet_lengths_test =\
			train_test_split(merged_tweets_of_authors, truths, author_ids, original_tweet_lengths,
							 test_size=0.4, random_state=42, stratify=truths)
		# ↳ *stratify=truths* selects a balanced sample from the data, with the same class proportion as the *truths* list.

		# • Sort all lists in the ascending order of *author_ids* (separately, for the training and test set)
		# This is only done for the sakes of consistency between the *load_datasets_development()* and
		# *load_datasets_tira_evaluation()* functions, because the output of the latter is sorted by *author_ids*, while the
		# former is shuffled by the *train_test_split()* function.
		# Sort the training set
		author_ids_train, docs_train, y_train, original_tweet_lengths_train = [list(tuple) for tuple in zip(*sorted(zip(
			author_ids_train, docs_train, y_train, original_tweet_lengths_train)))]
		# Sort the test set
		author_ids_test, docs_test, y_test, original_tweet_lengths_test = [list(tuple) for tuple in zip(*sorted(zip(
			author_ids_test, docs_test, y_test, original_tweet_lengths_test)))]

		return docs_train, docs_test, y_train, y_test

	def build_classifier(self, merged_tweets_of_authors, truths, author_ids, original_tweet_lengths):

		"""The "main" function for the development phase.
		Every time the script runs, it will call this function.
		"""

		print("Building custom classifier")

		docs_train, docs_test, y_train, y_test = self.get_train_test(merged_tweets_of_authors, truths, author_ids, original_tweet_lengths)
		X_train, X_test, feature_names = self.extract_features(docs_train, docs_test, dim_reduce=False)
		self.cross_validate_model(X_train, y_train)
		self.train_and_test_model(X_train, y_train, X_test, y_test)

		# Log run time
		print("%.2f seconds: Run finished\n" % time.process_time())
