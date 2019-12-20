# -*- coding: utf-8 -*-

#############################################################################
#
#	Required imports.
#
#############################################################################

import os
import requests
import csv
import time

from twitter_gender.gender_classifier import GenderClassifier
from twitter_gender.performance import process_ground_truth
from twitter_gender.profile2vec import CustomGenderClassifier


#############################################################################
#
#	A necessary utility for accessing the data local to the installation.
#
#############################################################################

_ROOT = os.path.abspath(os.path.dirname(__file__))
def get_data(path):
    return os.path.join(_ROOT, 'data', path)


CLASS_LABELS = ['unknown', 'female', 'male', 'ambiguous', 'undefinable',
				'female_hidden', 'male_hidden']
CLASS_MAP = {label: index for index, label in enumerate(CLASS_LABELS)}

#############################################################################
#
#	A class to process the 2013 "Tweets Loud and Quiet" twitter user dataset.
#	Processes user records and assigns gender.
#
#############################################################################

class ProcessUsers(object):

	def __init__(self, dataset_url=
	"https://github.com/jonbruner/twitter-analysis/raw/master/users.csv", ):
		file_name="users.csv"

		self.dataset_url=dataset_url
		self.file_name = file_name

		self.gender_classifier = GenderClassifier()

		self.id_to_index = {}

		self.process()

		self.demo()


	def demo(self):

		labels = ['unknown', 'female', 'male']
		print("Demo: \n")

		for text in ['I am a man of great faith... in technology',
					 'I like cars',
					 'Why is the earth round?',
					 'I am a woman of great faith... in unicorns',
					 'I like fashion',
					 'Why does astrology work for my friends?']:
			print('Input: ', text)
			print('Prediction: ', labels[self.gender_classifier.get_gender_by_text_custom(text)])


	def download_dataset(self):

		print('Downloading dataset')
		r = requests.get(self.dataset_url)

		with open(get_data(self.file_name), 'wb') as f:
			f.write(r.content)

		# Retrieve HTTP meta-data
		print('Download completed with status %s, content type: %s and encoding: %s'
			  % (r.status_code, r.headers['content-type'], r.encoding))


	def process_dataset(self):

		"""Load twitter data
		This function loads the dataset and the truth and returns:
		Merged description and tweet of the authors, the truth, Author IDs, and the original length of the text.

		Returns:
			merged_tweets_of_authors: List. Each item is all of the tweets of an author, merged into one string.
				Refer to the list of replacements in the remarks.
			truths: List of truths for authors.
			author_ids: List of Author IDs.
			original_tweet_lengths: List of original tweet lengths.
		Remarks:
			- List of replacements:
				Line feed		<LineFeed>
				End of Tweet	<EndOfTweet>
		"""

		''' 
		*os.listdir* returns a list containing the name of all files and folders in the given directory.
		Normally, the list is created in ascending order. However, the Python documentation states,
		“the list is in arbitrary order”.
		To ensure consistency and avoid errors in syncing the order of the items among
		different lists (e.g., *author_ids*, *truths*), we sort the list by calling *sorted*.
		*sorted()* returns a new sorted list (in ascending lexicographical order) of all the items in an iterable.
		'''
		# Store the Author IDs in a list
		# The Author IDs list will have the same order as the XML filenames list.
		author_ids = []  # Create an empty list

		# Initialize the lists.
		# The lists will have the same order as the XML filenames list (refer to: “Iterate over XML Files”)
		original_text_lengths = []  # Create an empty list
		# ↳ Every row will represent an author, every column will represent a tweet.
		merged_texts_of_authors = []  # Create an empty list
		# ↳ Each cell will contain all 100 tweets of an author, merged.
		truths = []

		print('Processing dataset')

		out_file = open(get_data(self.file_name[:-4] + '_clean.csv'), 'w')

		writer = csv.writer(out_file, quoting=csv.QUOTE_MINIMAL,
								delimiter=',')

		counter = 0
		author_index = 0
		class_counter = [0,0,0]
		with open(get_data(self.file_name), 'r', newline='', errors='ignore') as f:

			reader = csv.reader(f, quoting=csv.QUOTE_MINIMAL, lineterminator='\n', delimiter=',')

			# Iterate over profiles
			for row in reader:

				counter += 1
				if counter % 50000 == 0:
					print(counter, ' lines processed')
					# break

				# Replace line feeds and carriage returns (\n, \r, \r\n) with “ <LineFeed> ”
				for i in [4, 18]:
					row[i] = row[i].replace('\n', " <LineFeed> ").replace('\r', " <LineFeed> ")

				if row[2]:
					writer.writerow(row)

					if row[6] == 'en' and row[4] and row[18]:
						name = row[3]
						gender = self.gender_classifier.get_gender_by_name(name)
						if gender:
							class_counter[gender] += 1
							truths.append(gender)
							author_ids.append(author_index)

							# Record the text lengths. Each row represents an author.
							original_text_lengths.append([len(row[4]), len(row[18])])

							# Process description and last tweet
							# , replace line feeds, and append the tweet to a list
							# Concatenate the tweets of this author, and append it to the main list
							merged_texts_of_this_author = row[4] + " <EndOfDescription> " + row[18] + " <EndOfTweet>"
							# ↳ " <EndOfTweet> ".join adds the tag between every two strings, so we need to add another tag to the end.
							merged_texts_of_authors.append(merged_texts_of_this_author)

							author_index += 1

			print("%.2f seconds: Finished loading the dataset" % time.process_time())

			print("CLASSES: ", class_counter)
			return merged_texts_of_authors, truths, author_ids, original_text_lengths


	def enrich_dataset(self):

		print('Enriching dataset')

		out_file = open(get_data(self.file_name[:-4] + '_enriched.csv'), 'w')

		writer = csv.writer(out_file, quoting=csv.QUOTE_MINIMAL,
								delimiter=',')

		counter = 0
		with open(get_data(self.file_name[:-4] + '_clean.csv'), 'r', newline='', errors='ignore') as f:

			reader = csv.reader(f, quoting=csv.QUOTE_MINIMAL, lineterminator='\n', delimiter=',')

			# Iterate over profiles
			for row in reader:
				counter += 1
				if counter == 1:
					writer.writerow(row + ['prediction', 'source'])
					continue

				source = 'not found'

				counter += 1
				if counter % 50000 == 0:
					print(counter, ' lines processed')
					# break

				name = row[3]
				gender = self.gender_classifier.get_gender_by_name(name)
				if gender:
					source = 'name'

				elif row[4] or row[18]:
					text = row[4] + " <EndOfDescription> " + row[18] + " <EndOfTweet>"
					gender = self.gender_classifier.get_gender_by_text_custom(text)
					if gender:
						source = 'text'

				new_row = row + [CLASS_LABELS[gender], source]
				writer.writerow(new_row)

			print("%.2f seconds: Finished enriching the dataset" % time.process_time())


	def process(self):

		if self.file_name not in os.listdir(get_data('')):

			self.download_dataset()

		merged_texts_of_authors, truths, author_ids, original_text_lengths = self.process_dataset()

		self.gender_classifier.custom_text_classifier = \
			CustomGenderClassifier(merged_texts_of_authors, truths, author_ids, original_text_lengths)

		print('Performance with standard classifier:')
		process_ground_truth(self.gender_classifier, classifier_type='wolohan')
		print('Performance with CUSTOM classifier:')
		process_ground_truth(self.gender_classifier, classifier_type='custom')
		print('CUSTOM classifier for english profiles only:')
		process_ground_truth(self.gender_classifier, classifier_type='custom', language='en')

		self.enrich_dataset()


# Make script accessible for Terminal commands
if __name__ == '__main__':

	ProcessUsers()
