# -*- coding: utf-8 -*-

import os
import csv
import datetime
import time
import sys

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
#	A class to measure performance of enrichment process using the metrics:
# 	precision, recall and f1 score based on given confusion matrix values.
#
##############################################################################

class PerformanceResults (object):

	def __init__(self, tp, fp, tn, fn):

		self.tp = tp		# True Positives
		self.fp = fp		# False Positives
		self.tn = tn		# True Negatives
		self.fn = fn		# False Negatives

		# Collect UTC datetime of measurement and performance results
		self.time = None
		self.prec = None
		self.rec = None
		self.f_one = None

		# Method to measure precision, recall and F1 score
		self.measurePerformance()


	# Customize the print format of instances of this class
	def __str__(self):

		# temp = u'\n' + u"==== Performance Results ====" + u'\n'
		# temp = str(self.time) + u"\n"
		temp =	u"\tPrecision: %.2f\n" % self.prec
		temp +=	u"\tRecall: %.2f\n" % self.rec
		temp +=	u"\tF1 Score: %.2f\n\n" % self.f_one

		temp +=	u"\tTrue Positives: " + str(self.tp) + u"\n"
		temp +=	u"\tFalse Positives: " + str(self.fp) + u"\n"
		temp +=	u"\tTrue Negatives: " + str(self.tn) + u"\n"
		temp +=	u"\tFalse Negatives: " + str(self.fn) + u"\n"

		return temp

	# Define how class instances are represented as a string
	def __repr__(self):
		return str(self)


	# Returns an iterable record of performance measurement results
	def record(self):

		return [self.time, self.prec, self.rec, self.f_one, self.tp, self.fp,
				self.tn, self.fn]


	# Calculate performance results based on confusion matrix
	def measurePerformance(self):

		# Get current UTC time to save with performance results
		self.time = str(datetime.datetime.utcnow()).split('.')[0]
		# Calculate precision
		if (self.tp + self.fp) != 0:
			self.prec = self.tp / (self.tp + self.fp)
		else:
			self.prec = 0
		# Calculate recall
		if (self.tp + self.fn) != 0:
			self.rec = self.tp / (self.tp + self.fn)
		else:
			self.rec = 0
		# Calculate F1 score
		if (self.prec + self.rec) != 0:
			self.f_one = 2 * self.prec * self.rec / (self.prec + self.rec)
		else:
			self.f_one = 0


def add_confusion_matrix(pred, truth, conf_matrix):

	if truth == 1:

		if pred == truth:
			conf_matrix[1]['tp'] += 1
			conf_matrix[2]['tn'] += 1

		elif pred == 2:
			conf_matrix[2]['fp'] += 1
			conf_matrix[1]['fn'] += 1

		else:
			conf_matrix[1]['fn'] += 1

	elif truth == 2:

		if pred == truth:
			conf_matrix[2]['tp'] += 1
			conf_matrix[1]['tn'] += 1

		elif pred == 1:
			conf_matrix[1]['fp'] += 1
			conf_matrix[2]['fn'] += 1

	else:
			conf_matrix[2]['tn'] += 1
			conf_matrix[1]['tn'] += 1


def process_ground_truth(classifier, name_only=False, text_only=False,
						 classifier_type='custom', language=False):

	if classifier_type not in {'wolohan', 'custom'}:
		raise ValueError('Text classifier type invalid')

	classes = [1, 2]
	conf_items = ['tp', 'fp', 'tn', 'fn']
	conf_matrix = {c: {item: 0 for item in conf_items} for c in classes}


	# out_file = open(get_data('users_sample.csv'), 'w')
	#
	# writer = csv.writer(out_file, quoting=csv.QUOTE_MINIMAL,
	# 						delimiter=',')


	with open(get_data('users_sample.csv'), 'r', newline='', errors='ignore') as f:

		reader = csv.reader(f, quoting=csv.QUOTE_MINIMAL, lineterminator='\n', delimiter=',')

		for row in reader:

			gender_reviewed = CLASS_MAP[row[-1]]

			if language and row[6] != language:
				continue

			gender = None

			if not text_only:
				name = row[3]
				gender = classifier.get_gender_by_name(name)

			if name_only:
				add_confusion_matrix(gender, gender_reviewed, conf_matrix)
				continue

			if not row[4] and not row[18]:
				continue

			if not gender:
				text = row[4] + ' ' + row[18]
				if classifier_type == 'custom':
					gender = classifier.get_gender_by_text_custom(text)
				else:
					gender = classifier.get_gender_by_text(text)

			add_confusion_matrix(gender, gender_reviewed, conf_matrix)

			# writer.writerow(row + [CLASS_LABELS[gender], CLASS_LABELS[gender_reviewed]])

	results = []
	for label in conf_matrix:
		result = PerformanceResults(conf_matrix[label]['tp'],
									 conf_matrix[label]['fp'],
									 conf_matrix[label]['tn'],
									 conf_matrix[label]['fn']
									 )
		results.append(result)
		print('Class: ', label)
		print(result)

	count = sum([result.tp + result.fn for result in results])
	prec = sum([result.prec * (result.tp + result.fn) for result in results]) / count
	rec = sum([result.rec * (result.tp + result.fn) for result in results]) / count
	f1 = sum([result.f_one * (result.tp + result.fn) for result in results]) / count

	print('Weighted Average:\nPrecision: %.2f\nRecall: %.2f\nF1: %.2f' % (prec, rec, f1))


# def rebuild_sample(classifier):
# 	out_file = open(get_data('users_sample.csv'), 'w')
#
# 	writer = csv.writer(out_file, quoting=csv.QUOTE_MINIMAL,
# 							delimiter=',')
#
# 	with open(get_data('users_sample.csv'), 'r', newline='', errors='ignore') as f:
#
# 		reader = csv.reader(f, quoting=csv.QUOTE_MINIMAL, lineterminator='\n', delimiter=',')
#
# 		for row in reader:
#
# 			gender_reviewed = int(row[0])
# 			row = row[1:]
#
# 			name = row[3]
# 			gender = classifier.get_gender_by_name(name)
#
# 			if not gender:
# 				text = row[4] + ' ' + row[18]
#
# 				gender = classifier.get_gender_by_text_custom(text)
#
# 			writer.writerow(row + [CLASS_LABELS[gender], CLASS_LABELS[gender_reviewed]])



def main():

	process_ground_truth(text_only=True)


# Make script accessible for Terminal commands
if __name__ == '__main__':

	sys.exit(main())
