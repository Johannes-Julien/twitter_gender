# -*- coding: utf-8 -*-

#############################################################################
#
#	Required imports.
#
#############################################################################

import re
from gender_guesser import detector
from twitter_gender.gender_lib.SapGenderPrediction import GndrPrdct

#############################################################################
#
#	A class to classify gender based on name.
#
#############################################################################

class GenderClassifier (object):

	def __init__(self):

		self.name_classifier = detector.Detector()
		self.text_classifier = GndrPrdct()
		self.custom_text_classifier = None


	def get_gender_by_name(self,name):

		# Pass in the first name, assuming format "first last"
		# Make sure the name is in title case.
		name = name.title()
		for name_component in re.split(r'[.,;_\- ]', name):
			gndr = self.name_classifier.get_gender(name_component)
			if gndr == 'female' or gndr == 'mostly_female':
				return 1
			elif gndr == 'male' or gndr == 'mostly_male':
				return 2
		return 0


	def get_gender_by_text(self, text):

		gndr = self.text_classifier.predict_gender(text)

		if gndr == 0:
			return 2
		elif gndr == 1:
			return 1


	def get_gender_by_text_custom(self, text):

		if self.custom_text_classifier:
			return self.custom_text_classifier.classify(text)[0]
