# twitter_gender
A simple experiment to enrich twitter profiles with gender

# How to run it?
1. Clone Repository

2. Install Requirements:
`pip install -r requirements.txt`

3. Run the file processor
`python file_processor.py`


—> It automatically downloads the dataset, builds the classifier, prints the test results and stores an enriched version of the dataset (csv file) under:
`twitter_gender/data/users_enriched.csv`

The last two columns are added: 
* ‘prediction’: the prediction result ‘female’ or ‘male'
* ‘source’: If the final prediction has happened based on the name (’name) or on description + last tweet (’text’)

# Can I classify unseen examples?

Yes.
```
from twitter_gender.file_processor import ProcessUsers

process_users = ProcessUsers()

labels = ['female', 'male']

gender = process_users.gender_classifier.get_gender_by_name('Johannes Erett')
print(labels[gender])

gender = process_users.gender_classifier.get_gender_by_text_custom('I am a woman of great faith... in unicorns ❤️ ')
print(labels[gender])
```

Tested with Python 3.7.1
