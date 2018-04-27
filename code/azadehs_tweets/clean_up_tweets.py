#!/usr/bin/python
import nltk
from nltk.corpus import stopwords


nltk.download('stopwords')
nltk.download('punkt')
stop_words = stopwords.words('english')

# Import tweets and annotaiton
with open('/Users/srensi/Documents/GitHub/side-effect-NLP/code/azadehs_tweets/train_tweets.tsv','r') as f:
	tweets = [ i.strip().split('\t')  for i in f.readlines()]

with open('/Users/srensi/Documents/GitHub/side-effect-NLP/code/azadehs_tweets/train_tweet_annotations.tsv') as f:
	annotations = [i.strip().split('\t') for i in f.readlines()]

# Match tweets to annotation
annotation_map = {i[2]: [i[3]] for i in tweets}
annotated_tweets = []
for i in annotations:
	try:
		annotated_tweets.append(i[1:] + annotation_map[ i[0] ])
	except KeyError:
		pass

# Cleans up tweets
for line in annotated_tweets[:15]:
	sentence = line[-1]
	sentence = nltk.tokenize.word_tokenize(sentence.replace('/', ' '))
	sentence = [word for word in sentence if word.isalpha()]
	sentence = [w.lower() for w in sentence if w not in stop_words]
	line[-1] = sentence
