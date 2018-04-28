"""
Character LSTM with dropout for binary classification of tweets. Current settings should get 
around 0.90 accuracy on dataset.
"""

import numpy
import random
import re
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence


"""
Global Vars and Functions
"""
numpy.random.seed(7)	# set random seed for reproducibility
max_tweet_length = 140		# Set max tweet length

# Coverts to lowercase and replaces non-ascii characteracters with spaces.
def clean_string(text):
	return re.sub(r'[^\x00-\x7F]+',' ', text.lower())

"""
Import, clean, and encode the twitter data for the LSTM
"""
# Import data
with open('./binaryclassifier/binary_downloaded.tsv','r') as f:
	data = [ i.strip().split('\t')  for i in f.readlines() ]


# Shuffle the rows of the data for cross validation
random.shuffle(data)

# Pull out Tweets and their labels
Labels , Tweets = zip( *[ [label, tweet] for tweet_id, user_id, label, tweet in data] )

# Clean up tweets
Tweets = [clean_string(tweet) for tweet in Tweets]

# Get dict to encode characters as integers and total number of characters
alphabet = set()
[alhpabet.update(tweet) for tweet in Tweets]
encoder = dict( (character, integer) for integer, character in enumerate(sorted(alphabet), 1 ) )  
alphabet_size = len(encoder) + 1


# Encode tweets and labels as numeric types for LSTM
Tweets = [[encoder[character] for character in tweet] for tweet in Tweets]
Labels = numpy.array([ int( float( label ) ) for label in Labels])

"""
Train and test the LSTM
"""
# Get index for 80/20 split of data into training and test
split = numpy.floor( len(Tweets)*0.8 )
split = int( split )

# Split into training and test sets
trainTweets, testTweets = Tweets[:split], Tweets[split:]
trainLabels, testLabels = Labels[:split], Labels[split:]

# Pad training examples for to make constant length
trainTweets = sequence.pad_sequences(trainTweets, maxlen=max_tweet_length, padding='post')
testTweets = sequence.pad_sequences(testTweets, maxlen=max_tweet_length, padding='post')

# Create the model
embedding_vecor_length = 32
model = Sequential()
model.add(Embedding(alphabet_size, embedding_vecor_length, input_length=max_tweet_length))
model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
model.fit(trainTweets, trainLabels, epochs=3, batch_size=64)

# Final evaluation of the model
scores = model.evaluate(testTweets, testLabels, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))
