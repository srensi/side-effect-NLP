# characteracter LSTM with dropout for binary classification of tweets
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
def clean_strings(text):
	return re.sub(r'[^\x00-\x7F]+',' ', text.lower())

"""
Start of main program
"""

with open('./binaryclassifier/binary_downloaded.tsv','r') as f:
	data = [ i.strip().split('\t')  for i in f.readlines() ]

## This fisrt part cleans the data, and gets it ready for lstm

# Shuffle the rows of the data (for cross validation) pull 
random.shuffle(data)

# Pull out Tweets and their labels
Labels , Tweets = zip( *[ [label, tweet] for tweet_id, user_id, label, tweet in data] )

# Clean up tweets
Tweets = [clean_strings(tweet) for tweet in Tweets]

# Get total number of characters and dict to encode characters as integers
characters = set()
[characters.update(tweet) for tweet in Tweets]
encoder = dict( (character,idx) for idx, character in enumerate(sorted(characters), 1 ) )  
num_characters = len(encoder) + 1


# Encode tweets and labels as numeric numpy arrays for LSTM
Tweets = [[encoder[character] for character in tweet] for tweet in Tweets]
Tweets = sequence.pad_sequences(Tweets, maxlen=max_tweet_length, padding='post')
Labels = numpy.array([ int( float( label ) ) for label in Labels])


## This part trains and evaluates the lstm

# Get index for 80/20 data split
n_train = numpy.floor( len(Tweets)*0.8 )
split = int( n_train )

# Split into training and test sets
trainTweets, testTweets = Tweets[:split], Tweets[:-split]
trainLabels, testLabels = Labels[:split], Labels[:-split]

# Pad training examples for to make constant length
trainTweets = sequence.pad_sequences(trainTweets, maxlen=max_tweet_length, padding='post')
testTweets = sequence.pad_sequences(testTweets, maxlen=max_tweet_length, padding='post')

# Create the model
embedding_vecor_length = 32
model = Sequential()
model.add(Embedding(num_characters, embedding_vecor_length, input_length=max_tweet_length))
model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
model.fit(X_train, y_train, epochs=3, batch_size=64)

# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))