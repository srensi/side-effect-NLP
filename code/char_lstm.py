# Character LSTM with dropout for binary classification of tweets
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
max_length = 140		# Set max tweet length

# Replaces non-ascii characters with spaces.
def clean_strings(text):
	return re.sub(r'[^\x00-\x7F]+',' ', text.lower())

"""
Start of main program
"""
with open('/Users/srensi/Documents/GitHub/side-effect-NLP/code/binaryclassifier/binary_downloaded.tsv','r') as f:
	tweets = [ i.strip().split('\t')  for i in f.readlines() ]


# Randomize order of tweets in datasets
random.shuffle(tweets)

# Pull out tweets (X), and labels (y)
y , X = zip( *[ [i[-2], i[-1]] for i in tweets] )

# Clean up tweets
X = [clean_strings(x) for x in X]

# Map characters in tweets (X) to integers
chars = set()
[chars.update(x) for x in X]
encoder = dict( (c,i) for i, c in enumerate(sorted(chars), 1 ) )  
X = [[encoder[c] for c in x] for x in X]

# Convert labels (y) to numpy array
y = numpy.array([int(float(i)) for i in y])

# Get training set size and number of characters
n_train = int(numpy.floor(len(tweets)*0.8))
num_chars = len(encoder) + 1

# Separate ino training and test sets
y_train, X_train = y[:n_train], X[:n_train]
y_test, X_test = y[n_train:], X[n_train:] 

# Pad training examples for to make constant length
X_train = sequence.pad_sequences(X_train, maxlen=max_length, padding='post')
X_test = sequence.pad_sequences(X_test, maxlen=max_length, padding='post')

# Create the model
embedding_vecor_length = 32
model = Sequential()
model.add(Embedding(num_chars, embedding_vecor_length, input_length=max_length))
model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
model.fit(X_train, y_train, epochs=3, batch_size=64)

# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))