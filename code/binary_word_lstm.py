# LSTM for sequence classification
import numpy
import random
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
import nltk
from nltk.corpus import stopwords



nltk.download('stopwords')
nltk.download('punkt')
stop_words = stopwords.words('english')

# fix random seed for reproducibility
numpy.random.seed(7)
# load the dataset but only keep the top n words, zero the rest
max_tweet_words = 50

# clean up data
def clean_sentence(text):
	# print(text)
	sentence = nltk.tokenize.word_tokenize(text.replace('/', ' '))
	sentence = [word for word in sentence if word.isalpha()]
	sentence = [w.lower() for w in sentence if w not in stop_words]
	return sentence


# Import data
with open('./binaryclassifier/binary_downloaded.tsv','r') as f:
	data = [ i.decode('utf-8').strip().split('\t')  for i in f.readlines() ]


# Shuffle the rows of the data for cross validation
random.shuffle(data)

# Pull out Tweets and their labels
Labels , Tweets = zip( *[ [label, tweet] for tweet_id, user_id, label, tweet in data] )

# Clean up tweets
Tweets = [clean_sentence(tweet) for tweet in Tweets]

# Get dict to encode characters as integers and total number of characters
vocab = set(['<unk>'])
[vocab.update(tweet) for tweet in Tweets]
encoder = dict( (word, integer) for integer, word in enumerate(sorted(vocab), 1 ) )  
vocab_size = len(encoder) + 1

# Encode tweets and labels as numeric types for LSTM
Tweets = [[encoder[word] for word in tweet] for tweet in Tweets]
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
trainTweets = sequence.pad_sequences(trainTweets, maxlen=max_tweet_words, padding='post')
testTweets = sequence.pad_sequences(testTweets, maxlen=max_tweet_words, padding='post')

# Create the model
embedding_vecor_length = 32
model = Sequential()
model.add(Embedding(vocab_size, embedding_vecor_length, input_length=max_tweet_words))
model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
model.fit(trainTweets, trainLabels, epochs=3, batch_size=64)

# Final evaluation of the model
scores = model.evaluate(testTweets, testLabels, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))
