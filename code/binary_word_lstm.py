# LSTM for sequence classification
import numpy as np
import random
import re
import os
from keras import regularizers
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Bidirectional
from keras.layers import BatchNormalization
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras import optimizers
from sklearn.metrics import confusion_matrix
# from keras.preprocessing import text
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences



# nltk.download('stopwords')
# nltk.download('punkt')
# stop_words = stopwords.words('english')

# fix random seed for reproducibility
# np.random.seed(7)
# load the dataset but only keep the top n words, zero the rest
max_tweet_words = 50


BASE_DIR = '/Users/srensi/Documents/GitHub/side-effect-NLP/data/'
GLOVE_DIR = os.path.join(BASE_DIR, 'glove.twitter.27B')
# TEXT_DATA_DIR = os.path.join(BASE_DIR, '20_newsgroup')
MAX_SEQUENCE_LENGTH = 100
MAX_NUM_WORDS = 20000
EMBEDDING_DIM = 100
VALIDATION_SPLIT = 0.2

# first, build index mapping words in the embeddings set
# to their embedding vector

print('Indexing word vectors.')

embeddings_index = {}
with open(os.path.join(GLOVE_DIR, 'glove.twitter.27B.100d.txt')) as f:
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs


# clean up data
# def clean_sentence(text):
# 	# print(text)
# 	sentence = nltk.tokenize.word_tokenize(text.replace('/', ' '))
# 	sentence = [word for word in sentence if word.isalpha()]
# 	sentence = [w.lower() for w in sentence if w not in stop_words]
# 	return sentence


# Import data
with open('../data/adr-pubmed.csv', 'r') as f:
    pmed_pos = [[0, 0, 1, i[3:-3]] for i in f.readlines()[1:]]

print(pmed_pos[1:10])

with open('../data/no-adr-pubmed.csv', 'r') as f:
    pmed_neg = [[0, 0, 0, i[3:-3]] for i in f.readlines()[1:]]

with open('../data/binary_downloaded.tsv','rb') as f:
	stuff = [ i.decode('utf-8').strip().split('\t')  for i in f.readlines() ]
data_pos = [i for i in stuff if i[2] == '1']
data_neg = [i for i in stuff if i[2] == '0']

pmed = pmed_pos + pmed_neg
data = data_pos + data_neg[:len(data_pos)]

data = pmed + data
# Shuffle the rows of the data for cross validation
random.shuffle(data)

# Pull out Tweets and their labels
Labels , Tweets = zip( *[ [label, tweet] for tweet_id, user_id, label, tweet in data] )

# Clean up tweets
# Tweets = [clean_sentence(tweet) for tweet in Tweets]
tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)
tokenizer.fit_on_texts(Tweets)
word_index = tokenizer.word_index

sequences = tokenizer.texts_to_sequences(Tweets)

Tweets = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
# Labels = to_categorical(np.asarray(Labels))
Labels = np.array([ int( float( label ) ) for label in Labels])

indices = np.arange(Tweets.shape[0])
np.random.shuffle(indices)
Tweets = Tweets[indices]
Labels = Labels[indices]
num_validation_samples = int(VALIDATION_SPLIT * Tweets.shape[0])

trainTweets = Tweets[:-num_validation_samples]
trainLabels = Labels[:-num_validation_samples]
testTweets = Tweets[-num_validation_samples:]
testLabels = Labels[-num_validation_samples:]

num_words = min(MAX_NUM_WORDS, len(word_index) + 1)
embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
for word, i in word_index.items():
    if i >= MAX_NUM_WORDS:
        continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector

# load pre-trained word embeddings into an Embedding layer
# note that we set trainable = False so as to keep the embeddings fixed
embedding_layer = Embedding(num_words,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=False)


# # Get dict to encode words as integers and total number of words
# vocab = set(['<unk>'])
# [vocab.update(tweet) for tweet in Tweets]
# encoder = dict( (word, integer) for integer, word in enumerate(sorted(vocab), 1 ) )  
# vocab_size = len(encoder) + 1

# # Encode tweets and labels as numeric types for LSTM
# Tweets = [[encoder[word] for word in tweet] for tweet in Tweets]
# Labels = numpy.array([ int( float( label ) ) for label in Labels])

"""
Train and test the LSTM
"""
# Get index for 80/20 split of data into training and test
# split = numpy.floor( len(Tweets)*0.8 )
# split = int( split )

# # Split into training and test sets
# trainTweets, testTweets = Tweets[:split], Tweets[split:]
# trainLabels, testLabels = Labels[:split], Labels[split:]

# # Pad training examples for to make constant length
# trainTweets = sequence.pad_sequences(trainTweets, maxlen=max_tweet_words, padding='post')
# testTweets = sequence.pad_sequences(testTweets, maxlen=max_tweet_words, padding='post')

# Create the model
# embedding_vecor_length = 32
model = Sequential()
model.add(embedding_layer)
# model.add(Embedding(vocab_size, embedding_vecor_length, input_length=max_tweet_words))
# model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
# model.add(Bidirectional(LSTM(100, dropout=0.5, recurrent_dropout=0.25, return_sequences=True)))
# model.add(LSTM(100, dropout=0.5, recurrent_dropout=0.25, return_sequences=True))
# model.add(LSTM(100, dropout=0.5, recurrent_dropout=0.25, return_sequences=True))
# model.add(LSTM(100, dropout=0.8, recurrent_dropout=0.2))
# model.add(Bidirectional(LSTM(100)))
model.add(Bidirectional(LSTM(100, return_sequences=True, dropout=0.1, recurrent_regularizer=regularizers.l2(0.01), kernel_regularizer=regularizers.l2(0.01))))
model.add(BatchNormalization())
# model.add(Bidirectional(LSTM(100,  return_sequences=True)))
model.add(LSTM(100, return_sequences=True, recurrent_regularizer=regularizers.l2(0.01), kernel_regularizer=regularizers.l2(0.01)))
model.add(BatchNormalization())
# model.add(LSTM(100,  return_sequences=True))
# model.add(LSTM(100,  return_sequences=True))
# model.add(LSTM(200,  return_sequences=True))
model.add(LSTM(100, recurrent_regularizer=regularizers.l2(0.01)))
model.add(BatchNormalization())
model.add(Dense(10, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
model.add(BatchNormalization())
model.add(Dense(1, activation='sigmoid', kernel_regularizer=regularizers.l2(0.01)))
adam = optimizers.Adam(lr=0.001, amsgrad=True, decay=1e-6)
model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])
print(model.summary())
model.fit(trainTweets, trainLabels, epochs=3, batch_size=256)

# Final evaluation of the model
scores = model.evaluate(testTweets, testLabels, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))
preds = model.predict_classes(testTweets)
# print("Accuracy: %.2f%%" % (scores[1]*100))
print(confusion_matrix(testLabels, preds))
