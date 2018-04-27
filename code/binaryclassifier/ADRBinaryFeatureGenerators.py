'''
Created on Mar 19, 2016

@author: asarker

#Copyright 2016 The authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================


This is a binary classifier for social media posts. The classifier learns to distinguish between posts that mention 
adverse drug reactions and those that don't. Please cite the following publication when using this script:

Sarker A, Gonzalez G. Portable automatic text classification for adverse drug reaction detection via multi-corpus training.
J Biomed Inform. 2015 Feb;53:196-207. doi: 10.1016/j.jbi.2014.11.002. Epub 2014 Nov 8.
PMID: 25451103

Data set available at: http://diego.asu.edu/Publications/ADRClassify.html

The performance of this SVM classifier varies with the kernel, the cost parameter and the weights. 

Last best run used the parameters:
kernel: RBF
cost (c): 140
weight: 3

Please run 10-fold cross validation with a range of values to optimize classifier for a new data set.

*******THIRD PARTY RESOURCES*******
The classifier utilizes a number of third party resources. 


SENTIMENT SCORES
1. POSITIVE AND NEGATIVE TERMS
AVAILABLE PUBLICLY AT: https://www.cs.uic.edu/~liub/FBS/sentiment-analysis.html#lexicon

CITATION:

2. PRIOR POLARITIES
AVAILABLE PUBLICLY AT: https://hlt-nlp.fbk.eu/technologies/sentiwords
CITATION:
Marco Guerini, Lorenzo Gatti, and Marco Turchi. 2013. Sentiment Analysis: How to Derive Prior Polarities
from SentiWordNet. In Proceedings of Empirical Methods in Natural Language Processing (EMNLP), pages 1259-1269.

3. MULTI-PERSPECTIVE QUESTION ANSWERING
AVAILABLE AT: http://mpqa.cs.pitt.edu/lexicons/subj_lexicon/
LICENSE: GNU public license: http://www.gnu.org/licenses/gpl.html


TWITTER WORD CLUSTERS
filename: 50mpaths2.txt
Available at: http://www.cs.cmu.edu/~ark/TweetNLP/
CITATION: Olutobi Owoputi, Brendan O'Connor, Chris Dyer Kevin Gimpel, and Nathan Schneider. 2012. Part-of-Speech Tagging for Twitter: Word Clusters and Other Ad-
vances. Technical report, School of Computer Science, Carnegie Mellon University.


***************************

SAMPLE FILE FORMAT:

TWEET_ID [TAB] USER_ID [TAB] CLASS (1 -> ADR, 0 -> NOADR) [TAB] TWEET

The sample file contains 31 anonymyzed entries.
More data available at: http://diego.asu.edu/Publications/ADRClassify.html

'''

import string
from nltk.stem.porter import *
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import preprocessing
import numpy as np
from sklearn import svm
from collections import defaultdict
from featureextractionmodules.FeatureExtractionUtilities import FeatureExtractionUtilities

stemmer = PorterStemmer()


def loadFeatureExtractionModuleItems():
    '''
        Load the various feature extraction resources
    '''
    FeatureExtractionUtilities.loadItems()

def loadData(f_path):
    '''
        Given a path, loads a data set and puts it into a dataframe
    '''
    loaded_data_set = defaultdict(list)
    infile = open(f_path)
    for line in infile:
        line = line.decode('utf8', 'ignore').encode('ascii', 'ignore')
        try:
            items = line.split('\t')
            if len(items) > 3:
                tweet_id =  items[0]
                user_id = items[1]
                text = string.lower(string.strip(items[-1]))
                class_ = items[2]

                senttokens = text.split()  #nltk.word_tokenize(_text)
                stemmed_text = ''
                for t in senttokens:
                    stemmed_text += ' ' + stemmer.stem(t)


                loaded_data_set['id'].append(tweet_id + '-' + user_id)
                loaded_data_set['synsets'].append(FeatureExtractionUtilities.getSynsetString(text, None))
                loaded_data_set['clusters'].append(FeatureExtractionUtilities.getclusterfeatures(text))
                loaded_data_set['text'].append(stemmed_text)
                loaded_data_set['unstemmed_text'].append(text)
                loaded_data_set['class'].append(class_)

        except UnicodeDecodeError:
            print 'please convert to correct encoding..'

    infile.close()
    return loaded_data_set

if __name__ == '__main__':
    #LOAD THE FEATURE EXTRACTION RESOURCES
    loadFeatureExtractionModuleItems()

    #LOAD THE DATA -- *SAMPLE SCRIPT USES THE SAME DATA FOR TRAINING AND TESTING*
    data_set_filename = 'adr_classify_twitter_data_downloaded.txt'
    training_data = loadData(data_set_filename)
    testing_data = loadData(data_set_filename)


    #GENERATE THE TRAINING SET FEATURES
    print 'GENERATING TRAINING SET FEATURES.. '
    training_data['sentiments'] = FeatureExtractionUtilities.getsentimentfeatures(training_data['unstemmed_text'])
    training_data['structuralfeatures'] = FeatureExtractionUtilities.getstructuralfeatures(training_data['unstemmed_text'])
    training_data['adrlexicon'] = FeatureExtractionUtilities.getlexiconfeatures(training_data['unstemmed_text'])
    training_data['topictexts'], training_data['topics'] = FeatureExtractionUtilities.gettopicscores(training_data['text'])
    training_data['goodbad'] = FeatureExtractionUtilities.goodbadFeatures(training_data['text'])

    #SCALE THE STRUCTURAL FEATURES
    scaler1 = preprocessing.StandardScaler().fit(training_data['structuralfeatures'])
    train_structural_features = scaler1.transform(training_data['structuralfeatures'])

    #INITIALIZE THE VARIOUS VECTORIZERS
    vectorizer = CountVectorizer(ngram_range=(1,3), analyzer = "word", tokenizer = None, preprocessor = None, max_features = 5000)
    synsetvectorizer = CountVectorizer(ngram_range=(1,1),analyzer="word",tokenizer=None,preprocessor=None,max_features = 2000)
    clustervectorizer = CountVectorizer(ngram_range=(1,1),analyzer="word",tokenizer=None,preprocessor=None,max_features = 1000)
    topicvectorizer = CountVectorizer(ngram_range=(1,1),analyzer="word",tokenizer=None,preprocessor=None,max_features=500)

    #FIT THE TRAINING SET VECTORS
    print 'VECTORIZING TRAINING SET FEATURES.. '
    training_data_vectors = vectorizer.fit_transform(training_data['text']).toarray()
    train_data_synset_vector = synsetvectorizer.fit_transform(training_data['synsets']).toarray()
    train_data_cluster_vector = clustervectorizer.fit_transform(training_data['clusters']).toarray()
    train_data_topic_vector = topicvectorizer.fit_transform(training_data['topictexts']).toarray()
    
    #CONCATENATE THE TRAINING SET VECTORS
    training_data_vectors = np.concatenate((training_data_vectors, train_data_synset_vector), axis=1)
    training_data_vectors = np.concatenate((training_data_vectors, training_data['sentiments']), axis=1)
    training_data_vectors = np.concatenate((training_data_vectors, train_data_cluster_vector), axis=1)
    training_data_vectors = np.concatenate((training_data_vectors, train_structural_features), axis=1)
    training_data_vectors = np.concatenate((training_data_vectors, training_data['adrlexicon']), axis=1)
    training_data_vectors = np.concatenate((training_data_vectors, training_data['topics']), axis=1)
    training_data_vectors = np.concatenate((training_data_vectors, train_data_topic_vector), axis=1)
    training_data_vectors = np.concatenate((training_data_vectors, training_data['goodbad']), axis=1)
 
    
    #GENERATE THE TEST SET FEATURES
    print 'GENERATING TEST SET FEATURES.. '
    testing_data['sentiments'] = FeatureExtractionUtilities.getsentimentfeatures(testing_data['unstemmed_text'])
    testing_data['structuralfeatures'] = FeatureExtractionUtilities.getstructuralfeatures(testing_data['unstemmed_text'])
    testing_data['adrlexicon'] = FeatureExtractionUtilities.getlexiconfeatures(testing_data['unstemmed_text'])
    testing_data['topictexts'],testing_data['topics'] = FeatureExtractionUtilities.gettopicscores(testing_data['text'])
    testing_data['goodbad'] = FeatureExtractionUtilities.goodbadFeatures(testing_data['text'])

    #TRANSFORM THE TEST SET STRUCTURAL FEATURES
    test_structural_features = scaler1.transform(testing_data['structuralfeatures'])

    #TRANSFORM THE TEST SET VECTORS
    print 'VECTORIZING TEST SET FEATURES.. '
    test_data_vectors = vectorizer.transform(testing_data['text']).toarray()
    test_data_synset_vectors = synsetvectorizer.transform(testing_data['synsets']).toarray()
    test_data_cluster_vectors = clustervectorizer.transform(testing_data['clusters']).toarray()
    test_data_topic_vectors = topicvectorizer.transform(testing_data['topictexts']).toarray()

    #CONCATENATE THE TEST SET VECTORS
    test_data_vectors = np.concatenate((test_data_vectors, test_data_synset_vectors), axis=1)
    test_data_vectors = np.concatenate((test_data_vectors, testing_data['sentiments']), axis=1)
    test_data_vectors = np.concatenate((test_data_vectors, test_data_cluster_vectors), axis=1)
    test_data_vectors = np.concatenate((test_data_vectors, test_structural_features), axis=1)
    test_data_vectors = np.concatenate((test_data_vectors, testing_data['adrlexicon']), axis=1)
    test_data_vectors = np.concatenate((test_data_vectors, testing_data['topics']), axis=1)
    test_data_vectors = np.concatenate((test_data_vectors, test_data_topic_vectors), axis=1)
    test_data_vectors = np.concatenate((test_data_vectors, testing_data['goodbad']), axis=1)
    

    #TRAIN THE SVM CLASSIFIER
    print 'TRAINING THE CLASSIFIER WITH THE FOLLOWING PARAMETERS: '

    f_scores = {}
    c = 140
    w = 3
    print 'C = ' + str(c) + ', ' + 'positive class weight ' + ' = ' + str(w)+'\n'

    svm_classifier = svm.SVC(C=c, cache_size=200, class_weight={'1':w,'0':1}, coef0=0.0, degree=3,
                             gamma='auto', kernel='rbf', max_iter=-1, probability=True, random_state=None,
                             shrinking=True, tol=0.001, verbose=False)
    svm_classifier = svm_classifier.fit(training_data_vectors, training_data['class'])

    #MAKE PREDICTIONS ON THE TEST SET
    print 'MAKING PREDICTIONS ON THE TEST SET..\n'
    result = svm_classifier.predict(test_data_vectors)
    test_gold_classes = testing_data['class']
    
    #COMPUTE THE ADR F-SCORE
    print 'PERFORMANCE METRICS:\n'
    try:
        tp=0.0
        tn=0.0
        fn=0.0
        fp=0.0
        for pred,gold in zip(result,test_gold_classes):
            if pred == '1' and gold == '1':
                tp+=1
            if pred == '1' and gold == '0':
                fp+=1
            if pred == '0' and gold == '0':
                tn +=1
            if pred == '0' and gold == '1':
                fn+=1
        adr_prec = tp/(tp+fp)
        adr_rec = tp/(tp+fn)
        fscore = (2*adr_prec*adr_rec)/(adr_prec + adr_rec)
        print 'Precision for the ADR class .. ' + str(adr_prec)
        print 'Recall for the ADR class .. ' + str(adr_rec)
        print 'ADR F-score .. ' + str(fscore)
        f_scores[str(c)+'-'+str(w)] = fscore
    except ZeroDivisionError:
        print 'There was a zerodivisionerror'
        print 'Precision for the ADR class .. ' + str(0)
        print 'Recall for the ADR class .. ' + str(0)
        print 'ADR F-score .. ' + str(0)