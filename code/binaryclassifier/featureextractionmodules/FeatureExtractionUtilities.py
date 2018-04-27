from featureextractionmodules import twokenize
#from cryptography.x509 import SubjectAlternativeName
from scipy.spatial.distance import cosine
from numpy import nan_to_num
#from __builtin__ import None
__author__ = 'abeedsarker'


#from gensim.models import Word2Vec
from sklearn.cluster import KMeans
from nltk.corpus import wordnet as wn
import nltk,string
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.feature_extraction import DictVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
#from KaggleWord2VecUtility import KaggleWord2VecUtility
import pandas as pd
from sklearn import svm
import numpy as np
#import gloS
import json
import re
#from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer, porter
from nltk.stem.porter import *
#import aspell
import scipy.spatial.distance
stemmer = PorterStemmer()
#s = aspell.Speller('lang', 'en')
from collections import defaultdict




class FeatureExtractionUtilities:
    bingnegs = []
    bingposs = []
    ade_list = []
    sentinegscores = {}
    sentiposscores = {}
    polarity_dict = {}
    topic_keys = {}
    word_clusters = defaultdict(list)
    neg_feature_vecs = np.zeros((100,200),dtype="float32")
    neg_model = None
    pos_feature_vecs = np.zeros((100,200),dtype="float32")
    pos_model=None 
    goodwords = []
    badwords = []
    lesswords = []
    morewords = []
           
    
    @staticmethod
    def loadItems():
        #load bingliu items
        goodwords = []
        badwords = []
        lesswords = []
        morewords = []
        bingposs = []
        bingnegs = []
        ade_list = []
        topic_keys ={}
        word_clusters = defaultdict(list)
        infile = open('./sentimentanalysisresources/bingliunegs.txt')
        for line in infile:
            if not line[0]==';':
                bingnegs.append(stemmer.stem(string.strip(line.decode('utf8','ignore').encode('ascii','ignore'))))
        infile = open('./sentimentanalysisresources/bingliuposs.txt')
        for line in infile:
            if not line[0]==';':
                bingposs.append(stemmer.stem(string.strip(line.decode('utf8','ignore').encode('ascii','ignore'))))
        FeatureExtractionUtilities.bingnegs = bingnegs  
        FeatureExtractionUtilities.bingposs = bingposs
        
        sentinegscores = {}
        sentiposscores = {}
        
        infile = open('sentimentanalysisresources/SentiWordNet_3.0.txt')
        for line in infile:
            if not line[0]=='#' and not line[0] == ';':
                    items = line.split('\t')
                    pos = items[0]
                    id_ = items[1]
                    posscore = items[2]
                    negscore = items[3]
                    term = stemmer.stem(items[4][:items[4].index('#')].decode('utf8','ignore').encode('ascii','ignore'))
                    #print term
                    sentiposscores[(term,pos)]= posscore
                    sentinegscores[(term,pos)] = negscore

        FeatureExtractionUtilities.sentinegscores = sentinegscores
        FeatureExtractionUtilities.sentiposscores = sentiposscores
        
        #loadclusters
        
        #load the subjectivity scores
        polarity_dict={}
        infile = open('sentimentanalysisresources/subjectivity_score.tff')
        for line in infile:
            if not line[0] == ';':
                items = line.split()
                type = items[0][5:]
                word = stemmer.stem(items[2][6:])
                pos = items[3][5:]
                polaritystr = items[5][14:]
                #print type
                #print word
                #print type
                #print pos
                #print polaritystr
                multip = 0.0
                pol = 0.0
                if type =='strongsubj':
                    multip=1.0
                if type =='weaksubj':
                    multip=0.5
                if polaritystr == 'positive':
                    pol = 1.0
                if polaritystr == 'negative':
                    pol = -1.0
                if polaritystr=='neutral':
                    pol = 0.0
                polval = multip*pol
                polarity_dict[(word,pos)]=polval
        
        infile = open('featureextractionmodules/50mpaths2.txt')
        for line in infile:
                items = line.split()
                class_ = items[0]
                term=items[1]
                word_clusters[class_].append(term)
        FeatureExtractionUtilities.word_clusters = word_clusters 
        
        infile = open('ADR_lexicon.tsv')
        for line in infile:
            items = line.split('\t')
            if len(items[1])>6:
                _name = string.strip(string.lower(items[1]))
                _cui = string.strip(items[0])
                ade_list.append((_cui,_name))
        FeatureExtractionUtilities.ade_list = ade_list  

        max_weight = 0.0
        infile = open('TW_keys.txt')
        for line in infile:
            items = line.split()
            weight = float(items[1])
            if weight>max_weight:
                max_weight = weight
            for t in items[2:]:
                topic_keys[string.strip(t)] = weight
            
        for k in topic_keys.keys():
            topic_keys[k] = topic_keys[k]/max_weight    
        FeatureExtractionUtilities.topic_keys = topic_keys
    
        #LOAD THE GOOD/BAD/MORE/LESS WORDS
        FeatureExtractionUtilities.loadgoodbadwords()
        
    @staticmethod
    def generateCentroidSimilarityScore(sent):
        #print sent
        terms = twokenize.tokenizeRawTweetText(sent)
        
        averagevec = np.zeros((300,),dtype="float32")
        for t in terms:
            try:
                averagevec = np.add(averagevec,FeatureExtractionUtilities.neg_model[t.lower()])
            except KeyError:
                pass
        try:
            averagevec = np.divide(averagevec,len(terms))
        except:
            pass
            
        sims = []
        for nfv in FeatureExtractionUtilities.neg_feature_vecs:
            sims.append(cosine(nfv,averagevec))
        averagepvec = np.zeros((300,),dtype="float32")
        for t in terms:
            try:
                averagepvec = np.add(averagepvec,FeatureExtractionUtilities.neg_model[t.lower()])
            except KeyError:
                pass
        if len(terms)>0:
            averagevec = np.divide(averagepvec,len(terms))
        for nfv in FeatureExtractionUtilities.neg_feature_vecs:
            sims.append(cosine(nfv,averagepvec))
            
        #print sims
        return nan_to_num(sims)
            
            
         
    @staticmethod
    def makeFeatureVec(words, model, num_features):
        # Function to average all of the word vectors in a given
        # paragraph
        #
        # Pre-initialize an empty numpy array (for speed)
        featureVec = np.zeros((num_features,),dtype="float32")
        #
        nwords = 0.
        # 
        # Index2word is a list that contains the names of the words in 
        # the model's vocabulary. Convert it to a set, for speed 
        index2word_set = set(model.index2word)
        #
        # Loop over each word in the review and, if it is in the model's
        # vocaublary, add its feature vector to the total
        for word in words:
            if word in index2word_set: 
                nwords = nwords + 1.
                featureVec = np.add(featureVec,model[word])
        # 
        # Divide the result by the number of words to get the average
        if len(words)>0:
            featureVec = np.divide(featureVec,nwords)
        return featureVec

    @staticmethod
    def getAvgFeatureVecs(reviews, model, num_features):
        num_features=300
        # Given a set of reviews (each one a list of words), calculate 
        # the average feature vector for each one and return a 2D numpy array 
        # 
        # Initialize a counter
        counter = 0.
        # 
        # Preallocate a 2D numpy array, for speed
        reviewFeatureVecs = np.zeros((len(reviews),num_features),dtype="float32")
        # 
        # Loop through the reviews
        for k in reviews.keys():
           review = reviews[k] 
           #
           # Print a status message every 1000th review
           if counter%1000. == 0.:
               print "Review %d of %d" % (counter, len(reviews))
           # 
           # Call the function (defined above) that makes average feature vectors
           reviewFeatureVecs[counter] = FeatureExtractionUtilities.makeFeatureVec(review, model, num_features)
           #
           # Increment the counter
           counter = counter + 1.
        return reviewFeatureVecs   
    @staticmethod
    def getclusterfeatures(sent):   
        terms = twokenize.tokenizeRawTweetText(sent)
        #pos_tags = nltk.pos_tag(terms, 'universal')
        #terms = parsed_sent.split('\t')
        cluster_string = ''
        for t in terms:
            for k in FeatureExtractionUtilities.word_clusters.keys():
                if t in FeatureExtractionUtilities.word_clusters[k]:
                    cluster_string+= ' clust_'+ k + '_clust '
        return cluster_string     
    @staticmethod
    def getbingliuscores(processed_data):
        bingposcount = 0.0
        bingnegcount = 0.0
        bposcounts = []
        bnegcounts = []
        for d in processed_data:
            bingposcount = 0.0
            bingnegcount = 0.0
        
            items = d.split()
            for i in items:
                if i in FeatureExtractionUtilities.bingnegs:
                    bingnegcount +=1
                if i in FeatureExtractionUtilities.bingposs:
                    bingposcount +=1
            bposcounts.append([bingposcount/len(items)])
            bnegcounts.append([bingnegcount/len(items)])
        #print bnegcounts
        #print bposcounts
        return bnegcounts,bposcounts
    
    @staticmethod
    def getsentiwordscores(processed_data):
        negscore = 0.0
        posscore = 0.0
        negscores = []
        posscores = []
        for d in processed_data:
            negscore = 0.0
            posscore = 0.0
        
            terms = twokenize.tokenizeRawTweetText(d)
            pos_tags = nltk.pos_tag(terms, 'universal')
            for i in range(0,len(pos_tags)):
                try:
                    if string.lower(str(pos_tags[i][1]))=='adj':
                        if FeatureExtractionUtilities.sentiposscores.has_key((string.lower(str(pos_tags[i][0])),'a')):
                            posscore+= float(FeatureExtractionUtilities.sentiposscores[(string.lower(str(pos_tags[i][0])),'a')])
                        if FeatureExtractionUtilities.sentinegscores.has_key((string.lower(str(pos_tags[i][0])),'a')):
                            negscore+= float(FeatureExtractionUtilities.sentinegscores[(string.lower(str(pos_tags[i][0])),'a')])
                    if string.lower(str(pos_tags[i][1]))=='verb':
                        if FeatureExtractionUtilities.sentiposscores.has_key((string.lower(str(pos_tags[i][0])),'v')):
                            posscore+= float(FeatureExtractionUtilities.sentiposscores[(string.lower(str(pos_tags[i][0])),'v')])
                        if FeatureExtractionUtilities.sentinegscores.has_key((string.lower(str(pos_tags[i][0])),'v')):
                            negscore+= float(FeatureExtractionUtilities.sentinegscores[(string.lower(str(pos_tags[i][0])),'v')])
                    if string.lower(str(pos_tags[i][1]))=='noun':
                        if FeatureExtractionUtilities.sentiposscores.has_key((string.lower(str(pos_tags[i][0])),'n')):
                            posscore+= float(FeatureExtractionUtilities.sentiposscores[(string.lower(str(pos_tags[i][0])),'n')])
                        if FeatureExtractionUtilities.sentinegscores.has_key((string.lower(str(pos_tags[i][0])),'n')):
                            negscore+= float(FeatureExtractionUtilities.sentinegscores[(string.lower(str(pos_tags[i][0])),'n')])
                except Exception:
                    pass
            negscores.append([negscore])
            posscores.append([posscore])
        #print negscores
        #print posscores
        return negscores,posscores
    @staticmethod
    def getsubjectivityscores(processed_data):
        subjectivity_scores = []
        for d in processed_data:
            subjectivity_score = 0.0
            
            terms = twokenize.tokenizeRawTweetText(d)
            pos_tags = nltk.pos_tag(terms, 'universal')
            for i in range(0,len(pos_tags)):
                try:
                    if FeatureExtractionUtilities.polarity_dict.has_key(pos_tags[i]):
                        print 'yes'
                        subjectivity_score+=FeatureExtractionUtilities.polarity_dict[pos_tags[i]]
                except Exception:
                    pass
            subjectivity_score = subjectivity_score/len(terms)
            subjectivity_scores.append([subjectivity_score])
        return subjectivity_scores
    
    @staticmethod
    def getlexiconfeatures(processed_data):
        lexicon_features = []
        for d in processed_data:
            ade_presence = 0
            ade_count = 0.0
            #to make sure that concepts with the same cui are not searched multiple times
            addedcuilist = []
            sentence=string.lower(d)
            for (cui,ade) in FeatureExtractionUtilities.ade_list:
                if re.search(ade,sentence):
                    ade_presence = 1.0
                    if not cui in addedcuilist:
                        ade_count +=1.0
                        addedcuilist.append(cui)
                    
            ade_count = ade_count/len(sentence.split())
            lexicon_features.append([ade_presence,ade_count])
        #print lexicon_features
        #print lexicon_features
        return lexicon_features  
    
    @staticmethod
    def gettopicscores(processed_data):
        topic_features = []
        topic_texts = []
        for d in processed_data:
            weighted_score = 0.0
            #topic_presence = 0
            topic_terms = ''
            for k in FeatureExtractionUtilities.topic_keys.keys():
                if k in d.split():
                    topic_presence = 1
                    topic_terms += 'top_'+k+'_top '
                    weighted_score += FeatureExtractionUtilities.topic_keys[k]
            topic_features.append([weighted_score])
            topic_texts.append(topic_terms)
        return topic_texts,topic_features
        #return topic_terms, weighted_score   
                
    @staticmethod
    def getsentimentfeatures(processed_data):
        negcounts,poscounts = FeatureExtractionUtilities.getbingliuscores(processed_data)
        negscores,posscores = FeatureExtractionUtilities.getsentiwordscores(processed_data)
        subjectivity_scores = FeatureExtractionUtilities.getsubjectivityscores(processed_data)
        
        features = map(list.__add__,poscounts,negcounts)
        features2 = map(list.__add__,posscores,negscores)
        features = map(list.__add__,features,features2)
        features = map(list.__add__,features,subjectivity_scores)
    
        return features 
    
    @staticmethod
    def getstructuralfeatures(processed_data):
        lens = FeatureExtractionUtilities.getreviewlengths(processed_data)
        numsents = FeatureExtractionUtilities.getnumsentences(processed_data)
        avelengths = FeatureExtractionUtilities.getaveragesentlengths(processed_data)
    
        features = map(list.__add__,lens,avelengths)
        features = map(list.__add__,features,numsents)
        return features
    
    @staticmethod
    def getreviewlengths(processed_data):
        lengths = []
        for d in processed_data:
            items = d.split()
            lengths.append([len(items)])
    
        return lengths
    @staticmethod
    def getnumsentences(processed_data):
        numsents = []
        for d in processed_data:
            items = nltk.sent_tokenize(d)
            numsents.append([len(items)])
        return numsents
    @staticmethod
    def getaveragesentlengths(processed_data):
        avelengths = []
        for d in processed_data:
            items = nltk.sent_tokenize(d)
            words = d.split()
            numsents = len(items)
            numwords = len(words)
            avelengths.append([numwords/(numsents+0.0)])
        return avelengths
    
    

    @staticmethod
    def getSynsetString(sent, negations):
        terms = twokenize.tokenizeRawTweetText(sent)
        pos_tags = nltk.pos_tag(terms, 'universal')
      
        #terms = parsed_sent.split('\t')
        sent_terms = []
        #now terms[0] will contain the text and terms[1] will contain the POS (space separated)
        #sentence_tokens = terms[0].split()
        #pos_tags = terms[1].split()
        for i in range(0,len(pos_tags)):
          
            if string.lower(str(pos_tags[i][1]))=='adj':
                synsets = wn.synsets(string.lower(pos_tags[i][0]),pos=wn.ADJ)
                for syn in synsets:
                    lemmas=[string.lower(lemma) for lemma in syn.lemma_names()]
                    sent_terms += lemmas
        
            if string.lower(pos_tags[i][1])=='verb':
                synsets = wn.synsets(string.lower(pos_tags[i][0]),pos=wn.VERB)
                for syn in synsets:
                    lemmas=[string.lower(lemma) for lemma in syn.lemma_names()]
                    sent_terms += lemmas
            
            if string.lower(pos_tags[i][1])=='noun':
              
                synsets = wn.synsets(string.lower(pos_tags[i][0]),pos=wn.NOUN)
                for syn in synsets:
                   
                    lemmas=[string.lower(lemma) for lemma in syn.lemma_names()]
                    
                    sent_terms += lemmas
        sent_terms = list(set(sent_terms))
        #print sent_terms
        senttermsstring = ''
    
        for term in sent_terms:
            senttermsstring += ' ' +'syn_'+ stemmer.stem(term)+'_syn'
        #print senttermsstring
        return senttermsstring
    
    
    @staticmethod
    def detectModals(senttokens,modals):
        for word in senttokens:
            for item in modals:
                if cmp(string.lower(item),string.lower(word))==0:
                    return 1
        return 0
    
    @staticmethod
    def loadModals():
        modals = []
        infile = open('./polaritycues/modals.txt')
        for line in infile:
            modals.append(string.strip(line))
        return modals

    @staticmethod
    def loadGoodWords():
        goodwords = []
        infile  = open('./polaritycues/new_good_words.txt')
        for line in infile:
            goodwords.append(stemmer.stem(string.strip(line)))
        FeatureExtractionUtilities.goodwords = goodwords
    
    @staticmethod
    def loadBadWords():
        badwords = []
        infile  = open('./polaritycues/new_bad_words.txt')
        for line in infile:
            badwords.append(stemmer.stem(string.strip(line)))        
        FeatureExtractionUtilities.badwords= badwords
    
    @staticmethod
    def loadMoreWords():
        morewords = []
        infile = open('./polaritycues/more_words.txt')
        for line in infile:
            morewords.append(stemmer.stem(string.strip(line)))
        FeatureExtractionUtilities.morewords= morewords
    
    @staticmethod
    def loadLessWords():
        lesswords = []
        infile = open('./polaritycues/less_words.txt')
        for line in infile:
            lesswords.append(stemmer.stem(string.strip(line)))
        FeatureExtractionUtilities.lesswords= lesswords
    
    @staticmethod
    def loadgoodbadwords():
        FeatureExtractionUtilities.loadGoodWords()
        FeatureExtractionUtilities.loadBadWords()
        FeatureExtractionUtilities.loadMoreWords()
        FeatureExtractionUtilities.loadLessWords()
        

    '''
        Given a sentence, addsmore/less..good/bad features as proposed by niu et al.
        once a more/less word is found, a 4 word window is inspected on either side to detect
         the presence of good/bad words.
         returns a binary vector of the form: [moregood,morebad,lessgood,lessbad]
    '''
    @staticmethod
    def goodbadFeatures(processed_data):
        goodbadfeatures =[]
        for pd in processed_data:

            moreGood = 0
            moreBad = 0
            lessGood = 0
            lessBad = 0
            sentence = string.lower(pd)
            word_tokens = nltk.word_tokenize(sentence)
            for i in range(0,len(word_tokens)):
                stemmedword =word_tokens[i]
                if stemmedword in FeatureExtractionUtilities.morewords:
                    
                    minboundary = max(i-4,0)
                    maxboundary = min(i+4,len(word_tokens)-1)
                    j = minboundary
                    while j<=maxboundary:
                        if word_tokens[j] in FeatureExtractionUtilities.goodwords:
                            moreGood = 1
                        elif word_tokens[j] in FeatureExtractionUtilities.badwords:
                            moreBad = 1
                        j+=1    
                if stemmedword in FeatureExtractionUtilities.lesswords:
                    minboundary = max(i-4,0)
                    maxboundary = min(i+4,len(word_tokens)-1)
                    j = minboundary
                    while j<=maxboundary:
                        if word_tokens[j] in FeatureExtractionUtilities.goodwords:
                            lessGood = 1
                        elif word_tokens[j] in FeatureExtractionUtilities.badwords:
                            lessBad = 1
                        j+=1
            goodbadfeatures.append([moreGood,moreBad,lessGood,lessBad])
        return goodbadfeatures