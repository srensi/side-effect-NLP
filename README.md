# side-effect-NLP
NLP Algorithms for detecting adverse events in patient narratives.

## Data download
cd side-effect-NLP  
mkdir data  
cd data  
wget http://evexdb.org/pmresources/vec-space-models/PMC-w2v.bin  

## Strategy
Word Vector and other resources  
BioLab NLP homepage: http://bio.nlplab.org/#word-vectors  
Pretrained Vectors: http://evexdb.org/pmresources/vec-space-models/  
Side Effect Database: http://sideeffects.embl.de/  
Pubmed (biomedical lit database): https://www.ncbi.nlm.nih.gov/pubmed/  
Gensim Tutorial: https://radimrehurek.com/gensim/models/word2vec.html  

## Pseudo Code

Step 0.
Pull all word vectors for set of MedDRA terms Y = (y_1, y_2, …, y_n)
If meddra term consist of 2 words, throw away second word.

Step 1. Input sentence. Output array of symptom words
Extract symptom potential words/terms
- Normalize words/terms (we will need to find tool for this - NIH, Emily M.)

Step 2. Input symptom words (w_1, w_2, …, w_n).  Output sum of symptom word vectors (x_sum).
Get vectors for words (w_1, w_2, …, w_n)
Return array of word vectors (x_1, x_2, …, x_n)

Step 3.  Input array of word vectors X.  Return sum, average, or max of word vectors x_agg = agg_fxn(x_1, x_2, …, x_n).


Step 4. Input aggregate word vector (x_agg) and array of MedDRA word vectors (y_1, y_2, …, y_n).  Output medDRA term.
For each medDRA word vector y_i in array
compute dist(x_agg, y_i) = z_i
softmax(Z)
return argmin(Z)

Side Effects File.
Note columns 5 and 6 are most important. (indexing from 1)


# More Data Resources
## Binary ADR Classifier for Tweets
Azadeh's adverse drug reaction (ADR) lexicon: https://healthlanguageprocessing.files.wordpress.com/2018/03/adr_lexicon.zip    
Annotated (labeled) tweets: https://healthlanguageprocessing.files.wordpress.com/2018/03/download_tweets1.zip  

Twiiter dataset for binary classification: https://healthlanguageprocessing.files.wordpress.com/2018/03/adr_classify_twitter_data.zip  
Script for downloading tweets: https://healthlanguageprocessing.files.wordpress.com/2018/03/download_binary_twitter_data.zip  
Polarity cues (?): https://healthlanguageprocessing.files.wordpress.com/2018/03/polaritycues.zip  
General twitter dataset: https://healthlanguageprocessing.files.wordpress.com/2018/03/download_tweets.zip  
Binary Classifier Repo: https://bitbucket.org/asarker/adrbinaryclassifier/get/bce087f4cc5d.zip  

