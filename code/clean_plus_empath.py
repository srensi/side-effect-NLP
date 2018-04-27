import string
import datetime
import sys
import pandas as pd
import numpy as np
import nltk
from empath import Empath
import re

filePath = sys.argv[1]

print(filePath)

subID = filePath.split("/")[3].split("_")[0]
print(subID)
outPathClean = "/share/PI/rbaltman/alavertu/Screenome/data/analysis_data/cleaned_text/"
outPathEmpath = "/share/PI/rbaltman/alavertu/Screenome/data/analysis_data/empath_scored_dat/"

refWordsToKeep = "../data/reference/nonWordsToKeep.txt"
words2Keep = []
with open(refWordsToKeep) as inRef:
    for line in inRef.readlines():
        words2Keep.append(line.strip())

print("Cleaning data...")
data = pd.read_csv(filePath)
data = data.dropna(subset=['id'])
stripped = [nltk.word_tokenize(str(w)) for w in data.iloc[:,1]]

stop_words = set(nltk.corpus.stopwords.words('english') + nltk.corpus.stopwords.words('spanish')+ ['el', 'en'])
words = set(nltk.corpus.words.words() + nltk.corpus.cess_esp.words() + words2Keep)

word_filtered=[]
for x in stripped:
    x = [word.lower() for word in x]
    word_filtered.append(" ".join([word for word in x if word.isalpha() and word not in stop_words and word in words and word not in string.ascii_letters]))
    
data['cleaned_text'] = word_filtered

def getTimeSecs(temp):
    m = re.search('(2017).{10}',temp)
    timeStamp = m.group(0)
    year = int(timeStamp[:4])
    month = int(timeStamp[4:6])
    day = int(timeStamp[6:8])
    hour = int(timeStamp[8:10])
    minute= int(timeStamp[10:12])
    second = int(timeStamp[12:14])
    t = datetime.datetime(year, month, day, hour, minute, second)
    timeSecs = int((t-datetime.datetime(1970,1,1)).total_seconds())
    return(timeSecs)

dateAndTime= [getTimeSecs(x) for x in data.iloc[:,0]]

data['timeInSeconds'] = dateAndTime

print("Writing clean data to file...")
data = data[['id', 'timeInSeconds', 'text_all', 'cleaned_text']]
data.to_csv(outPathClean + subID + "_cleaned.csv.gz", index=False, quoting=1,compression='gzip')


print("Calculating empath scores...")
lexicon = Empath()
keys_oi = ["depression_drugs","depression","diabetes_drugs","diabetes","nsaids","pain","side_effect"]
default_values = {"depression_drugs":0,"depression":0,"diabetes_drugs":0,"diabetes":0,"nsaids":0,"pain":0,"side_effect":0}

empath_scores = []
for index, row in data.iterrows():
    scores = []
    temp_default = lexicon.analyze(row['cleaned_text'], tokenizer="default", categories=keys_oi, normalize=False)
    try:
        temp_bigram = lexicon.analyze(row['cleaned_text'], tokenizer="bigrams", categories=keys_oi, normalize=False)
    except IndexError:
        temp_bigram = default_values
    for key in keys_oi:
        score = temp_default[key] + temp_bigram[key]
        scores.append(score)
    empath_scores.append(scores)
 
tempFile = data.join(pd.DataFrame(empath_scores, columns=keys_oi))
final = tempFile.sort_values('timeInSeconds')
print("Writing empath data to file...")
final.to_csv(outPathEmpath + subID + "_cleaned_with_empath_stats.csv.gz", index=False, quoting=1, compression='gzip')
