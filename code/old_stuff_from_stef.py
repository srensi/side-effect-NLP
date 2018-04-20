from gensim.models.keyedvectors import KeyedVectors


# Change to load adverse event words from file
w1 = "hypertension"
w2 = "symptoms"

# Load word vectors
word_vectors = KeyedVectors.load_word2vec_format('../data/PubMed-and-PMC-w2v.bin', binary=True)

# Get rid of stuff in quotes
"""
# test3 = word_vectors.most_similar_cosmul(positive=[w2, w1], topn=20)
test = word_vectors.most_similar_cosmul(positive=w2)
test2 = word_vectors.most_similar_cosmul(positive=w1)
inpt = test + test2
test3 = word_vectors.most_similar_cosmul(positive=[i[0] for i in inpt], topn=20)
# test = word_vectors.most_similar(positive=["hypotension", "dehydration", "polyuria", "gout"])
# test2 = word_vectors.most_similar(positive=["weakness", "dizziness", "urination"], negative=["hypotension", "dehydration"])
print(test)
print(test2)
# print(test3)
stuff = (set([i[0] for i in test3]) - set([i[0] for i in test])) - set([i[0] for i in test2])
print(stuff)
# rev_test = word_vectors.most_similar(positive=list(stuff), negative=[i[0] for i in test])
# rev_test = word_vectors.most_similar(positive=[i[0] for i in test3])
# rev_test = word_vectors.most_similar(positive=list(stuff), topn=20)
# print(rev_test)
"""
