from gensim.models.keyedvectors import KeyedVectors
import numpy as np

all_MEDDRA = []
all_MEDDRA_firest_term = []

# Load MEDDRA terms
with open('../lasix_meddra.tsv') as file:
	lines_in_file = file.read().splitlines()
	for line in lines_in_file:
		line_list = line.split('\t')
		meddra_term = line_list[5]
		all_MEDDRA.append(meddra_term)
		all_MEDDRA_firest_term.append(meddra_term.split()[0])

# Change to load adverse event words from file
w1 = "hypertension"
w2 = "symptoms"

# Load word vectors
word_vectors = KeyedVectors.load_word2vec_format('../data/PubMed-and-PMC-w2v.bin', binary=True)
MEDDRA_word_vectors = []
word_vecs_found_for = []
no_word_vecs = []
for word in all_MEDDRA_firest_term:
	try:
		word_vec = word_vectors.get_vector(word)
		MEDDRA_word_vectors.append(word_vec)
		word_vecs_found_for.append(word)
	except KeyError, e:
		no_word_vecs.append(word)

MEDDRA_word_vectors = np.array(MEDDRA_word_vectors)

print word_vecs_found_for
print no_word_vecs
print len(word_vecs_found_for), len(no_word_vecs)


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
