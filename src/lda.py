# Code Referenced From: 
# http://www.cs.cornell.edu/~xanda/winlp2017.pdf
# https://towardsdatascience.com/topic-modeling-and-latent-dirichlet-allocation-in-python-9bf156893c24

# implementing LDA with gensim
# 1. import dataset
# 2. preprocess text
# 3. create gensim dictionary and corpus
# 4. build the topic model
# 5. analyze

import numpy as np
import pandas as pd
import re
import string
import gensim
import pyLDAvis.gensim
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from gensim import corpora, models, similarities 

import time
start = time.time()

nltk.download('all')

# (1) IMPORT DATASET
data = pd.read_csv('metadata.csv', low_memory = False)
keep_columns = ['abstract', 'publish_time']

# new_data contains the data in metadata.csv, but it only keeps the
# abstract and publish time
new_data = data[keep_columns]

# new_data is stored in a new csv file, this is the file that we'll be
# working with
new_data.to_csv("newdata.csv", index = False)
file = 'newdata.csv'

# print out the first five rows of the data that we'll be working with
print(new_data.head(5))

# (2) PREPROCESS TEXT
def clean(text):
  # remove punctuation
  text = re.sub("[^a-zA-Z ]", "", str(text))

  # lowercase everything
  text = text.lower()

  # tokenize the text
  text = nltk.word_tokenize(text)
  return text

# as seen in the first few runs of lda, these words contributed to the noise
# when it came to topic modeling. To remove the noise, we'll remove these words
# and get the relevant topics
common_words = stopwords.words('english')
common_words.extend(['of', 'and', 'the', 'in', 'were', 'to', 'nan', 'with'])
def remove_words(text):
  return [word for word in text if word not in common_words]
  
# stemming helps reduce some of the noise too. It chops some words, but we still
# know what the word says
stemmer = PorterStemmer()
def stem_words(text):
  text = [stemmer.stem(word) for word in text]
  return text
     
# this function applies the stemmer, removal of common words, removal of punctuation,
# lowercases the text, and tokenizes everything
def preprocess(text):
  return stem_words(remove_words(clean(text)))

# token the data to be used with gensim, just looking at the abstracts
new_data['tokenized_data'] = new_data['abstract'].apply(preprocess)

# print the first two rows of the cleaned data
# this will show the tokenized data from the abstract, basically splits up
# the abstract into single words
print(new_data.head(2))

# (3) CREATE GENSIM LIBRARY AND CORPUS
# gensim dictionary from tokenized data
token = new_data['tokenized_data']

# dictionary will be used in corpus
dictionary = corpora.Dictionary(token)

# filter keywords, we want the ones that show up the most in abstracts
dictionary.filter_extremes(no_below = 1, no_above = 0.8)         # filter keywords

# dictionary to corpus
corpus = [dictionary.doc2bow(tokens) for tokens in token]

# prints the corpus for the first document
# corpus (1, 1) implies that the word with the id of 1 has occurred only once in the first document
# corpus (14, 4) implies that the word with the id of 14 has occurred 4 times in the first document
print(corpus[:1])

# (4) BUILD THE TOPIC MODEL
# output shows the Topic-Words matrix for 5 of the topic that were created and 5 words within
# each topic that describes them
ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics = 5, id2word = dictionary, passes = 10)
# ldamodel = gensim.models.ldamulticore.LdaModel(corpus, num_topics=5, id2word=dictionary, passes=10)
ldamodel.save('model.gensim')
topics = ldamodel.print_topics(num_words = 5)
for topic in topics: 
  print(topic)
  
# Document-Topic matrix to find the probability of the topics
# get_document_topics is a function in lda
doc_topic = ldamodel.get_document_topics(corpus[1])

# prints the topic proportions
# (3, 0.744) implies that topic 3 showed up in 74.4% of the abstracts
print(doc_topic)

top_topics = ldamodel.top_topics(corpus)
# average topic coherence is the sum of topic coherences of all topics, divided by the number of topics
avg_topic_coherence = sum([t[1] for t in top_topics]) / len(topics)
print('Average topic coherence: %.4f.' % avg_topic_coherence)

from pprint import pprint
pprint(top_topics)

# (5) ANALYZE THE DATA
import pyLDAvis

# output the visuals to lda-vis-data.html, an interactive html file
lda_vis_data = pyLDAvis.gensim.prepare(ldamodel, corpus, dictionary)
pyLDAvis.save_html(lda_vis_data, "lda-vis-data.html")

# determin when the algorithm stopped
end = time.time()
print(f"Runtime: {end - start}")

# show the results
pyLDAvis.show(lda_vis_data)
