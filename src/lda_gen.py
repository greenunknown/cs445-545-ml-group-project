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

# (1) IMPORT DATASET
data = pd.read_csv('metadata.csv', low_memory = False)
keep_columns = ['abstract', 'publish_time']
new_data = data[keep_columns]
new_data.to_csv("newdata.csv", index = False)
file = 'newdata.csv'
print(new_data.head(5))

# (2) PREPROCESS TEXT
def clean(text):
  # remove punctuation 
  text = re.sub("[^a-zA-Z ]", "", text)
  
  # lowercase everything
  text = text.lower()
  
  # tokenize the text
  text = nltk.word_tokenize(text)
  return text
  
stemmer = PorterStemmer()
def stem_words(text):
  text = [stemmer.stem(word) for word in text]
  return text
     
def preprocess(text):
  return stem_words(clean(text))

# token the data to be used with gensim, just looking at the abstracts
data['tokenized_data'] = data['abstract'].apply(preprocess)    

# (3) CREATE GENSIM LIBRARY AND CORPUS
# gensim dictionary from tokenized data
token = data['tokenized_data']

# dictionary will be used in corpus
dictionary = corpora.Dictionary(token)

# filter keywords, we want the ones that show up the most in abstracts
dictionary.filter_extremes(no_below = 1, no_above = 0.8)         # filter keywords

# dictionary to corpus
corpus = [dictionary.doc2bow(tokens) for tokens in tokenized]

# (4) BUILD THE TOPIC MODEL
ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics = 5, id2word = dictionary, passes = 10)
ldamodel.save('model.gensim')
topics = ldamodel.print_topics(num_words = 5)
for topic in topics: 
  print(topic)