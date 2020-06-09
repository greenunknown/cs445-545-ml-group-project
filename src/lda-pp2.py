# implementing LDA with gensim
# 1. import dataset
# 2. preprocess text
# 3. create gensim dictionary and corpus
# 4. build the topic model
# 5. analyze

import numpy as np
import pandas as pd
import re
import gensim
import pyLDAvis.gensim
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import RegexpTokenizer
from gensim import corpora, models, similarities

from nltk.stem.wordnet import WordNetLemmatizer
from gensim.models import Phrases
from pprint import pprint
import pyLDAvis

import time


# Functions

def clean(text):
    # Split the documents into tokens.
    tokenizer = RegexpTokenizer(r'\w+')

    # remove punctuation
    text = re.sub("[^a-zA-Z ]", "", str(text))

    text = text.lower()  # Convert to lowercase.
    text = tokenizer.tokenize(text)  # Split into words.
    # # Remove numbers, but not words that contain numbers.
    # text = [[token for token in doc if not token.isnumeric()] for doc in text]
    #
    # # Remove words that are only one character.
    # text = [[token for token in doc if len(token) > 1] for doc in text]

    # lemmatizer = WordNetLemmatizer()
    # text = [[lemmatizer.lemmatize(token) for token in doc] for doc in text]

    # text = nltk.word_tokenize(text)
    return text


def remove_words(text):
    common_words = stopwords.words('english')
    common_words.extend(['of', 'and', 'the', 'in', 'were', 'to', 'nan', 'with'])
    return [word for word in text if word not in common_words]


def preprocess(text):
    return remove_words(clean(text))


def print_top_topics(corpus, len_topics, ldamodel):
    top_topics = ldamodel.top_topics(corpus)
    # average topic coherence is the sum of topic coherences of all topics, divided by the number of topics
    avg_topic_coherence = sum([t[1] for t in top_topics]) / len(topics)
    print('Average topic coherence: %.4f.' % avg_topic_coherence)
    pprint(top_topics)



start = time.time()

nltk.download('all')

# nltk.download('punkt')


# (1) IMPORT DATASET
data = pd.read_csv('metadata.csv', low_memory=False)
keep_columns = ['abstract', 'publish_time']

# new_data contains the data in metadata.csv, but it only keeps the abstract and publish time
new_data = data[keep_columns]

# new_data is stored in a new csv file, this is the file that we'll be working with
new_data.to_csv("newdata.csv", index=False)
file = 'newdata.csv'

# (2) PREPROCESS TEXT
# token the data to be used with gensim, just looking at the abstracts
new_data['tokenized_data'] = new_data['abstract'].apply(preprocess)

# (3) CREATE GENSIM LIBRARY AND CORPUS
token = new_data['tokenized_data']  # gensim dictionary from tokenized data


dictionary = corpora.Dictionary(token)  # dictionary used in corpus

# Filter out words that occur in less than 5 documents or more than 50% of the documents
# dictionary.filter_extremes(no_below=1, no_above=0.8)         # filter keywords
dictionary.filter_extremes(no_below=5, no_above=0.5)

# dictionary to corpus
corpus = [dictionary.doc2bow(tokens) for tokens in token]

print('Number of unique tokens: %d' % len(dictionary))
print('Number of documents: %d' % len(corpus))

# (4) BUILD THE TOPIC MODEL
# output shows the Topic-Words matrix for 5 of the topic that were created and 5 words within
# each topic that describes them
# ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics = 5, id2word = dictionary, passes = 10)
ldamodel = gensim.models.ldamulticore.LdaModel(corpus, num_topics=5, id2word=dictionary, passes=10)
ldamodel.save('model.gensim')
topics = ldamodel.print_topics(num_words=5)
for topic in topics:
    print(topic)

# (5) ANALYZE THE DATA
print_top_topics(corpus, len(topics), ldamodel)
lda_vis_data = pyLDAvis.gensim.prepare(ldamodel, corpus, dictionary)
pyLDAvis.save_html(lda_vis_data, "lda-vis-data.html")
end = time.time()
print(f"Runtime: {end - start}")

# pyLDAvis.show(lda_vis_data)
