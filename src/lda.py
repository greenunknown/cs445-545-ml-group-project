# overview: lda tries to classify documents; documents are represented
# as a distribution of topics. We've got M number of documents, N number 
# of words, and K number of topics

# alpha parameter: represents document-topic density
# beta parameter: represents topic-word denstiy

# implementation: (1) data loading, (2) data cleaning, (3) exploratory 
# analysis, (4) data preparation for LDA analysis, (5) LDA model training

# using the Allen Institute data, we have 5 rows and 9 columns

import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import re
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer

# (1) DATA LOADING
input = pd.read_csv('metadata.csv', low_memory = False)

#####print(input.head())

# (2) DATA CLEANING
input = input.drop(columns = ['cord_uid', 'sha', 'source_x', 'pmcid', 'pubmed_id', 'license'], axis = 1)

#####print(input.head())

# remove punctuation
input['input_text_processed'] = input['input_text'].map(lambda x: re.sub('[,\.!?]', '', x))

# lowercase all text
input['input_text_processed'] = input['input_text_processed'].map(lambda x: x.lower())

#####print(input['input_text_processed'].head())

# (3) EXPLORATORY ANALYSIS
# it would be interesting to have a visual of all the key words in the abstract, so here's 
# the code for it

# put all the words in the abstracts together, that way it's easier for us to analyze
join_abstracts = ','.join(list(input['input_text_processed'].values))

wc = WordCloud().generate(join_abstracts)

plt.imshow(wc, interpolation = 'bilinear')
plt.axis("off")
plt.show()

# (4) DATA PREPARATION FOR LDA ANALYSIS
def plot_keywords(count, count_vectorizer): 
  keywords = count_vectorizer.get_feature_names()
  freq_of_keywords = np.zeros(len(keywords))
  for i in count: 
    freq_of_keywords += i.toarray()[0]

  count_keywords = (zip(keywords, freq_of_keywords))
  count_keywords = sorted(count_dict, key=lambda x:x[1], reverse=True)[0:10]
  keywords = [w[0] for w in count_keywords]
  counts = [w[1] for w in count_keywords]
  x = np.arange(len(keywords))

  # gotta add in plot code
