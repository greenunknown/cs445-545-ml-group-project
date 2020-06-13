import gensim
import numpy as np 
import pandas as pd
import re 
import string 
import nltk
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

#Please uncomment this on first run to make sure you have the 
#necessary files for nltk
#nltk.download('punkt') 

#Credit to Angie McGraw for her preprocessing steps

##########################################################################
###########NOTICE: UNCOMMENT THE SECTION BELOW ON FIRST RUN###############
##########################################################################
"""
metadata = pd.read_csv('../data/metadata.csv', low_memory= False)
keep_columns = ['abstract', 'publish_time']

data  = metadata[keep_columns]

data.to_csv("../data/newmeta.csv", index = False)
"""

data = pd.read_csv('newmeta.csv', low_memory= False)

#Remove punctuation and lower case all characters
#No removal of numbers or stop words.
def clean(txt):
    txt = re.sub("[^a-zA-Z0-9 ]", "", str(txt))
    txt = txt.lower()
    txt = nltk.word_tokenize(txt)
    return txt

#Apply minor cleaning to the data
token_data = data['abstract'].apply(clean)

#Using gensim to generate model below
model = gensim.models.Word2Vec(token_data, size = 100, window= 20, min_count= 15000, workers= 4)

#Getting the word label from the vocab built by our model
vocab = list(model.wv.vocab)
X = model.wv.__getitem__(vocab)
tsne = TSNE(perplexity= 40, n_components= 2, init= 'pca', n_iter= 2500, random_state= 23)
X_tsne = tsne.fit_transform(X)

#Loading into pandas Dataframe
df = pd.DataFrame(X_tsne, index=vocab, columns=['x', 'y'])

#plotting from the pandas Dataframe
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.scatter(df['x'], df['y'])
for word, pos in df.iterrows():
    ax.annotate(word, pos)
plt.show()
