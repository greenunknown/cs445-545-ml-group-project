import gensim
import numpy as np 
import pandas as pd
import re 
import string 
import nltk
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

#nltk.download('punkt') 

#Using the same method as Angie's LDA to filter and preprocess 
#the csv into corpus.  Credit to Angie McGraw

#NOTICE: UNCOMMENT THE SECTION BELOW ON FIRST RUN

"""
metadata = pd.read_csv('metadata.csv', low_memory= False)
keep_columns = ['abstract', 'publish_time']

data  = metadata[keep_columns]

data.to_csv("newmeta.csv", index = False)

"""

data = pd.read_csv('newmeta.csv', low_memory= False)

#Remove punctuation and lower case all characters
#No removal of numbers
def clean(txt):
    txt = re.sub("[^a-zA-Z ]", "", str(txt))
    txt = txt.lower()
    txt = nltk.word_tokenize(txt)
    return txt

token_data = data['abstract'].apply(clean)

#Using gensim to generate model below
model = gensim.models.Word2Vec(token_data, size = 100, window= 20, min_count= 15000, workers= 4)
#Save for future use
model.save("w2v.model")

#Trying to get the word label
vocab = list(model.wv.vocab)
X = model[vocab]

tsne = TSNE(n_components= 2)
X_tsne = tsne.fit_transform(X)

#Loading into pandas Dataframe
df = pd.DataFrame(X_tsne, index=vocab, columns=['x', 'y'])

#plotting 
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.scatter(df['x'], df['y'])

for word, pos in df.iterrows():
    ax.annotate(word, pos)
plt.show()

