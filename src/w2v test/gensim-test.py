import gensim
import numpy as np 
import pandas as pd
import re 
import string 
import nltk

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
def clean(txt):
    txt = re.sub("[^a-zA-Z ]", "", str(txt))
    txt = txt.lower()
    txt = nltk.word_tokenize(txt)
    return txt

token_data = data['abstract'].apply(clean)

#Using gensim to generate model below
#I'm unsure of whether using the token_data as the corpus
#is the most efficient approach without any windowing or size
model = gensim.models.Word2Vec(token_data, size = 100)

#Save for future use
model.save("w2v.model")

#Below section is just to test that we've read in correctly
five_word = [x for x in model.wv.vocab][0:5]
print(five_word)




