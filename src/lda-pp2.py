# Code referenced from:
# https://radimrehurek.com/gensim/auto_examples/tutorials/run_lda.html#sphx-glr-auto-examples-tutorials-run-lda-py
# https://datascienceplus.com/evaluation-of-topic-modeling-topic-coherence/

import numpy as np
import pyLDAvis.gensim
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
from gensim.models.coherencemodel import CoherenceModel
from gensim.models.ldamodel import LdaModel
from gensim.models import Phrases
from gensim.corpora.dictionary import Dictionary
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer


def docs_preprocessor(documents):
    """
    Tokenize and lemmatizing documents
    :param documents:
    :return:
    """
    tokenizer = RegexpTokenizer(r'\w+')
    for i in range(len(documents)):
        documents[i] = documents[i].lower()  # Convert to lowercase.
        documents[i] = tokenizer.tokenize(documents[i])  # Split into words.

    # Remove numbers, but not words that contain numbers.
    documents = [[tkn for tkn in doc if not tkn.isdigit()] for doc in documents]

    # Remove words that are only one character.
    documents = [[tkn for tkn in doc if len(tkn) > 3] for doc in documents]

    # Lemmatize all words in documents.
    lemmatizer = WordNetLemmatizer()
    documents = [[lemmatizer.lemmatize(tkn) for tkn in doc] for doc in documents]
    return documents


if __name__ == "__main__":
    # Setup
    print("Stage 1: Setup")

    # Import dataset
    data = pd.read_csv("metadata.csv", low_memory=False)
    keep_columns = ['abstract']
    new_data = data[keep_columns]
    new_data = new_data.dropna(axis='index')
    new_data.to_csv("newdata-pp2.csv", index=False)  # Write out csv with only abstract columns
    docs = np.array(new_data['abstract'])  # Convert to array

    # Perform function on our document
    print("Performing preprocessing.")
    docs = docs_preprocessor(docs)

    # Create Biagram & Trigram Models
    # Add bigrams and trigrams to docs,minimum count 10 means only that appear 10 times or more.
    bigram = Phrases(docs, min_count=10)
    trigram = Phrases(bigram[docs])

    for idx in range(len(docs)):
        for token in bigram[docs[idx]]:
            if '_' in token:
                # Token is a bigram, add to document.
                docs[idx].append(token)
        for token in trigram[docs[idx]]:
            if '_' in token:
                # Token is a bigram, add to document.
                docs[idx].append(token)

    # Remove rare & common tokens
    # Create a dictionary representation of the documents.
    dictionary = Dictionary(docs)
    dictionary.filter_extremes(no_below=10, no_above=0.2)
    # Create dictionary and corpus required for Topic Modeling
    corpus = [dictionary.doc2bow(doc) for doc in docs]
    print('Number of unique tokens: %d' % len(dictionary))
    print('Number of documents: %d' % len(corpus))
    print(corpus[:1])

    # Train the model
    print("Stage 2: Train the model.")
    # Set parameters.
    num_topics = 5
    chunksize = 500
    passes = 10  # 20
    iterations = 100  # 400

    # Make a index to word dictionary.
    temp = dictionary[0]  # only to "load" the dictionary.
    id2word = dictionary.id2token
    lda_model = LdaModel(corpus=corpus, num_topics=num_topics, id2word=id2word, passes=passes, chunksize=chunksize,
                         alpha='auto', eta='auto')

    # Calculate Coherence and saving pyLDAvis
    print("Stage 3: Calculating coherence and saving pyLDAvis")

    # Compute Coherence Score using c_v
    coherence_model_lda_c_v = CoherenceModel(model=lda_model, texts=docs, dictionary=dictionary, coherence='c_v')
    coherence_lda_c_v = coherence_model_lda_c_v.get_coherence()
    print('\nCoherence Score c_v: ', coherence_lda_c_v)

    # Compute Coherence Score using UMass
    coherence_model_lda_u_mass = CoherenceModel(model=lda_model, texts=docs, dictionary=dictionary, coherence="u_mass")
    coherence_lda_u_mass = coherence_model_lda_u_mass.get_coherence()
    print('\nCoherence Score u_mass: ', coherence_lda_u_mass)

    lda_vis_data = pyLDAvis.gensim.prepare(lda_model, corpus, dictionary)
    pyLDAvis.save_html(lda_vis_data, "ldavis-pp2.html")
