# Code referenced from:
# https://radimrehurek.com/gensim/auto_examples/tutorials/run_lda.html#sphx-glr-auto-examples-tutorials-run-lda-py
# https://datascienceplus.com/evaluation-of-topic-modeling-topic-coherence/

import numpy as np
import pyLDAvis.gensim
import pandas as pd
from gensim.models.coherencemodel import CoherenceModel
from gensim.models.ldamodel import LdaModel
from gensim.models import Phrases
from gensim.corpora.dictionary import Dictionary
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
import matplotlib.pyplot as plt


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


def compute_coherence_values(dictionary, corpus, texts, limit, start=2, step=3):
    """
    Compute c_v coherence for various number of topics

    Parameters:
    ----------
    dictionary : Gensim dictionary
    corpus : Gensim corpus
    texts : List of input texts
    limit : Max num of topics

    Returns:
    -------
    model_list : List of LDA topic models
    coherence_values : Coherence values corresponding to the LDA model with respective number of topics
    """
    coherence_values = []
    model_list = []
    for num_topics in range(start, limit, step):
        model = LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics)
        model_list.append(model)
        coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
        coherence_values.append(coherencemodel.get_coherence())

    return model_list, coherence_values


if __name__ == "__main__":
    # Setup
    print("Stage 1: Setup")

    # Import dataset
    data = pd.read_csv("metadata.csv", low_memory=False)
    keep_columns = ['abstract', 'publish_time']
    new_data = data[keep_columns]
    new_data = new_data.dropna(axis='index')
    new_data.to_csv("newdata-pp3.csv", index=False)  # Write out csv with only abstract and publish_time columns
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
    passes = 10  # 20

    # Make a index to word dictionary.
    temp = dictionary[0]  # only to "load" the dictionary.
    id2word = dictionary.id2token
    lda_model = LdaModel(corpus=corpus, num_topics=num_topics, id2word=id2word, passes=passes)
    # Print the Keyword in the 5 topics
    print(lda_model.print_topics())

    # Calculate Coherence and saving pyLDAvis
    print("Stage 3: Saving pyLDAvis")

    lda_vis_data = pyLDAvis.gensim.prepare(lda_model, corpus, dictionary)
    pyLDAvis.save_html(lda_vis_data, "ldavis-pp3.html")

    # Finding the optimal number of topics
    print("Stage 4: Finding the optimal number of topics")

    # Show graph
    limit = 10  #40
    start = 2
    step = 5  #6
    model_list, coherence_values = compute_coherence_values(dictionary=dictionary, corpus=corpus, texts=docs,
                                                            start=start, limit=limit, step=step)
    x = range(start, limit, step)
    plt.plot(x, coherence_values)
    plt.xlabel("Num Topics")
    plt.ylabel("Coherence score")
    plt.legend("coherence_values", loc='best')
    fname = "lda-pp3.png"
    plt.savefig(fname)
