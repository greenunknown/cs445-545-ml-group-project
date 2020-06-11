# Only run this once
import nltk
nltk.download('all')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import os
import string
import logging
from pprint import pprint
from collections import defaultdict # for pos tag -> wordnet tag

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from nltk.corpus import wordnet
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer

import gensim
import pyLDAvis.gensim
from gensim.models import Phrases # For adding n-grams
from gensim import corpora, models, similarities

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


# Perform topic modelling with the help of nltk and gensim
#
class TopicModeler:
    # Initialize an instance of the TopicModeler class
    # params:
    #   corpus<numpy.ndarray([string])>: A numpy array of strings, representing documents in a corpus
    #   disp_interval<integer>: Display progess after each interval
    #
    def __init__(self, corpus, disp_interval=1000):
        self.corpus = np.copy(corpus)
        self.dictionary = None
        self.model = None
        self.disp_interval = disp_interval
        self.N = np.shape(self.corpus)[0]
        logging.info('Initialized a TopicModeler with corpus size N=' + str(self.N))
        return

    # Split the documents in the corpus into lists of raw tokens
    # params:
    #   disp_interval<integer>: Display progess after each interval
    #
    def tokenize(self, disp_interval=None):
        if disp_interval == None:
            disp_interval = self.disp_interval

        # Convert each document in the corpus into a list of tokens
        for document in range(self.N):
            if (document % disp_interval == 0):
                logging.info('Performing tokenization: [' + str(document) + '/' + str(self.N) + ']')
            self.corpus[document] = word_tokenize(self.corpus[document])
        return

    # Perform morphological normalization on the corpus as a list of raw tokens
    # params:
    #   norm<'lemma'|'stem'>: The form of morphological normalization to use
    #   disp_interval<integer>: Display progess after each interval
    #
    def normalize(self, norm='lemma', disp_interval=None):
        logging.info('Performing normalization.')
        logging.debug('norm=' + str(norm))
        if disp_interval == None:
            disp_interval = self.disp_interval

        if (norm == 'lemma'):
            lemmatizer = WordNetLemmatizer()

            # Maps nltk pos tags into wordnet pos tags
            tag_map = defaultdict(lambda: wordnet.NOUN)
            tag_map['J'] = wordnet.ADJ
            tag_map['V'] = wordnet.VERB
            tag_map['R'] = wordnet.ADV

            for document in range(self.N):
                if (document % disp_interval == 0):
                    logging.info('Performing lemmatization: [' + str(document) + '/' + str(self.N) + ']')
                self.corpus[document] = [lemmatizer.lemmatize(tagged_token[0], tag_map[tagged_token[1][0]])
                                         for tagged_token in pos_tag(self.corpus[document])]
        elif (norm == 'stem'):
            stemmer = PorterStemmer()

            for document in range(self.N):
                if (document % disp_interval == 0):
                    logging.info('Performing stemming: [' + str(document) + '/' + str(self.N) + ']')
                self.corpus[document] = [stemmer.stem(token) for token in self.corpus[document]]
        else:
            logging.warning('Invalid parameter! Skipping normalization...')
        return

    # Filter out any tokens that are not within a specified string length
    # params:
    #   min_strlen<integer>: The minimum amount of chars a token can have
    #   max_strlen<integer>: The maximum amount of chars a token can have
    #
    def filter_length(self, min_strlen=1, max_strlen=100):
        logging.info('Filtering out tokens that are out of the length bounds.')
        logging.debug('min_strlen=' + str(min_strlen))
        logging.debug('max_strlen=' + str(max_strlen))

        self.corpus = [[token for token in document
                        if len(token) >= min_strlen
                        and len(token) <= max_strlen
                        ] for document in self.corpus]
        return

    # Filter out any tokens that match a regular expression
    # params:
    #   pattern<raw string>: The regular expresstion to match
    #
    def filter_match(self, pattern):
        logging.info('Filtering out tokens that match the regular expression: ' + str(pattern))

        self.corpus = [[token for token in document
                        if re.search(pattern, token)
                        ] for document in self.corpus]
        return

    # Lowercase all tokens
    #
    def lowercase(self):
        logging.info('Converting all tokens to lowercase.')
        self.corpus = [[token.lower() for token in document] for document in self.corpus]
        return

    # Add n-grams to the corpus
    # params:
    #   n<integer(2|3)>: The maximum length of the sequence of words to add to the corpus
    #   min_count<integer>: The minimum amount of token occurances needed for an n-gram to be included
    #
    def add_n_grams(self, n=2, min_count=1):
        logging.info('Performing normalization.')
        logging.debug('n=' + str(n))
        logging.debug('min_count=' + str(min_count))

        logging.info('Adding 2-grams')
        bigram = Phrases(self.corpus, min_count=min_count, delimiter=b' ')

        if n == 3:
            logging.info('Adding 3-grams')
            trigram = Phrases(bigram[self.corpus], min_count=1, delimiter=b' ')
            for document in range(self.N):
                self.corpus[document] = [n_gram for n_gram in trigram[bigram[self.corpus[document]]]
                                         if n_gram.count(' ') < n]
        elif n == 2:
            for document in range(self.N):
                self.corpus[document] = [n_gram for n_gram in bigram[self.corpus[document]]
                                         if n_gram.count(' ') < n]
        else:
            logging.warning('Invalid parameter! Skipping n-grams...')
        return

    # Remove stop-words
    # params:
    #   stop<list([string])>: A list containing all stop words to exclude
    #
    def remove_stop_words(self, stop=stopwords.words('english')):
        logging.info('Removing stop-words and n-grams with stop-words.')
        logging.debug('stop_words=' + str(stop))

        # Filter out any token containing a stop-word
        self.corpus = [[token for token in document
                        if all(token_part not in stop
                               for token_part in token.split())
                        ] for document in self.corpus]
        return

    # Prepare the corpus for topic modelling
    #
    def preprocess(self):
        logging.info('Pipeline step 2: Preprocessing')
        self.tokenize()
        self.normalize()
        self.filter_length(min_strlen=3)
        self.filter_match(pattern=r'\w*?[a-zA-Z]\w*')
        self.lowercase()
        self.add_n_grams(n=3)
        self.remove_stop_words(stop=stopwords.words('english') + ['use', 'also'])
        return

    # Transform lists of pre-processed tokens into an id: word frequency representation
    # params:
    #   no_below<integer>: The minimum amount of documents a word must appear in to be considered
    #   no_above<float>: The maximum % of documents a word may appear in to be considered
    #
    def generate_dict(self, no_below=100, no_above=0.5):
        logging.info('Generating a dictionary of the corpus.')
        logging.debug('no_below=' + str(no_below))
        logging.debug('no_above=' + str(no_above))

        self.dictionary = corpora.Dictionary(self.corpus)

        # Filter out rare and common tokens
        self.dictionary.filter_extremes(no_below=no_below, no_above=no_above)
        return

    # Transforms lists of pre-processed tokens into a bag of words representation
    #
    def generate_bow(self):
        logging.info('Generating a Bag of Words representation of the corpus.')

        # Generate the bag of words
        self.corpus = [self.dictionary.doc2bow(documents) for documents in self.corpus]

        logging.debug('Number of unique tokens: ' + str(len(self.dictionary)))
        logging.debug('Number of documents: ' + str(len(self.corpus)))
        return

    # Save a frozen, trained topic model to disk
    # params:
    #   path<string>: The path to the save location for the model
    #
    def save_model(self, path='model.gensim'):
        logging.info('Saving the current model at: ' + str(path))
        self.model.save(path)
        return


# A class for running LDA to do topic modelling
#
class LDA(TopicModeler):
    # Initialize an instance of the LDA class
    # params:
    #   corpus<numpy.ndarray([string])>: A numpy array of strings, representing documents in a corpus
    #   num_topics<integer>: The amount of hidden topics in the corpus
    #   disp_interval<integer>: Display progess after each interval
    #
    def __init__(self, corpus, num_topics=5, disp_interval=1000):
        super().__init__(corpus, disp_interval)

        self.num_topics = num_topics
        logging.info('Initialized an LDA with num_topics=' + str(self.num_topics))
        return

    # A setter for the parameter representing the number of latent semantic topics to model
    # params:
    #   num_topics<integer>: The amount of hidden topics in the corpus
    #
    def set_num_topics(self, num_topics):
        logging.debug('Set num_topics=' + str(self.num_topics))
        self.num_topics = num_topics
        return

    # Train an LDA model on the corpus
    # params:
    #   batch_size<integer>: The amount of documents to be processed at a time
    #   epochs<integer>: The amount of complete passes through the dataset before completing training
    #   iterations<integer>: Maximum iterations on the corpus before inferring a topic distribution
    #   eval_every<boolean>: Evaluate the log perplexity of the model (2x hit to training time)
    #   eta<string|list>: The dirichlet prior for topic-word distributions
    #   alpha<string|list>: The dirichlet piror for document-topic distributions
    #
    def train(self, batch_size=1000, epochs=10, iterations=400, eval_every=None, eta='auto', alpha='auto'):
        logging.info('Training the LDA model.')
        logging.debug('batch_size=' + str(batch_size))
        logging.debug('epochs=' + str(epochs))
        logging.debug('iterations=' + str(iterations))
        logging.debug('eval_every=' + str(eval_every))

        self.model = gensim.models.ldamodel.LdaModel(
            corpus=self.corpus,
            id2word=self.dictionary,
            num_topics=self.num_topics,
            passes=epochs,
            iterations=iterations,
            chunksize=batch_size,
            alpha=alpha, eta=eta,
            eval_every=eval_every
        )
        for (i, z) in zip(range(self.num_topics), self.model.print_topics(num_words=self.num_topics)):
            logging.debug('Topic #' + str(i) + ": " + str(z))
        return

    # Retrieve set of topics for a document from a trained LDA model
    # params:
    #   document<integer>: The index of the document for which topics will be retrieved
    # returns: <type>: TODO
    #
    def get_document_topics(self, document=0):
        return self.model.get_document_topics(self.corpus[document])

    # Calculate the topic coherence for an LDA model
    # This is the sum of topic coherences of all topics, divided by the number of topics
    # params:
    #   num_topics<integer>: The amount of hidden topics in the corpus
    # returns: <float>: The average topic coherence for this model
    #
    def get_coherence(self, num_topics=5):
        top_topics = self.model.top_topics(self.corpus)
        avg_topic_coherence = sum([z[1] for z in top_topics]) / self.num_topics
        logging.info('Average topic coherence: ' + str(avg_topic_coherence))
        return avg_topic_coherence

    # Get the perplexity of an LDA model over the entire corpus
    # returns: <float>: The average topic coherence for this model
    #
    def get_perplexity(self):
        perplexity = self.model.log_perplexity(self.corpus)
        logging.info('Perplexity: ' + str(perplexity))
        return perplexity

    # Pretty print the top topics for this model
    #
    def print_topics(self):
        pprint(self.model.top_topics(self.corpus))
        return

    # Generate an HTML page to visualize the top topic distrubutions as 2D vectors
    # params:
    #   path<string>: The the path to the location where this HTML page should be saved
    #   save<boolean>: Save the HTML page or no
    #
    def generate_visual_LDA(self, path='lda-vis-data.html', save=False):
        logging.info('Generating an HTML page to display the LDA topic distributions...')
        lda_vis_data = pyLDAvis.gensim.prepare(self.model, self.corpus, self.dictionary)
        if (save):
            logging.info('Saving the LDA visualization at: ' + str(path))
            pyLDAvis.save_html(lda_vis_data, path)
        # pyLDAvis.show(lda_vis_data)
        return

# (1) IMPORT DATASET
# Dataset is from https://www.kaggle.com/allen-institute-for-ai/CORD-19-research-challenge?select=metadata.csv
data = pd.read_csv('metadata.csv', low_memory = False)
keep_columns = ['publish_time', 'journal', 'abstract']

# Select the relevant columns and rows from the dataset
new_data = data[keep_columns].dropna(subset=['abstract']).to_numpy()

## no_below and no_above
## Within what bounds of frequency should a candidate keyword appear in order to be considered?
repetitions = 5
corpus_size = 10000
num_topics = 5

## The minimum amount of documents a word must appear in to be considered
candidate_no_below = [1, 10, 50, 100]
avg_coherence = []
perplexity = []

for nb in candidate_no_below:
    logging.info('no_below=' + str(nb))
    cur_coherence = 0
    cur_perplexity = 0
    # (2) PREPROCESS TEXT
    lda_model = LDA(corpus=new_data[:corpus_size, -1], disp_interval=1000)
    lda_model.set_num_topics(num_topics)
    lda_model.tokenize()
    lda_model.normalize()
    lda_model.filter_length(min_strlen=3)
    lda_model.filter_match(pattern=r'\w*?[a-zA-Z]\w*')
    lda_model.lowercase()
    lda_model.add_n_grams(n=3)
    lda_model.remove_stop_words(stop=stopwords.words('english') + ['use', 'also'])

    # (3) PROCESS TEXT
    # (3.1) CREATE THE DICTIONARY AND BAG-OF-WORDS REPRESENTATION OF THE CORPUS
    lda_model.generate_dict(no_below=nb)
    lda_model.generate_bow()

    for r in range(repetitions):
        # (3.2) BUILD THE TOPIC MODEL
        lda_model.train()
        lda_model.get_document_topics()

        cur_coherence += lda_model.get_coherence() / repetitions
        cur_perplexity += lda_model.get_perplexity() / repetitions

    # (3.3) Log the avg coherence and perplexity
    avg_coherence.append(cur_coherence)
    perplexity.append(cur_perplexity)

# (4) Visualize the results
## Avg. Coherence
plt.clf()
plt.title("Avg. Coherence for LDA w/ Varying no_below")
plt.xlabel("no_below")
plt.ylabel("Coherence")
plt.plot([str(cand) for cand in candidate_no_below], avg_coherence, color='navy', linewidth=2, label="LDA")
plt.legend()
if not os.path.exists('../bin/LDA/'):
    os.makedirs('../bin/LDA/')
plt.savefig('../bin/LDA/no_below-coherence.png')
# plt.show()

## Perplexity
plt.clf()
plt.title("Perplexity for LDA w/ Varying no_below")
plt.xlabel("no_below")
plt.ylabel("Perplexity")
plt.plot([str(cand) for cand in candidate_no_below], perplexity, color='firebrick', linewidth=2, label="LDA")
plt.legend()
plt.savefig('../bin/LDA/no_below-perplexity.png')
# plt.show()

## The maximum % of documents a word may appear in to be considered
candidate_no_above = [.5, .6, .7, .8]
avg_coherence = []
perplexity = []

for na in candidate_no_above:
    logging.info('no_above=' + str(na))

    # (2) PREPROCESS TEXT
    lda_model = LDA(corpus=new_data[:corpus_size, -1], disp_interval=1000)
    lda_model.set_num_topics(num_topics)
    lda_model.tokenize()
    lda_model.normalize()
    lda_model.filter_length(min_strlen=3)
    lda_model.filter_match(pattern=r'\w*?[a-zA-Z]\w*')
    lda_model.lowercase()
    lda_model.add_n_grams(n=3)
    lda_model.remove_stop_words(stop=stopwords.words('english') + ['use', 'also'])

    # (3) PROCESS TEXT
    # (3.1) CREATE THE DICTIONARY AND BAG-OF-WORDS REPRESENTATION OF THE CORPUS
    lda_model.generate_dict(no_above=na)
    lda_model.generate_bow()

    for r in range(repetitions):
        # (3.2) BUILD THE TOPIC MODEL
        lda_model.train()
        lda_model.get_document_topics()

        cur_coherence += lda_model.get_coherence() / repetitions
        cur_perplexity += lda_model.get_perplexity() / repetitions

    # (3.3) Log the avg coherence and perplexity
    avg_coherence.append(cur_coherence)
    perplexity.append(cur_perplexity)

# (4) Visualize the results
## Avg. Coherence
plt.clf()
plt.title("Avg. Coherence for LDA w/ Varying no_above")
plt.xlabel("no_above")
plt.ylabel("Coherence")
plt.plot([str(cand) for cand in candidate_no_above], avg_coherence, color='navy', linewidth=2, label="LDA")
plt.legend()
if not os.path.exists('../bin/LDA/'):
    os.makedirs('../bin/LDA/')
plt.savefig('../bin/LDA/no_above-coherence.png')
# plt.show()

## Perplexity
plt.clf()
plt.title("Perplexity for LDA w/ Varying no_above")
plt.xlabel("no_above")
plt.ylabel("Perplexity")
plt.plot([str(cand) for cand in candidate_no_above], perplexity, color='firebrick', linewidth=2, label="LDA")
plt.legend()
plt.savefig('../bin/LDA/no_above-perplexity.png')
# plt.show()

## Stemming or Lemmatization? Or no morphological normalization?
## Stemming - chop off suffixes; lemmatization - get the dictionary form of the word
repetitions = 5

corpus_size = 10000
num_topics = 5

normalization = ['none', 'stem', 'lemma']
avg_coherence = []
perplexity = []

# Baseline - no morphological normalization
logging.info('norm=None')
cur_coherence = 0
cur_perplexity = 0
# (2) PREPROCESS TEXT
lda_model = LDA(corpus=new_data[:corpus_size, -1], disp_interval=1000)
lda_model.set_num_topics(num_topics)
lda_model.tokenize()
# lda_model.normalize()
lda_model.filter_length(min_strlen=3)
lda_model.filter_match(pattern=r'\w*?[a-zA-Z]\w*')
lda_model.lowercase()
lda_model.add_n_grams(n=3)
lda_model.remove_stop_words(stop=stopwords.words('english') + ['use', 'also'])

# (3) PROCESS TEXT
# (3.1) CREATE THE DICTIONARY AND BAG-OF-WORDS REPRESENTATION OF THE CORPUS
lda_model.generate_dict()
lda_model.generate_bow()

for r in range(repetitions):
    # (3.2) BUILD THE TOPIC MODEL
    lda_model.train()
    lda_model.get_document_topics()

    cur_coherence += lda_model.get_coherence() / repetitions
    cur_perplexity += lda_model.get_perplexity() / repetitions

# (3.3) Log the avg coherence and perplexity
avg_coherence.append(cur_coherence)
perplexity.append(cur_perplexity)

# Stemming
logging.info('norm=stem')
cur_coherence = 0
cur_perplexity = 0
# (2) PREPROCESS TEXT
lda_model = LDA(corpus=new_data[:corpus_size, -1], disp_interval=1000)
lda_model.set_num_topics(num_topics)
lda_model.tokenize()
lda_model.normalize(norm='stem')
lda_model.filter_length(min_strlen=3)
lda_model.filter_match(pattern=r'\w*?[a-zA-Z]\w*')
lda_model.lowercase()
lda_model.add_n_grams(n=3)
lda_model.remove_stop_words(stop=stopwords.words('english') + ['use', 'also'])

# (3) PROCESS TEXT
# (3.1) CREATE THE DICTIONARY AND BAG-OF-WORDS REPRESENTATION OF THE CORPUS
lda_model.generate_dict()
lda_model.generate_bow()

for r in range(repetitions):
    # (3.2) BUILD THE TOPIC MODEL
    lda_model.train()
    lda_model.get_document_topics()

    cur_coherence += lda_model.get_coherence() / repetitions
    cur_perplexity += lda_model.get_perplexity() / repetitions

# (3.3) Log the avg coherence and perplexity
avg_coherence.append(cur_coherence)
perplexity.append(cur_perplexity)

# Lemmatization
logging.info('norm=lemma')
cur_coherence = 0
cur_perplexity = 0
# (2) PREPROCESS TEXT
lda_model = LDA(corpus=new_data[:corpus_size, -1], disp_interval=1000)
lda_model.set_num_topics(num_topics)
lda_model.tokenize()
lda_model.normalize(norm='lemma')
lda_model.filter_length(min_strlen=3)
lda_model.filter_match(pattern=r'\w*?[a-zA-Z]\w*')
lda_model.lowercase()
lda_model.add_n_grams(n=3)
lda_model.remove_stop_words(stop=stopwords.words('english') + ['use', 'also'])

# (3) PROCESS TEXT
# (3.1) CREATE THE DICTIONARY AND BAG-OF-WORDS REPRESENTATION OF THE CORPUS
lda_model.generate_dict()
lda_model.generate_bow()

for r in range(repetitions):
    # (3.2) BUILD THE TOPIC MODEL
    lda_model.train()
    lda_model.get_document_topics()

    cur_coherence += lda_model.get_coherence() / repetitions
    cur_perplexity += lda_model.get_perplexity() / repetitions

# (3.3) Log the avg coherence and perplexity
avg_coherence.append(cur_coherence)
perplexity.append(cur_perplexity)

# (4) Visualize the results
## Avg. Coherence
plt.clf()
plt.title("Avg. Coherence for LDA w/ Various Morphological Normalization")
plt.xlabel("Morphological Normalization")
plt.ylabel("Coherence")
plt.plot(normalization, avg_coherence, color='navy', linewidth=2, label="LDA")
plt.legend()
if not os.path.exists('../bin/LDA/'):
    os.makedirs('../bin/LDA/')
plt.savefig('../bin/LDA/normalization-coherence.png')
# plt.show()

## Perplexity
plt.clf()
plt.title("Perplexity for LDA w/ Various Morphological Normalization")
plt.xlabel("Morphological Normalization")
plt.ylabel("Perplexity")
plt.plot(normalization, perplexity, color='firebrick', linewidth=2, label="LDA")
plt.legend()
plt.savefig('../bin/LDA/normalization-perplexity.png')
# plt.show()

# (2) PREPROCESS TEXT
corpus_size = 10000
lda_model = LDA(corpus=new_data[:corpus_size, -1], disp_interval=1000)
lda_model.preprocess()

# (3) PROCESS TEXT
# (3.1) CREATE THE DICTIONARY AND BAG-OF-WORDS REPRESENTATION OF THE CORPUS
lda_model.generate_dict()
lda_model.generate_bow()

## Optimizing num_topics
repetitions = 5

candidate_num_topics = [5, 10, 15, 20, 25, 30]
avg_coherence = []
perplexity = []

for k in candidate_num_topics:
    logging.info('num_topics=' + str(num_topics))
    cur_coherence = 0
    cur_perplexity = 0
    lda_model.set_num_topics(k)

    # Get the average coherence and perplexity over a number of repetitions
    for r in range(repetitions):
        # (3.2) BUILD THE TOPIC MODEL
        lda_model.train()
        lda_model.get_document_topics()

        cur_coherence += lda_model.get_coherence() / repetitions
        cur_perplexity += lda_model.get_perplexity() / repetitions

    # (3.3) Log the avg coherence and perplexity
    avg_coherence.append(cur_coherence)
    perplexity.append(cur_perplexity)

# (4) Visualize the results
## Avg. Coherence
plt.clf()
plt.title("Avg. Coherence for LDA w/ Varying num_topics")
plt.xlabel("num_topics")
plt.ylabel("Coherence")
plt.plot(candidate_num_topics, avg_coherence, color='navy', linewidth=2, label="LDA")
plt.legend()
if not os.path.exists('../bin/LDA/'):
    os.makedirs('../bin/LDA/')
plt.savefig('../bin/LDA/num_topics-coherence.png')
# plt.show()

## Perplexity
plt.clf()
plt.title("Perplexity for LDA w/ Varying num_topics")
plt.xlabel("num_topics")
plt.ylabel("Perplexity")
plt.plot(candidate_num_topics, perplexity, color='firebrick', linewidth=2, label="LDA")
plt.legend()
plt.savefig('../bin/LDA/num_topics-perplexity.png')
# plt.show()

## Optimizing eta (the dirichlet prior for word-topic distributions)
# Expectation: smaller eta == more focused topics; larger == more sparse topics
repetitions = 5

num_topics = 5
lda_model.set_num_topics(num_topics)

candidate_etas = [0.01, 0.1, 1, 10, 'auto']
avg_coherence = []
perplexity = []

for eta in candidate_etas:
    logging.info('candidate_etas=' + str(candidate_etas))
    cur_coherence = 0
    cur_perplexity = 0

    # Get the average coherence and perplexity over a number of repetitions
    for r in range(repetitions):
        # (3.2) BUILD THE TOPIC MODEL
        lda_model.train(eta=eta)
        lda_model.get_document_topics()

        cur_coherence += lda_model.get_coherence() / repetitions
        cur_perplexity += lda_model.get_perplexity() / repetitions

    # (3.3) Log the avg coherence and perplexity
    avg_coherence.append(cur_coherence)
    perplexity.append(cur_perplexity)

# (4) Visualize the results
## Avg. Coherence
plt.clf()
plt.title("Avg. Coherence for LDA w/ Varying eta")
plt.xlabel("eta")
plt.ylabel("Coherence")
plt.plot([str(eta) for eta in candidate_etas], avg_coherence, color='navy', linewidth=2, label="LDA")
plt.legend()
if not os.path.exists('../bin/LDA/'):
    os.makedirs('../bin/LDA/')
plt.savefig('../bin/LDA/eta-coherence.png')
# plt.show()

## Perplexity
plt.clf()
plt.title("Perplexity for LDA w/ Varying eta")
plt.xlabel("eta")
plt.ylabel("Perplexity")
plt.plot([str(eta) for eta in candidate_etas], perplexity, color='firebrick', linewidth=2, label="LDA")
plt.legend()
plt.savefig('../bin/LDA/eta-perplexity.png')
# plt.show()

## ALL ABSTRACTS
# (1) IMPORT DATASET
# Dataset is from https://www.kaggle.com/allen-institute-for-ai/CORD-19-research-challenge?select=metadata.csv
data = pd.read_csv('metadata.csv', low_memory = False)
keep_columns = ['publish_time', 'journal', 'abstract']

# Select the relevant columns and rows from the dataset
new_data = data[keep_columns].dropna(subset=['abstract']).to_numpy()

# (2) PREPROCESS TEXT
lda_model = LDA(corpus=new_data[:, -1], num_topics=5)
lda_model.preprocess()

# (3) PROCESS TEXT
# (3.1) CREATE THE DICTIONARY AND BAG-OF-WORDS REPRESENTATION OF THE CORPUS
lda_model.generate_dict()
lda_model.generate_bow()
if not os.path.exists('../bin/LDA/'):
    os.makedirs('../bin/LDA/')

# (3.2) BUILD THE TOPIC MODEL
lda_model.train()
lda_model.save_model(path='../bin/LDA/all_model.gensim')
lda_model.get_document_topics()
lda_model.get_coherence()
lda_model.get_perplexity()

# (4) ANALYZE THE DATA
lda_model.print_topics()
lda_model.generate_visual_LDA(path='../bin/LDA/all_lda-vis-data.html', save=True)
