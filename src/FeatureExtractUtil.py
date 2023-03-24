# https://medium.com/analytics-vidhya/topic-modeling-using-gensim-lda-in-python-48eaa2344920
import re
import numpy as np
import pandas as  pd
from pprint import pprint# Gensim
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel# spaCy for preprocessing
import spacy# Plotting tools
import pyLDAvis
import pyLDAvis.gensim
import matplotlib.pyplot as plt

from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import string

#------Preparing Documents
def loadData():
    file = open("C:\\Users\\ajuneja\\Documents\\Learnings\\AI\\TokenModelling\\Python\\src\\tableTennisReviews.txt", encoding="utf8")
    data = file.readlines()
    file.close()
    return data

#--------------Cleaning and Preprocessing
def sent_to_words(sentences):
  for sentence in sentences:
    yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))            #deacc=True removes punctuations
# Define function for stopwords, bigrams, trigrams and lemmatization
def remove_stopwords(stop_words, texts):
    return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]

def make_bigrams(bigram_mod, texts):
    return [bigram_mod[doc] for doc in texts]

def make_trigrams(trigram_mod, bigram_mod, texts):
    return [trigram_mod[bigram_mod[doc]] for doc in texts]

def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    """https://spacy.io/api/annotation"""
    # Initialize spacy 'en' model, keeping only tagger component (for efficiency)
    # python3 -m spacy download en
    nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])

    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent)) 
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out

def cleanData(data):
    stop_words = stopwords.words('english')
    stop_words.extend(['from', 'subject', 're', 'edu', 'use'])

    # Remove Emails 
    data = [re.sub('\S*@\S*\s?', '', sent) for sent in data]  
    # Remove new line characters 
    data = [re.sub('\s+', ' ', sent) for sent in data]  
    # Remove distracting single quotes 
    data = [re.sub("\'", "", sent) for sent in data]  
    data_words = list(sent_to_words(data))

    # Build the bigram and trigram models
    bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100) # higher threshold fewer phrases.
    trigram = gensim.models.Phrases(bigram[data_words], threshold=100)
    # Faster way to get a sentence clubbed as a trigram/bigram
    bigram_mod = gensim.models.phrases.Phraser(bigram)
    trigram_mod = gensim.models.phrases.Phraser(trigram)
    # See trigram example
    print(trigram_mod[bigram_mod[data_words[0]]])

    # Remove Stop Words
    data_words_nostops = remove_stopwords(stop_words=stop_words, texts=data_words)

    # Form Bigrams
    data_words_bigrams = make_bigrams(bigram_mod=bigram_mod, texts=data_words_nostops)

    # Do lemmatization keeping only noun, adj, vb, adv
    data_lemmatized = lemmatization(data_words_bigrams, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])

    print(data_lemmatized[:1])
    return data_lemmatized

#--------------
#-----------------------Preparing Document-Term Matrix
# Importing Gensim

# Creating the term dictionary of our courpus, where every unique term is assigned an index. 
# Create Dictionary 
def peformLDA(data):
    id2word = corpora.Dictionary(data)  
    # Create Corpus 
    texts = data  
    # Term Document Frequency 
    corpus = [id2word.doc2bow(text) for text in texts]  
    # View 
    print(corpus[:1])

    print(corpus)
    #---------------------
    #----------Running LDA Model
    # Creating the object for LDA model using gensim library
    lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                            id2word=id2word,
                                            num_topics=5, 
                                            random_state=100,
                                            update_every=1,
                                            chunksize=100,
                                            passes=10,
                                            alpha='auto',
                                            per_word_topics=True)
    pprint(lda_model.print_topics(num_topics=1, num_words=5))
    doc_lda = lda_model[corpus]

    # Compute Perplexity
    print('\nPerplexity: ', lda_model.log_perplexity(corpus))  
    return lda_model.print_topics()
# a measure of how good the model is. lower the better.

def buildChart(lda_model, corpus, id2word):
    # Compute Coherence Score
    #coherence_model_lda = CoherenceModel(model=lda_model, texts=data_lemmatized, dictionary=id2word, coherence='c_v')
    #coherence_lda = coherence_model_lda.get_coherence()
    #print('\nCoherence Score: ', coherence_lda)
    # pyLDAvis.enable_notebook()
    vis = pyLDAvis.gensim.prepare(lda_model, corpus, id2word)
    pyLDAvis.save_html(vis, 'LDA_Visualization.html')


def generateFeatureTopics(data=None):
    if data is None:
        data = loadData()
    print("--------------------")
    print(data)
    print("--------------------")
    data = cleanData(data=data)
    topics = peformLDA(data=data)
    return topics

#generateFeatureTopics()