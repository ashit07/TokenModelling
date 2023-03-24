##https://towardsdatascience.com/end-to-end-topic-modeling-in-python-latent-dirichlet-allocation-lda-35ce4ed6b3e0

from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer

import gensim
from gensim import corpora
from gensim.utils import simple_preprocess

import string

#------Preparing Documents
doc1 = "Sugar is bad to consume. My sister likes to have sugar, but not my father."
doc2 = "My father spends a lot of time driving my sister around to dance practice."
doc3 = "Doctors suggest that driving may cause increased stress and blood pressure."
doc4 = "Sometimes I feel pressure to perform well at school, but my father never seems to drive my sister to do better."
doc5 = "Health experts say that Sugar is not good for your lifestyle."

# compile documents
#doc_complete = [doc1, doc2, doc3, doc4, doc5]

file = open("comments.txt", encoding="utf8")
doc_complete = file.readlines()

#print(doc_complete)
#--------------Cleaning and Preprocessing
stop = set(stopwords.words('english'))
#stop.extend(['from', 'subject', 're', 'edu', 'use'])

exclude = set(string.punctuation)
def sent_to_words(sentences):
    for sentence in sentences:
        # deacc=True removes punctuations
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))
def remove_stopwords(texts):
    return [[word for word in simple_preprocess(str(doc)) 
             if word not in stop] for doc in texts]

doc_clean = list(sent_to_words(doc_complete))
# remove stop words
doc_clean = remove_stopwords(doc_clean)
print(doc_clean[:1][0][:30])

#doc_clean = [clean(doc).split() for doc in doc_complete]

#print(doc_clean)
print("--------------------------")
#--------------
#-----------------------Preparing Document-Term Matrix
# Importing Gensim

# Creating the term dictionary of our courpus, where every unique term is assigned an index. 
dictionary = corpora.Dictionary(doc_clean)

# Converting list of documents (corpus) into Document Term Matrix using dictionary prepared above.
doc_term_matrix = [dictionary.doc2bow(doc) for doc in doc_clean]

#print(doc_term_matrix)
#---------------------
#----------Running LDA Model
# Creating the object for LDA model using gensim library
from pprint import pprint
# number of topics
num_topics = 10
# Build LDA model
lda_model = gensim.models.ldamodel.LdaModel(corpus=doc_term_matrix,
                                       id2word=dictionary,
                                       num_topics=num_topics)
# Print the Keyword in the 10 topics
pprint(lda_model.print_topics(num_words=5))
doc_lda = lda_model[doc_term_matrix]


####--------Generating html file
import pyLDAvis.gensim
import pickle 
import pyLDAvis
# Visualize the topics
pyLDAvis.enable_notebook()
LDAvis_data_filepath = os.path.join('./results/ldavis_prepared_'+str(num_topics))
# # this is a bit time consuming - make the if statement True
# # if you want to execute visualization prep yourself
if 1 == 1:
    LDAvis_prepared = pyLDAvis.gensim.prepare(lda_model, dictionary, doc_term_matrix)
    with open(LDAvis_data_filepath, 'wb') as f:
        pickle.dump(LDAvis_prepared, f)
# load the pre-prepared pyLDAvis data from disk
with open(LDAvis_data_filepath, 'rb') as f:
    LDAvis_prepared = pickle.load(f)
pyLDAvis.save_html(LDAvis_prepared, './results/ldavis_prepared_'+ str(num_topics) +'.html')