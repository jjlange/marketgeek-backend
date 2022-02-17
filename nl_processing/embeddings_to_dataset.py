import pandas as pd
import nltk
import re
import io
import gensim
import numpy as np


from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from gensim.parsing.preprocessing import remove_stopwords
from sentence_transformers import SentenceTransformer
from numpy import asarray
from numpy import savetxt

# nltk.download('wordnet')
# nltk.download('omw-1.4')
# nltk.download('stopwords')
# nltk.download('punkt')



# import embeddings from csv

embeddings_csv = pd.read_csv('embeddings.csv')

# savetxt('embeddings.csv', sentence_embeddings, delimiter=',')


# convert to list
# print(sentence_embeddings[1])
print(embeddings_csv.head(5))

# data_frame['embeddings'] = pd.Series(embeddings)