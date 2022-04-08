import pandas as pd
import dask.dataframe as dd
import dask.array as da
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


model = SentenceTransformer('all-MiniLM-L6-v2', device='cuda')

# nltk.download('wordnet')
# nltk.download('omw-1.4')
# nltk.download('stopwords')
# nltk.download('punkt')

# step 1, obtain sentences

# merging two datasets and outputting as data_frame -------------
# df1 = pd.read_csv('small_data_1.csv')
# df2 = pd.read_csv('small_data_2.csv')

# data_frame = pd.read_csv('small_data_1.csv')

# frames = [df1, df2]

# data_frame = pd.concat(frames).reset_index()


# for full set ------------

data_frame = pd.read_csv('../data/full_data_complete.csv')

sentences = data_frame['title']

#using one large data set

# step 2, tokenize sentences

# tokenized_sent = []
# for s in sentences:
#     tokenized_sent.append(word_tokenize(s.lower()))

# print(tokenized_sent)

# step 4, use SentenceBert to create the vectors
# we will now encode the sentences and can show the sentence vectors

sentence_embeddings = model.encode(sentences, show_progress_bar=True)

ddf = dd.from_pandas(data_frame, npartitions=2)

s = pd.Series(list(sentence_embeddings))

ddf['embeddings'] = s

ddf.to_csv('data/dataset_w_embeddings_*.csv', index=False)


def lemmatize_function(type):
    lem = WordNetLemmatizer()
    bodies = []
    filtered_tokens = []
    # stop_words = set(stopwords.words('english'))

    string = data_frame[type]
    for row in string:
        bodies.append(word_tokenize(remove_stopwords(re.sub(r'[^\w\s]', '', row))))

    for sentence in bodies:
        for word in sentence:
            filtered_tokens.append(lem.lemmatize(word))

    return filtered_tokens





