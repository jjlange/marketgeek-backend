import re

import pandas as pd

import numpy as np

from bertopic import BERTopic
# from sklearn.datasets import fetch_20newsgroups

# docs = fetch_20newsgroups(subset='all',  remove=('headers', 'footers', 'quotes'))['data']

# print(type(docs))
data_frame = pd.read_csv('full_data_1.csv')

bodies = data_frame['body'].head(5)
doc_new = []


def clean_body():
    pattern = r'[0-9]'
    for b in bodies:
        print(b)
        string = re.sub(pattern, '', b)
        print(string)
        # doc_new.append(string)

clean_body()
print(doc_new)


topic_model = BERTopic(verbose=True)
topics = topic_model.fit_transform(bodies)
# print(topic_model.get_topic_info())
print(topic_model.get_topic(0))
