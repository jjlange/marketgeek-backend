import re

import pandas as pd

import numpy as np
import plotly.graph_objects as go


from bertopic import BERTopic
# from sklearn.datasets import fetch_20newsgroups

# docs = fetch_20newsgroups(subset='all',  remove=('headers', 'footers', 'quotes'))['data']

# print(type(docs))
data_frame = pd.read_csv('data/full_dataset_march16.csv').head(10000)

sample_df = data_frame.sample(n = 10000)

# bodies = data_frame['body']
bodies = sample_df['body']



doc_new = []


def clean_body():
    pattern = r'[0-9]'
    for b in bodies:
        print(b)
        string = re.sub(pattern, '', b)
        print(string)
        # doc_new.append(string)

# clean_body()
# print(doc_new)


topic_model = BERTopic(verbose=True)

topic_model.fit_transform(bodies)

# print(topic_model.visualize_topics())

# print(topic_model.get_topic_info())
# print(topic_model.get_topic(0))


# ------------ visualise -------------
# fig = go.Figure(topic_model.visualize_topics())
# fig = go.Figure(topic_model.visualize_barchart())
fig = go.Figure(topic_model.visualize_hierarchy())
fig.show()
