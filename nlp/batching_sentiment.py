from transformers import AutoTokenizer, AutoModelForSequenceClassification
# from transformers import BertTokenizer, BertForSequenceClassification

import numpy as np
import pandas as pd
import dask.dataframe as dd
import torch
from tqdm import tqdm
# from sklearn.datasets import make_blobs
import hdbscan
from hdbscan.flat import (HDBSCAN_flat,
                          approximate_predict_flat,
                          membership_vector_flat,
                          all_points_membership_vectors_flat)


tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")

df = pd.read_csv("data/topic_size_min_100.csv")
# df = pd.read_csv("data/full_data_complete.csv")
ddf = dd.from_pandas(df, npartitions=1)

tensors = []

for chunk in tqdm(np.array_split(df, 1024)): #1024 split
    titles = list(chunk['title'])
    inputs = tokenizer(titles, padding = True, truncation = True, return_tensors='pt')
    outputs = model(**inputs)
    predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
    predictions = predictions.detach().numpy()
    tensors += list(predictions)

np.save('data/tensors.npy', tensors)

# ddf['tensors(+/-/n)'] = pd.Series(tensors)
# ddf.to_csv('data/full_dataset_w_tensors_*.csv', index=False)

# df.to_csv('data/dataset_w_tensors_*.csv', index=False)

# --------- clustering with hdbscan ----------------

# clusterer = HDBSCAN_flat(tensors,
#                          n_clusters=15, min_cluster_size=1000)
#
# # data = list(df['tensors(+/-/n)'])
# data = list(ddf['tensors(+/-/n)'])
#
# # hdbscan = hdbscan.HDBSCAN(min_cluster_size=10, min_samples=5)
# #                          cluster_selection_method='eom',
# # labels = hdbscan.fit_predict(data)
#
# labels = clusterer.labels_
# output = pd.Series(list(labels))
#
# # df['sentiment_cluster_id'] = output
# ddf['sentiment_cluster_id'] = output
#
# # df.to_csv('data/full_dataset.csv', index=False)
# # ddf.to_csv('data/full_dataset_*.csv', index=False)
# ddf.to_csv('data/full_dataset_*.csv', index=False)


