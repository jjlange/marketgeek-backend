from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
import pandas as pd
import dask.dataframe as dd
import torch
from tqdm import tqdm
from hdbscan.flat import (HDBSCAN_flat,
                          approximate_predict_flat,
                          membership_vector_flat,
                          all_points_membership_vectors_flat)

# script for calculating sentiment using FinBERT and outputting a numpy file of all tensors

tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")

df = pd.read_csv("../data/topic_size_min_100.csv")
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

np.save('../data/tensors.npy', tensors)


# saving in csv, discontinued as does not preserve data type
ddf['tensors(+/-/n)'] = pd.Series(tensors)
ddf.to_csv('data/full_dataset_w_tensors_*.csv', index=False)
df.to_csv('data/dataset_w_tensors_*.csv', index=False)



# --------- clustering with hdbscan ----------------
clusterer = HDBSCAN_flat(tensors,
                         n_clusters=15, min_cluster_size=1000)

data = list(ddf['tensors(+/-/n)'])
labels = clusterer.labels_
output = pd.Series(list(labels))

ddf['sentiment_cluster_id'] = output
ddf.to_csv('data/full_dataset_*.csv', index=False)


