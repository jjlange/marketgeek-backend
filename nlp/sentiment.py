from transformers import AutoTokenizer, AutoModelForSequenceClassification
# from transformers import BertTokenizer, BertForSequenceClassification

import numpy as np
import pandas as pd
import dask.dataframe as dd
import torch
from sklearn.datasets import make_blobs
import hdbscan

tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")

cuda = torch.device('cuda')

df = pd.read_csv("data/full_data_complete.csv")

sentences = list(df['title'])

ddf = dd.from_pandas(df, npartitions=1)

inputs = tokenizer(sentences, padding = True, truncation = True, return_tensors='pt')

# for

outputs = model(**inputs)
predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
predictions = predictions.detach().numpy()
predictions = pd.Series(list(predictions))

ddf['tensor'] = predictions


# --------- clustering with hdbscan

data = list(predictions)

import hdbscan
hdbscan = hdbscan.HDBSCAN(min_cluster_size=10, min_samples = 5)
labels = hdbscan.fit_predict(data)

output = pd.Series(list(labels))

ddf['sentiment_clusters'] = output


# ------- export
ddf.to_csv('data/dataset_w_tensors_*.csv', index=False)

# ddf.to_csv('data/dataset_w_tensors_.csv', index=False)



# ------ cosine similarity
# cos = torch.nn.CosineSimilarity(dim=0)
# output = cos(tensor1, tensor2)
# print("Cosine similarity",output)

