from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
import pandas as pd
import dask.dataframe as dd
import torch
from sklearn.datasets import make_blobs
import hdbscan

# import FinBERT
tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")

# activate cuda
cuda = torch.device('cuda')

# import dataset and apply to dataframe
df = pd.read_csv("../data/full_data_complete.csv")

sentences = list(df['title'])

# set pandas to dask dataframe for managing large quantity of data
ddf = dd.from_pandas(df, npartitions=1)

# calculate tensors
inputs = tokenizer(sentences, padding = True, truncation = True, return_tensors='pt')
outputs = model(**inputs)
predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
predictions = predictions.detach().numpy()
predictions = pd.Series(list(predictions))
ddf['tensor'] = predictions


# clustering with hdbscan
data = list(predictions)
hdbscan = hdbscan.HDBSCAN(min_cluster_size=10, min_samples = 5)
labels = hdbscan.fit_predict(data)
output = pd.Series(list(labels))
ddf['sentiment_clusters'] = output

# ------- export -------- #
ddf.to_csv('data/dataset_w_tensors_*.csv', index=False)


# cosine similarity, currently not being used
cos = torch.nn.CosineSimilarity(dim=0)
output = cos(tensor1, tensor2)
print("Cosine similarity",output)

