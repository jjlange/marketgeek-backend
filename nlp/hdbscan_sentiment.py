import numpy as np
import pandas as pd
import dask.dataframe as dd
import torch
from tqdm import tqdm
# from sklearn.datasets import make_blobs
import hdbscan

# script to calculate sentiment ids based on the tensors, discontinued due to a new method for calcalating ids

from hdbscan.flat import (HDBSCAN_flat,
                          approximate_predict_flat,
                          membership_vector_flat,
                          all_points_membership_vectors_flat)

df = pd.read_csv("../data/full_dataset_w_tensors_0.csv")
ddf = dd.from_pandas(df, npartitions=1)

# tensors = df['tensors(+/-/n)']
# print((tensors[0]))

tensors = np.load('../data/tensors.npy')
print(tensors)

# --------- clustering with hdbscan ----------------

clusterer = HDBSCAN_flat(tensors,
                         n_clusters=3, min_cluster_size=150)

labels = clusterer.labels_
output = pd.Series(list(labels))

ddf['sentiment_cluster_id'] = output
ddf.to_csv('data/full_sentiment_id_*.csv', index=False)

