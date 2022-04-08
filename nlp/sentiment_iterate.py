import numpy as np
import pandas as pd

# load tensors from numpy file
tensors = np.load('../data/tensors.npy')

# init list for storing the max index
max_indexes = []

# loop through tensors and find max value for each one
for tensor in tensors:
    max = 0
    for t in range(0, 3):
        if tensor[t] > max:
            max = tensor[t]
            max_index = t
    max_indexes.append(max_index)

# import dataframe and add column to dataframe
df = pd.read_csv('../data/topic_size_min_100.csv')
df['sentiment_id'] = max_indexes

# export
df.to_csv('data/full_dataset_march15.csv', index=False)