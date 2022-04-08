import numpy as np
import pandas as pd

tensors = np.load('../data/tensors.npy')

max_indexes = []

for tensor in tensors:
    max = 0
    for t in range(0, 3):
        if tensor[t] > max:
            max = tensor[t]
            max_index = t
    max_indexes.append(max_index)


# print(max_indexes)

df = pd.read_csv('../data/topic_size_min_100.csv')

df['sentiment_id'] = max_indexes

# print(df)

df.to_csv('data/full_dataset_march15.csv', index=False)