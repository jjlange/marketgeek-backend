import re
import pandas as pd
import plotly.graph_objects as go
from bertopic import BERTopic

# script to compute topic clustering using BERTopic

data_frame = pd.read_csv('../data/full_dataset_march16.csv').head(10000)

sample_df = data_frame.sample(n = 10000)

bodies = sample_df['body']
doc_new = []

def clean_body():
    pattern = r'[0-9]'
    for b in bodies:
        print(b)
        string = re.sub(pattern, '', b)
        print(string)
        # doc_new.append(string)

clean_body()
topic_model = BERTopic(verbose=True)
topic_model.fit_transform(bodies)

# ------------ visualise -------------
# fig = go.Figure(topic_model.visualize_topics())
# fig = go.Figure(topic_model.visualize_barchart())
fig = go.Figure(topic_model.visualize_hierarchy())
fig.show()
