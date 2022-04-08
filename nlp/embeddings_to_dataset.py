import pandas as pd
import nltk
from numpy import savetxt

# script to take the embeddings calculated from the text export to a dataset

nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('stopwords')
nltk.download('punkt')

# import embeddings from csv
embeddings_csv = pd.read_csv('embeddings.csv')
savetxt('embeddings.csv', sentence_embeddings, delimiter=',')

print(sentence_embeddings[1])
print(embeddings_csv.head(5))
data_frame['embeddings'] = pd.Series(embeddings)