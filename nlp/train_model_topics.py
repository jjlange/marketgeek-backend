import pandas as pd
import re
import os
import gensim
import gensim.corpora as corpora
import plotly.graph_objects as go
import pyLDAvis
import pyLDAvis.gensim_models
from bertopic import BERTopic
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from gensim.parsing.preprocessing import remove_stopwords

# script that calculates topic clustering using both LDA and BERTopic

os.environ["TOKENIZERS_PARALLELISM"] = "false"
# import data: training articles are 80% of full dataset = 337680.8 (round up to 337861), transforming the whole dataset

# training data_frame
data_frame = pd.read_csv('../data/full_data.csv').head(337861)

# data to transform
data_frame_2 = pd.read_csv('../data/full_data.csv')

# find articles, can be samples or full dataframes - scale for testing
training_articles = data_frame['body'].sample(100)
input_articles = data_frame_2['body'].sample(n=5)
output_data_frame = data_frame_2
single_article = data_frame_2['body'].iloc[0]

# find length of total data set and train on 80%, transform all articles and insert into dataframe
# preprocess and compute topics insert current news dataset

# init list to store the topics
topics = []

# set hyper parameters of BERTopic
topic_model = BERTopic(min_topic_size=100, calculate_probabilities=False, low_memory=True, verbose=True, n_gram_range=(1, 2))


# function that lemmatizes a single news article
def lemmatize_function_single(article):
    lem = WordNetLemmatizer()
    filtered_tokens = []
    body = word_tokenize(remove_stopwords(re.sub(r'[^\w\s]', '', article).lower()))
    for word in body:
        filtered_tokens.append(lem.lemmatize(word))
    return filtered_tokens

# function that lemmatizes multiple news articles
def lemmatize_function_multi(articles):
    lem = WordNetLemmatizer()
    bodies = []
    for a in articles:
        if type(a) != str:
            a = str(a)
        sentences = lemmatize_function_single(a)
        bodies.append(sentences)
    return bodies

# function that converts articles to type string
def convert_to_str(articles):
    output = []
    for a in articles:
        if type(a) != str:
            a = str(a)
        output.append(a)
    return output

# function that fits and transforms a single article
def compute_BERTopics_single(article):
    tokens = lemmatize_function_single(article)
    topic_model = BERTopic(verbose=True)
    topic_model.fit_transform(tokens)
    return topic_model.visualize_topics()

# function that computes topic ids and list of topics
# params: training_data (dataframe), input_data (dataframe) and the model (BERTopic)
# returns a list of the topic_ids and list of topics
def compute_BERTopics_multi(training_data, input_data, model):
    training_data = lemmatize_function_multi(training_data)
    input_data = lemmatize_function_multi(input_data)
    # fit model
    model = topic_model.fit(training_data, embeddings=None, y=None)
    topics_output = []
    # transform input_data and output a list of the topic_ids and list of topics
    topic = model.transform(input_data)
    topics_output.append(topic)
    return topics_output


def compute_lda(article):
    tokens = lemmatize_function_multi(article)
    dict_ = corpora.Dictionary(tokens)
    matrix = [dict_.doc2bow(i) for i in tokens]
    lda = gensim.models.ldamodel.LdaModel
    # running and training the lda model
    lda_model = lda(matrix, num_topics=15, id2word=dict_, passes=10, random_state=100, eval_every=None)
    for i, topic in lda_model.show_topics(formatted=True, num_topics=15, num_words=5):
        print(str(i) + ": " + topic)
        print()
    vis = pyLDAvis.gensim_models.prepare(lda_model, matrix, dict_)
    return vis

# fit and transform using function
topic_ids_per_doc = compute_BERTopics_multi(training_articles, input_articles, topic_model)

# organise and insert data
all_topics = topic_model.get_topics()
topics_array = find_total_topics(all_topics)


def append_to_data():
    ids = topic_ids_per_doc[0][0]
    dictionary = {'topic_id': [], 'topics': []}
    for i in ids:
        dictionary['topic_id'].append(i)
        dictionary['topics'].append(topics_array[i + 1])
    df = pd.DataFrame.from_dict(dictionary)
    result_df = output_data_frame.join(df)
    return result_df


# export data_frame to csv
output = append_to_data()
output.to_csv('data/topic_size_min_100.csv', index=False)


# visualise results for LDA
result = compute_lda(training_articles)
print(result)
fig = go.Figure(result)
fig.show()

# visualise BERTopics
fig = go.Figure(compute_BERTopics_multi(training_articles, input_articles))
fig.show()



