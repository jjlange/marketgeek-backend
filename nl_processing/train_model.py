import pandas as pd
import dask.dataframe as dd
import dask.array as da
import re
import os
import gensim
import gensim.corpora as corpora
import plotly.graph_objects as go
from bertopic import BERTopic


from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from gensim.parsing.preprocessing import remove_stopwords

os.environ["TOKENIZERS_PARALLELISM"] = "false"
# from wordcloud import WordCloud

# import
data_frame = pd.read_csv('full_data_1.csv')

# get article bodies from dataframe
training_articles = data_frame['body'].head(20)
testing_articles = data_frame['body'].tail(10)
single_doc = data_frame['body'].tail(1)
# print(articles)

topics = []


def lemmatize_function_single(article):
    lem = WordNetLemmatizer()
    filtered_tokens = []
    body = word_tokenize(remove_stopwords(re.sub(r'[^\w\s]', '', article).lower()))
    for word in body:
        # filtered_tokens.append([lem.lemmatize(word)])
        filtered_tokens.append(lem.lemmatize(word))
    return filtered_tokens


def lemmatize_function_multi(articles):
    lem = WordNetLemmatizer()
    filtered_tokens = []
    bodies = []
    for a in articles:
        if type(a) != str:
            a = str(a)
        article = re.sub(r'[0-9]+', '', a)
        output = word_tokenize(remove_stopwords(re.sub(r'[^\w\s]', '', article).lower()))
        bodies.append(output)


    for b in bodies:
        for word in b:
            # filtered_tokens.append([lem.lemmatize(word)])
            filtered_tokens.append(lem.lemmatize(word))
    return filtered_tokens


def compute_BERTopics_single(article):
    tokens = lemmatize_function_single(article)
    topic_model = BERTopic(verbose=True)
    topic_model.fit_transform(tokens)
    # print(topic_model.get_topic_info())
    # print(topic_model.get_topic(0))
    return topic_model.visualize_topics()


def compute_BERTopics_multi(training_data, testing_data):
    tokens_training = lemmatize_function_multi(training_data)
    topic_model = BERTopic(nr_topics=8, top_n_words=5, calculate_probabilities=False, verbose=True, low_memory=True)
    print("Stage 1")
    # topic_model.fit_transform(tokens_training, embeddings=None, y=None)
    topic_model.fit(tokens_training, embeddings=None, y=None)
    print("Stage 2")
    for a in testing_data:
        # print(a)
        tokens_data = lemmatize_function_single(a)
        topic_model.transform(tokens_data)
        print(topic_model.get_topic_info())



    # return topic_model
    # print(topic_model.get_topic_info())
    # print(topic_model.get_topics())
    # return topic_model.visualize_topics()


def compute_lda(article):
    # loop through all articles and each time add the topics to the list
    tokens = lemmatize_function_multi(article)
    # convert to corpora dictionary
    dict_ = corpora.Dictionary(tokens)
    # convert to vectors
    matrix = [dict_.doc2bow(i) for i in tokens]
    # creating object for lda model
    lda = gensim.models.ldamodel.LdaModel
    # running and training the lda model
    lda_model = lda(matrix, num_topics=3, id2word=dict_, passes=10, random_state=100, eval_every=None)
    for i, topic in lda_model.show_topics(formatted=True, num_topics=3, num_words=5):
        print(str(i) + ": " + topic)
        print()


# print(articles[0])
# compute_lda(articles[0])


# -------- all articles ----------
# for a in articles:
#     compute_lda(a)
#     print()


# for a in articles:
#     compute_BERTopics_multi(a)
# compute_BERTopics_multi(training_articles, testing_articles).save("test_model_1")
compute_BERTopics_multi(training_articles, testing_articles)
# result.transform(single_doc)
# compute_BERTopics_multi(articles).save("test_model_1")


# ------------ visualise -------------
# fig = go.Figure(compute_BERTopics_multi(training_articles, testing_articles))
# fig.show()
