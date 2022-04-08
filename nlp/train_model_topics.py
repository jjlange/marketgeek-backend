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

# custom functions

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# import data: training articles are 80% of full dataset = 337680.8 (round up to 337861), transforming the whole dataset

# training data_frame
data_frame = pd.read_csv('../data/full_data.csv').head(337861)
# data to transform
data_frame_2 = pd.read_csv('../data/full_data.csv')
# find articles
training_articles = data_frame['body'].sample(100)
input_articles = data_frame_2['body'].sample(n=5)
output_data_frame = data_frame_2
single_article = data_frame_2['body'].iloc[0]

# find length of total data set and train on 80%, transform all articles and insert into dataframe

# set up mongodb server on igor

# preprocess and compute topics insert current news dataset

topics = []


def lemmatize_function_single(article):
    lem = WordNetLemmatizer()
    filtered_tokens = []
    body = word_tokenize(remove_stopwords(re.sub(r'[^\w\s]', '', article).lower()))
    for word in body:
        filtered_tokens.append(lem.lemmatize(word))

    # sentences = " ".join(filtered_tokens)
    # article = re.sub(r'[0-9]+', '', sentences)
    return filtered_tokens


def lemmatize_function_multi(articles):
    lem = WordNetLemmatizer()
    bodies = []
    for a in articles:
        if type(a) != str:
            a = str(a)
        sentences = lemmatize_function_single(a)
        bodies.append(sentences)
    return bodies


def convert_to_str(articles):
    output = []
    for a in articles:
        if type(a) != str:
            a = str(a)
        output.append(a)
    return output




def compute_BERTopics_single(article):
    tokens = lemmatize_function_single(article)
    topic_model = BERTopic(verbose=True)
    topic_model.fit_transform(tokens)
    return topic_model.visualize_topics()


topic_model = BERTopic(min_topic_size=100, calculate_probabilities=False, low_memory=True, verbose=True, n_gram_range=(1, 2))


def compute_BERTopics_multi(training_data, input_data, model):
    training_data = lemmatize_function_multi(training_data)
    input_data = lemmatize_function_multi(input_data)
    model = topic_model.fit(training_data, embeddings=None, y=None)
    topics_output = []
    topic = model.transform(input_data)
    topics_output.append(topic)
    return topics_output


def compute_lda(article):
    # loop through all articles and each time add the topics to the list
    tokens = lemmatize_function_multi(article)
    # tokens = lemmatize_function_single(article)
    # convert to corpora dictionary
    print(tokens)
    dict_ = corpora.Dictionary(tokens)
    matrix = [dict_.doc2bow(i) for i in tokens]
    lda = gensim.models.ldamodel.LdaModel
    # running and training the lda model
    lda_model = lda(matrix, num_topics=15, id2word=dict_, passes=10, random_state=100, eval_every=None)
    # for i, topic in lda_model.show_topics(formatted=True, num_topics=15, num_words=5):
        # print(str(i) + ": " + topic)
        # print()
    vis = pyLDAvis.gensim_models.prepare(lda_model, matrix, dict_)
    return vis

# fit and transform using function
# topic_ids_per_doc = compute_BERTopics_multi(training_articles, input_articles, topic_model)

# organise and insert data
# all_topics = topic_model.get_topics()
# topics_array = find_total_topics(all_topics)


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
# output = append_to_data()
# output.to_csv('data/topic_size_min_100.csv', index=False)

# print(input)
# print(single_article)
result = compute_lda(training_articles)

print(result)
fig = go.Figure(result)
fig.show()





# data = {'topic_id': topic_ids_per_doc, 'topics': ''}

# df = pd.DataFrame(columns=['topic_id', 'topics'])
# df['topic_id'] = data[0]

# df['topic_id'] = df.assign(topic_ids_per_doc[0].str.split(','))
# df['topics'] = topics_array[df['topic_id']]

# print(df)


# new_df =

# for id in topic_ids_per_dec:
#     print("id=" + str(id) + ", topics are: " + str(topics_array[id + 1]))
#     print()

# print(topics_per_doc)
# for t in topics_per_doc:
#     topic_id = t[0]
#     topic_word = t[1][0][0]
#
#     print(topic_id)
#     print("---------------------")
#     print(topic_word)
#     print("\n\n")



#     print(t)
#     print()
# for t, x in topics_per_doc, training_articles:
#     print(x + t)

# lemmatize_function_multi(training_articles)

# result.transform(single_doc)
# compute_BERTopics_multi(articles).save("test_model_1")


# ------------ visualise -------------
# fig = go.Figure(compute_BERTopics_multi(training_articles, testing_articles))
# fig.show()
