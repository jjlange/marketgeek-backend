# import pandas as pd
#
# topics = [['said', 'percent', 'million', 'bank', 'rashid', 'company', 'today', 'billion', 'capital'], ['said', 'company', 'percent', 'million', 'new', 'inc', 'story', 'shares', 'contact'], ['percent', 'oil', 'oao', 'said', 'russian', 'company', 'russia', 'moscow', 'gas'], ['percent', 'said', 'year', 'output', 'ore', 'production', 'nickel', 'tons', 'iron'], ['china', 'yuan', 'percent', 'year', 'said', 'shanghai', 'billion', 'bank', 'aircraft']]
# topic_ids = [([-1, -1, -1, -1, -1, -1, -1, -1, 3, 3, -1, -1, 2, 2, -1, -1, -1, -1, -1, -1], None)]
#
# # df = pd.DataFrame(columns=['topic_id', 'topics'])
# df_og = pd.read_csv('data/full_data.csv').tail(20).reset_index()
#
# ids = topic_ids[0][0]
#
# dictionary = {'topic_id': [], 'topics': []}
# for i in ids:
#     print(i)
#     dictionary['topic_id'].append(i)
#     dictionary['topics'].append(topics[i + 1])
#
# df = pd.DataFrame.from_dict(dictionary)
#
# df_final = df_og.join(df)
#
# print(df_final)
# import pandas as pd
#
# df = pd.read_csv('data/full_data_complete.csv')
#
# print(df.iloc[420000]['body'])
# print(df.iloc[420000]['topics'])

import pandas as pd
import dask.dataframe as dd
import dask.array as da
import re
import os
import gensim
import numpy as np
import gensim.corpora as corpora
import plotly.graph_objects as go
from bertopic import BERTopic


from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from gensim.parsing.preprocessing import remove_stopwords

from clean_topics import find_total_topics

def lemmatize_function_single(article):
    lem = WordNetLemmatizer()
    filtered_tokens = []
    body = word_tokenize(remove_stopwords(re.sub(r'[^\w\s]', '', article).lower()))
    for word in body:
        filtered_tokens.append(lem.lemmatize(word))

    sentences = " ".join(filtered_tokens)
    article = re.sub(r'[0-9]+', '', sentences)
    return article


def lemmatize_function_multi(articles):
    lem = WordNetLemmatizer()
    filtered_tokens = []
    bodies = []
    for a in articles:
        if type(a) != str:
            a = str(a)
        article = re.sub(r'[0-9]+', '', a)
        # output = word_tokenize(remove_stopwords(re.sub(r'[^\w\s]', '', article).lower()))
        output2 = remove_stopwords(article.lower())
        x = lemmatize_function_single(output2)
        print(x)
        bodies.append(output2)
    return bodies


text = 'guangzhou shipyard international co. (317) , a unit of chinas biggest shipbuilder, said first-quarter profit fell 50 percent because of higher prices of steel plates.    net income for the three months ended march 31 declined to 129.3 million  yuan  ($18.9 million), or 0.26 yuan a share, from 258 million yuan, or 0.52 yuan, the company said in a statement to the shanghai stock exchange.    they may still be using some of the high-priced inventory acquired in the middle of 2008 when steel prices were high, jack xu, a shanghai-based analyst at sinopac securities asia ltd., said before the announcement. the company had a very good first quarter in 2008, which makes the comparison tough.    guangzhou shipyard also incurred higher financing costs during the period as it received fewer orders, according to an april 9 exchange filing. orders placed with chinese shipbuilders plunged 94 percent in the first three months, according to government estimates, as a global recession erased demand for ships to carry raw materials and consumer goods.    guangzhou shipyard fell 0.7 percent in hong kong trading to close at hk$11.22 today before the earnings announcement. the shares have surged 58 percent this year,  compared  with a 5.8 percent gain in the key hang seng index.    the guangzhou, southern china-based company completed and delivered four vessels, the statement said. it secured orders for building 60 vessels, with a total tonnage of 2.66 million deadweight tons, it said.    chinese shipyards may have cash shortages of about $30 billion in the next three to four years, the  china  daily reported on april 8, citing li li, a deputy general manager of export-import bank of chinas ship finance department.    the biggest decline in global  shipping rates  in two decades led to a worldwide 95 percent decline in new vessel orders in march, according to clarkson plc, the worlds largest shipbroker.    to contact the reporter on this story: lee spears in beijing at   lspears2@bloomberg.net .    to contact the editor responsible for this story: teo chian wei at   cwteo@bloomberg.net .'

x1 = lemmatize_function_single(text)

print(x1)


