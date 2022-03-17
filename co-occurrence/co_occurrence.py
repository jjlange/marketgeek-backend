import pymongo
from pymongo import MongoClient
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta


# from tickers import trends_for_ticker


df = pd.read_csv('data/full_dataset_march15.csv')

# find the start and end of the price trends
# get some news at the start and the end of the price trends
# iterate through all the news on a specific day, closing price of a specific day

upwards_ticker = pd.read_csv('data/tickers/EURUSD=X_upward_trends.csv')
downwards_ticker = pd.read_csv('data/tickers/EURUSD=X_downward_trends.csv')

date_string = str(upwards_ticker['Start_Date'].iloc[len(upwards_ticker) - 1])
# str2 = '2013-10-21'

datetime_object_ticker = datetime.strptime(date_string, "%Y-%m-%d %H:%M:%S")
prev_day = datetime_object_ticker - timedelta(days=2)

# print(prev_day)
# print(datetime_object_ticker)
# news_datetime = df['datetime'].iloc[1]
# datetime_object_news = datetime.strptime(news_datetime, "%Y-%m-%dt%H:%M:%Sz")
# print(datetime_object_news)


def convert_date(input):
    news_datetime = input[10:]
    datetime_object_news = datetime.strptime(news_datetime, "%Y-%m-%dt%H:%M:%Sz")
    return datetime_object_news


df['datetime'] = pd.to_datetime(df['datetime'], format="%Y-%m-%dt%H:%M:%Sz")

df.to_csv('data/full_dataset_march16.csv', index=False)

# print(df)



# print(df)











