import pymongo
from pymongo import MongoClient
import pandas as pd
import yfinance as yf
import numpy as np


client = MongoClient("localhost", 27017)
financedb = client["finance_database"]


def trends_for_ticker(ticker, short_term=True):
    coll = financedb[ticker]
    cur = coll.find()
    prices_lst_dct = []
    for doc in cur:
        prices_lst_dct.append(doc)

    df_prices = pd.DataFrame(prices_lst_dct)

    if short_term:
        # df_prices["12_DEMA"] = df_prices["Close_price"].ewm(span=12, min_periods=1).mean()
        # df_prices["26_DEMA"] = df_prices["Close_price"].ewm(span=26, min_periods=1).mean()
        df_prices['DEMA_12'] = dema(df_prices, 12, 'Close_price')
        df_prices['DEMA_26'] = dema(df_prices, 26, 'Close_price')
        df_prices['Signal'] = 0.0
        df_prices['Signal'] = np.where(df_prices['DEMA_12'] > df_prices['DEMA_26'], 1.0, 0.0)
    else:
        # df_prices["50_DEMA"] = df_prices["Close_price"].ewm(span=50, min_periods=1).mean()
        # df_prices["200_DEMA"] = df_prices["Close_price"].ewm(span=200, min_periods=1).mean()
        df_prices['DEMA_50'] = dema(df_prices, 50, 'Close_price')
        df_prices['DEMA_200'] = dema(df_prices, 200, 'Close_price')
        df_prices['Signal'] = 0.0
        df_prices['Signal'] = np.where(df_prices['50_DEMA'] > df_prices['200_DEMA'], 1.0, 0.0)

    df_prices['Position'] = df_prices['Signal'].diff()

    trends = df_prices[np.logical_or(df_prices['Position'] == 1, df_prices['Position'] == -1)]

    upward_trends = [{"Start_Date": trends['Date'].iloc[row], "End_date": trends['Date'].iloc[row + 1]} for row in
                     range(len(trends) - 1) if trends['Position'].iloc[row] == 1]
    downward_trends = [{"Start_Date": trends['Date'].iloc[row], "End_date": trends['Date'].iloc[row + 1]} for row in
                       range(len(trends) - 1) if trends['Position'].iloc[row] == -1]

    return {"upward_trends": upward_trends, "downward_trends": downward_trends}



def dema(data, timeframe, column):
    EMA = data[column].ewm(span=timeframe, min_periods=1).mean()
    DEMA = 2*EMA - EMA.ewm(span=timeframe, min_periods=1).mean()
    return DEMA


tickers_used = ["EURUSD=X","GBPUSD=X","CL=F", "GC=F", "^GSPC", "^IXIC", "^DJI"]

def construct_proper_dict_from_dates(dct):
    return [{"Date": date, "Close_price": val} for date, val in turn_date_to_str(dct).items()]

def download_ticker():
    financedb = client["finance_database"]
    for ticker in tickers_used:
        df_ticker = yf.download(ticker, start='2006-10-20', end='2013-11-26', interval = "1d")
        proper_ticker_dict_lst = construct_proper_dict_from_dates(df_ticker["Close"])
        ticker_in_db = financedb[ticker]
        ticker_in_db.insert_many(proper_ticker_dict_lst)

def turn_date_to_str(dct):
    return {str(date): val for (date, val) in dct.items()}


download_ticker()


list = []
# for ticker in tickers_used:
#     list.append(trends_for_ticker(ticker))

# print(len(trends_for_ticker(tickers_used[0])["upward_trends"]))

# print(len(trends_for_ticker(tickers_used[0])["downward_trends"]))

dict = trends_for_ticker(tickers_used[0])

df2 = pd.DataFrame(dict["upward_trends"])
df3 = pd.DataFrame(dict["downward_trends"])


df2.to_csv("data/tickers/" + tickers_used[0] + "_upward_trends.csv", index=False)
df3.to_csv("data/tickers/" + tickers_used[0] + "_downward_trends.csv", index=False)

