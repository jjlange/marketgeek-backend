##
# MarketGeek Backend
# Copyright (C) 2022, MarketGeek.
#
# This is the backend application for MarketGeek
# It is responsible for handling all the requests related to the news data that has been proceeded.
##

from newsapi.newsapi_client import NewsApiClient
import requests
import json
import pandas
from io import StringIO

from flask import Flask, request

newsapi = NewsApiClient(api_key='ee1453e25ade4e16b8fb219f9fe9e6f5')

#grab headlines from news-api 
top_headlines_url = 'https://newsapi.org/v2/top-headlines'
everything_news_url = 'https://newsapi.org/v2/everything'
sources_url = 'https://newsapi.org/v2/sources'

#dictionary
struct_dict = {'title': [], 'author': [], 'datetime': [], 'source': [], 'body': []}
df = pandas.DataFrame(struct_dict)

headers = {'Authorization': 'ee1453e25ade4e16b8fb219f9fe9e6f5'}

#set parameter for request
headlines_payload__business_usa = {'category': 'business', 'language': 'en','country': 'us'}
headlines_payload_business_uk = {'category': 'business', 'language': 'en','country': 'gb'}
everything_payload = {'q': 'business', 'language': 'en', 'sortBy': 'popularity'}

#grab the headlines from locations specified in parameters
response_uk = requests.get(url=top_headlines_url, headers=headers, params=headlines_payload_business_uk)
response_usa = requests.get(url=top_headlines_url, headers=headers, params=headlines_payload__business_usa)

#convert the responce into json so can be output as a string 
output = json.dumps(response_uk.json(), indent=2)
output2 = json.dumps(response_usa.json(), indent=2)

#print(output,output2)

#turn strings into a array so we can seperate the articles 
responses_uk = json.loads(output)
articles_list_uk = responses_uk['articles']
output_articles_uk = pandas.read_json(StringIO(json.dumps(articles_list_uk)))

responses_usa = json.loads(output2)
articles_list_usa = responses_usa['articles']
output_articles_usa = pandas.read_json(StringIO(json.dumps(articles_list_usa)))
 
output_articles = output_articles_uk.append(output_articles_usa).reset_index(drop=True)

#adding values from the output articles data (stored as objects within array) to a larger dataframe to be added 
df["title"] = output_articles["title"]

df["author"] = output_articles["author"]

df["datetime"] = output_articles["publishedAt"]
df["source"] = output_articles["url"]
df["body"] = output_articles["description"]

#write the outputs into a csv
df.to_csv('newsout.csv')
#df.query('title == "Stocks stumble as Ukraine tensions worsen, investors seek safety in bonds - Reuters"',inplace = True)


result = df.to_json(orient="split")

# Initalise the backend web app
app = Flask(__name__)
api_version = "/v1"

# Routes for the API

# Route to filter by title
# Example: http://127.0.0.1:5000/v1/news/getByTitle?title=Take it or leave it: second hand picks for 4 March - Autocar
@app.route(api_version + "/news/getByTitle", methods=['GET'])
def get_data():
    # Query the result by title
    df.query('`title` == "' + request.args.get("title") + '"', inplace = True)

    # Convert to JSON
    json_dict = df.assign(index=df.index).to_dict(orient="list")
    
    # Return the result
    return json.dumps(json_dict)

# Run the API server
if __name__ == "__main__":
    app.run(debug=True)
