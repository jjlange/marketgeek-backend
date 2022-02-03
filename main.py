import newsapi
from newsapi import NewsApiClient
import requests
import json
import pandas

newsapi = NewsApiClient(api_key='ee1453e25ade4e16b8fb219f9fe9e6f5')

top_headlines_url = 'https://newsapi.org/v2/top-headlines'
everything_news_url = 'https://newsapi.org/v2/everything'
sources_url = 'https://newsapi.org/v2/sources'


headers = {'Authorization': 'ee1453e25ade4e16b8fb219f9fe9e6f5'}

headlines_payload_us = {'category': 'business', 'country': 'us'}
headlines_payload_uk = {'category': 'business', 'country': 'gb'}
everything_payload = {'q': 'business', 'language': 'en', 'sortBy': 'popularity'}

response = requests.get(url=top_headlines_url, headers=headers, params=headlines_payload_uk)
output = json.dumps(response.json(), indent=2)
print(output)

responses = json.loads(output)
articles_list = responses['articles']
output_articles = pandas.read_json(json.dumps(articles_list))
output_articles.to_csv('C:/Users/Christian/Documents/newsout.csv')

















