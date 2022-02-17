import newsapi
from newsapi import NewsApiClient
import requests
import json
import pandas


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
output_articles_uk = pandas.read_json(json.dumps(articles_list_uk))

responses_usa = json.loads(output2)
articles_list_usa = responses_usa['articles']
output_articles_usa = pandas.read_json(json.dumps(articles_list_usa))
 
output_articles = output_articles_uk.append(output_articles_usa).reset_index(drop=True)

#adding values from the output articles data (stored as objects within array) to a larger dataframe to be added 
df["title"] = output_articles["title"]
#if author blank use name
df["author"] = output_articles["author"]

df["datetime"] = output_articles["publishedAt"]
df["source"] = output_articles["url"]
df["body"] = output_articles["description"]

#write the outputs into a csv
df.to_csv('newsout.csv', mode='a', index=False)

print("CSV Outported")




































