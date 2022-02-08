import pymongo
import pandas as pd 
import json

#local client
client = pymongo.MongoClient("mongodb://localhost:27017")
#Use pandas to read csv file as datafile
datafile = pd.read_csv("newsout.csv") 
data = datafile.to_dict(orient = "records")

#Create database
db = client["News"]

#Cram the data in 
db.NewsDataset.insert_many(data)
