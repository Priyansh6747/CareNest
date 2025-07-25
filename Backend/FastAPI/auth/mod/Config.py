import os
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from dotenv import load_dotenv

load_dotenv(dotenv_path='.env')
Pass = os.getenv("MONGO_DB")
uri = f"mongodb+srv://aryanguptajiit:{Pass}@cluster0.h2txw5b.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"

# Create a new client and connect to the server
client = MongoClient(uri, server_api=ServerApi('1'))

db = client.CareNest
collection = db.Users
vaccines = db.master_vaccine

# Send a ping to confirm a successful connection
try:
    client.admin.command('ping')
    print("Pinged your deployment. You successfully connected to MongoDB!")
except Exception as e:
    print(e)