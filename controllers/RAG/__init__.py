from flask import Blueprint
from config import mongodb_client
from sentence_transformers import SentenceTransformer
from pymongo.operations import SearchIndexModel
from constants.https_status_codes import *
from utils.ApiResponse import ApiResponse
from utils.ApiError import ApiError
from gpt4all import GPT4All
import os

RAG=Blueprint("RAG",__name__,url_prefix="/api/v1/RAG")

MODEL_PATH="path"
os.makedirs(MODEL_PATH, exist_ok=True)
model = SentenceTransformer('mixedbread-ai/mxbai-embed-large-v1')
model.save(MODEL_PATH,safe_serialization=False)
model = SentenceTransformer(MODEL_PATH)
collection = mongodb_client["ASKIO"]["chat_history"]


# DB_NAME = "mongodb_rag_lab"
# COLLECTION_NAME = data["collection_name"]

@RAG.route("/parse_file",methods=['POST'])
def rag_index():
        # Connect to your local Atlas deployment or Atlas Cluster
    client = mongodb_client
    # Select the sample_airbnb.listingsAndReviews collection
    # collection = client["ASKIO"]["chat_history"]
    # Load the embedding model (https://huggingface.co/sentence-transformers/mixedbread-ai/mxbai-embed-large-v1)
    
    # Define function to generate embeddings
   
    # Filters for only documents with a summary field and without an embeddings field
    # filter = { '$and': [ { 'summary': { '$exists': True, '$ne': None } }, { 'embeddings': { '$exists': False } } ] }
    filter={}
    # Creates embeddings for subset of the collection
    updated_doc_count = 0
    for document in collection.find().limit(50):
        print(document)
        text = document['content']
        embedding = get_embedding(text)
        collection.update_one({ '_id': document['_id'] }, { "$set": { 'embeddings': embedding } }, upsert=True)
        updated_doc_count += 1
    print("Documents updated: {}".format(updated_doc_count))

    search_index(collection)
    return ApiResponse("DOCUMENT FETCHED",HTTP_200_OK,updated_doc_count)


def search_index(collection):
    # Create your index model, then create the search index
    search_index_model = SearchIndexModel(
    definition = {
        "fields": [
        {
            "type": "vector",
            "numDimensions": 384, #1024
            "path": "embeddings",
            "similarity": "cosine"
        }
        ]
    },
    name = "vector_index",
    type = "vectorSearch" 
    )
    collection.create_search_index(model=search_index_model)

def get_embedding(text):
    return model.encode(text).tolist()

def get_query_results(query):
   query_embedding = get_embedding(query)
   pipeline = [
      {
            "$vectorSearch": {
               "index": "vector_index",
               "queryVector": query_embedding,
               "path": "embeddings",
               "exact": True,
               "limit": 5,
            }
      }, {
            "$project": {
               "_id": 0,
               "summary": 1,
               "listing_url": 1,
               "score": {
                  "$meta": "vectorSearchScore"
               }
            }
      }
   ]
   results = collection.aggregate(pipeline)
   array_of_results = []
   for doc in results:
      array_of_results.append(doc)
   return array_of_results

# local_llm_path = "path\orca-mini-3b.gguf"
local_llm_path="./orca-mini-3b-gguf2-q4_0.gguf"
local_llm = GPT4All(local_llm_path)


@RAG.route("/ask_question",methods=['POST'])
def ask_question():
    print("WORKING")
    question = "Can you recommend a few AirBnBs that are beach houses? Include a link to the listing."
    documents = get_query_results(question)
    print(documents)
    if (len(documents)==0) :
        return ApiError("COULDN't FIND DOCUMENT",HTTP_404_NOT_FOUND)
    text_documents = ""
    for doc in documents:
        summary = doc.get("summary", "")
        link = doc.get("listing_url", "")
        string = f"Summary: {summary} Link: {link}. \n"
        text_documents += string
    prompt = f"""Use the following pieces of context to answer the question at the end.
        {text_documents}
        Question: {question}
    """
    response = local_llm.generate(prompt)
    cleaned_response = response.replace('\\n', '\n')
    print(cleaned_response)
    return ApiResponse("Working QUEASTION",HTTP_200_OK,cleaned_response)