import numpy as np
import milvus_config as milvus_config
import milvus_operations as milvus_operations
import chunking as chunking
import generate_embeddings as generate_embeddings
import pdf_extraction as pdf_extraction
import similarity_search as similarity_search
from pymilvus import (
    connections, 
    utility, 
    FieldSchema, CollectionSchema, DataType,
    Collection,
)
import random
import json
import time

fmt = "\n === {} === \n"



milvus_config.list_connections_milvus()

milvus_config.connect_milvus("default")

milvus_config.list_connections_milvus()

milvus_config.drop_collection("embedding_collection")

milvus_config.create_collection("embedding_collection", "test", 100)

embedding_collection = milvus_config.retrieve_collection("embedding_collection")

list_of_lists = [[random.randint(1, 100)/10 for _ in range(100)] for _ in range(1)]


#GRAB DATA

pdf_path = input("Enter a pdf: ")
# pdf_path = "/Users/siddharthrajaraman/Sid/Dev/semantics/pdfs/Annual_Report_2022_web.pdf"


time.sleep(1)
text = pdf_extraction.text_extraction_pypdf(pdf_path)

chunks = chunking.contextual_chunk(text)

print("NUMBER OF CHUNKS: " + str(len(chunks)))


embeddings = generate_embeddings.compile_embeddings(chunks)

milvus_operations.insert_milvus_data(embedding_collection, embeddings)

# Generate Query
query = input("Enter a query: ")
model, query_embedding = generate_embeddings.generate_embedding(query)



results = milvus_operations.similarity_search(embedding_collection, query_embedding)


result_list = []

for id in results[0].ids:
    similarity_score = similarity_search.cosine_similarity(model, embeddings[int(id)], query_embedding)
    print(similarity_score)
    temp = {
        "query" : query,
        "similarity score" : str(similarity_score),
        "id" : id,
        "embedding" : embeddings[int(id)].tolist(),
        "content" : chunks[int(id)]
    }
    result_list.append(temp)

# for result in result_list:
#     print(result)

with open('data.json', 'w') as f:
    json.dump(result_list, f)
# ===================================================
# perform similarity search
#   -  