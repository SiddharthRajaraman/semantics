from pymilvus import (
    connections, 
    utility, 
    FieldSchema, CollectionSchema, DataType,
    Collection,
)

import pymilvus

fmt = "\n === {} === \n"
error_fmt = "\n --- Error: {} --- \n"


# =================== ADD CONNECTION TO MILVUS ===========================
# add connection to milvus
def add_connection_milvus():
    print(fmt.format(f"Add Milvus Connection: embedding_collection"))
    try:
        # temp = exec(connection_name)
        connections.add_connection(
            default = {"host": "localhost", "port": "19530"}
        )
    except:
        print(error_fmt.format("Cannot Add Milvus Connection"))

# =================== CONNECT TO MILVUS ===========================
# connect to milvus

def connect_milvus(connection_name):
    print(fmt.format(f"connect to milvus: {connection_name }"))
    try:
        connections.connect(connection_name, host = "localhost", port = "19530")
    except:
        print(error_fmt.format("Cannot Connect to Milvus"))

# =================== LIST MILVUS CONNECTIONS ===========================
# list milvus connections
def list_connections_milvus():
    print(fmt.format("List Milvus Connections"))

    try:
        print(connections.list_connections())
    except:
        print(error_fmt.format("Error listing Milvus Connections"))


# =================== CREATE MILVUS COLLECTION ===========================

def create_collection(collection_name, collection_description, embedding_dimension):
    print(fmt.format(f"Create milvus Collection: {collection_name}"))

    fields = [
        FieldSchema(name = "pk", dtype = DataType.VARCHAR, is_primary = True, auto_id = False, max_length = 100), #ids
        FieldSchema(name = "embeddings", dtype = DataType.FLOAT_VECTOR, dim = embedding_dimension)
    ]
    # create Collection Schema
    schema = CollectionSchema(fields, collection_description)
    temp_collection = Collection(collection_name, schema, consistency_level = "Strong")

    return temp_collection


# =================== CHECK IF COLLECTION EXISTS ===========================

def locate_collection(collection_name):
    print(utility.has_collection(collection_name))  # true if exists; false if not

# =================== RETRIEVE COLLECTION ===========================

def retrieve_collection(collection_name):
    try:
        collection = Collection(collection_name)
        return collection

    except pymilvus.exceptions.SchemaNotReadyException:
        print(error_fmt.format("Collection does not exist"))


# =================== DROP COLLECTION ===========================
# permanently deletes collection and everything inside it
def drop_collection(collection_name):
    try:
        utility.drop_collection(collection_name)
    except pymilvus.exceptions.SchemaNotReadyException:
        print(error_fmt.format("Collection does not Exist"))
