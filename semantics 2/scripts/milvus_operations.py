from pymilvus import (
    connections, 
    utility, 
    FieldSchema, CollectionSchema, DataType,
    Collection,
)

import pymilvus

fmt = "\n === {} === \n"
error_fmt = "\n --- Error: {} --- \n"



# =================== INSERT DATA ===========================

def insert_milvus_data(collection, data):
    # generate pk values
    pk = [str(i) for i in range(len(data))]

    # initialize entities
    entities = [pk, data]


    insert_result = collection.insert(entities) # insert data
    collection.flush()

    # Create Indeces
    #   - needed for fast and efficient similarity searches
    index = {
        "index_type": "IVF_FLAT",
        "metric_type": "L2",
        "params": {"nlist": 128},
    }

    collection.create_index("embeddings", index)

    return insert_result


# =================== SIMILARITY SEARCH ===========================

def similarity_search(collection, query_embedding):
    search_params = {
        "metric_type": "L2", 
        "offset": 0, 
        "ignore_growing": False, 
        "params": {"nprobe": 10}
    }

    collection.load()

    results = collection.search(
        data=[query_embedding], 
        anns_field="embeddings", 
        # the sum of `offset` in `param` and `limit` 
        # should be less than 16384.
        param=search_params,
        limit=10,
        expr=None,
        # set the names of the fields you want to 
        # retrieve from the search result.
        output_fields=['pk'],
        consistency_level="Strong"
    )

    return results