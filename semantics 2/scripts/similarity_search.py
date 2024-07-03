from gensim.models import Word2Vec
import gensim
from nltk.tokenize import sent_tokenize, word_tokenize
import numpy as np
from numpy.linalg import norm


def cosine_similarity(model, embedding, query_embedding):
    query_np = np.array(query_embedding)

    temp_np = np.array(embedding)

    similarity = np.dot(temp_np, query_np)/(norm(temp_np) * norm(query_embedding))

    return similarity
