from gensim.models import Word2Vec
import gensim
from nltk.tokenize import sent_tokenize, word_tokenize
import numpy as np
# import nltk
# nltk.download('punkt')

from pdf_extraction import text_extraction_pypdf




def generate_embedding(chunk):
    data = []
    
    for i in sent_tokenize(chunk):
        temp = []

        for j in word_tokenize(i):
            temp.append(j.lower())
    
        data.append(temp)

    # CBOW model
    model1 = gensim.models.Word2Vec(data, min_count=1, vector_size=100, window=5)

    text_vector = np.mean([model1.wv[word] for sentence in data for word in sentence if word in model1.wv], axis=0)

    # print(text_vector)
    return model1, text_vector

def compile_embeddings(chunks):
    embeddings = []
    for chunk in chunks:
        model, embedding = generate_embedding(chunk)
        embeddings.append(embedding)

    return embeddings    

'''
USEFUL FUNCTIONS

1) retreiving vector embedding for individual word

    selected_words = []             # list of words

    for word in selected_words:
        if word in model1.wv:
            print(f"Vector embedding for '{word}':\n{model1.wv[word]}\n")
        else:
            print(f"No vector embedding found for '{word}'.\n")


2) Similarity Results

    print("Cosine similarity between '*some word*' " +
        "and '*another word*' - *Model* : ",
        model1.wv.similarity('*some word*', '*another word*'))


3) Skip Gram Model
    model2 = gensim.models.Word2Vec(data, min_count=1, vector_size=100,
                                window=5, sg=1)


'''


