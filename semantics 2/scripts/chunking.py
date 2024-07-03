from pdf_extraction import text_extraction_pypdf
from langchain.text_splitter import SpacyTextSplitter #conda install langchain -c conda-forge






# text = text_extraction_pypdf()


def contextual_chunk(text):
    text_splitter = SpacyTextSplitter()

    docs = text_splitter.split_text(text) 

    return docs
    # print(docs)



