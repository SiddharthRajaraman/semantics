from pdf_extraction import text_extraction_pypdf
from generate_embeddings import generate_embedding
from chunking import contextual_chunk
from similarity_search import cosine_similarity


pdf_path = "/Users/siddharthrajaraman/Sid/Dev/semantics/pdfs/Annual_Report_2022_web.pdf"

extracted_text = text_extraction_pypdf(pdf_path)

chunks = contextual_chunk(extracted_text)

embeddings = []
model = None
for chunk in chunks:
    model, embedding = generate_embedding(chunk)
    embeddings.append(embedding)

query = input("Enter a Query: ")

model1, query_embedding = generate_embedding(query)

print(query_embedding)


similarity_list = cosine_similarity(model, embeddings, query_embedding)

for similarity in similarity_list:
    print(similarity)



max_similarity = max(similarity_list)

max_idx = similarity_list.index(max_similarity)

result = chunks[max_idx]
print("RESULT:")
print(result)