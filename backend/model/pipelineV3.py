# import re
# from typing import List

# def load_text(file_path):
#     with open(file_path, 'r', encoding='utf-8') as file:
#         return file.read()

# DOC_PATH="parsedEARregsV1.txt"
# texts = load_text(DOC_PATH)

# def split_text(text: str) -> List[str]:    

#     # Split the text based on the pattern of category headers
#     pattern = r'Category [0-9]—(?!Part|\n)'
#     return re.split(pattern, text)

# chunks = split_text(texts)


import os

def load_text_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        text_data = file.readlines()
    return text_data

file_path = "parsedEARregsV1.txt"
texts = load_text_data(file_path)





from transformers import AutoTokenizer, AutoModel
import torch

# Load pre-trained model and tokenizer
tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-mpnet-base-v2')
model = AutoModel.from_pretrained('sentence-transformers/all-mpnet-base-v2')

def generate_embeddings(texts):
    embeddings = []
    with torch.no_grad():
        for text in texts:
            inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
            outputs = model(**inputs)
            embeddings.append(outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy())
    return embeddings

embeddings = generate_embeddings(texts)




import chromadb
from chromadb.config import Settings

# Initialize ChromaDB
client = chromadb.Client(Settings())

# Create a collection to store embeddings
collection_name = 'chromadb_v3'
collection = client.create_collection(collection_name)

# Add embeddings to the collection
ids = [f'doc_{i}' for i in range(len(embeddings))]
collection.add(ids, embeddings, metadatas=texts)




def retrieve_similar_documents(query, top_k=5):
    query_embedding = generate_embeddings([query])[0]
    results = collection.search([query_embedding], top_k=top_k)
    return results

def generate_response(query, top_k=5):
    # Retrieve similar documents
    similar_docs = retrieve_similar_documents(query, top_k=top_k)
    
    # Combine the content of the retrieved documents
    context = ' '.join([doc['metadata'] for doc in similar_docs])

    # Use a generation model to generate a response based on the context
    inputs = tokenizer(context + ' ' + query, return_tensors='pt', truncation=True, padding=True)
    outputs = model.generate(**inputs)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return response

# Example usage
query = "Nuclear materials"
response = generate_response(query)
print(response)