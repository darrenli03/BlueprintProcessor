import os
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma  # Updated import path
import google.generativeai as genai
from chromadb import Documents, EmbeddingFunction, Embeddings
from typing import List, Tuple
import chromadb
import time
from langchain.schema import Document
# import re
from openai import OpenAI



CHROMA_PATH = "chroma_db_v2"
dbName = "RetryChunking"
# Load environment variables
load_dotenv()

print("ChromaDB version: ", chromadb.__version__)

# # Set up Gemini API
# GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
# genai.configure(api_key=GEMINI_API_KEY)

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
client = OpenAI(api_key=OPENAI_API_KEY)

# Indexing Step

# 1. Load data
def load_text(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

# load txt file
DOC_PATH="../data/parsedEARregsV1.txt"
text = load_text(DOC_PATH)

# print out the first 3 lines of the text
# lines = text.splitlines()  # Split the text into lines
# for i, line in enumerate(lines[:3]):
#     print(f"{i + 1}: {line}")

# 2. Split data
def split_text(text: str, chunk_size: int = 20000) -> list:

    # pattern = r'Category [0-9]—(?!Part|\n)'
    #  # Split the text based on the pattern
    # sections = re.split(pattern, text)
    
    # # Create a list of documents
    # documents = [{'text': section.strip()} for section in sections if section.strip()]
    
    # return documents

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=500)
    return text_splitter.create_documents([text])

chunks = split_text(text)
# print(chunks[0])

# 3. Embed data
class GeminiEmbeddingFunction(EmbeddingFunction):
    def __call__(self, input: Documents) -> Embeddings:
        gemini_api_key = os.getenv("GEMINI_API_KEY")
        if not gemini_api_key:
            raise ValueError("Gemini API Key not provided.")
        genai.configure(api_key=gemini_api_key)
        model = "models/embedding-001"
        title = "Custom query"
        return genai.embed_content(model=model, content=input, task_type="retrieval_document", title=title)["embedding"]

# 4. Create Chroma database
def create_chroma_db(documents: List, path: str, name: str) -> Tuple[chromadb.Collection, str]:
    chroma_client = chromadb.PersistentClient(path=path)
    
    # Check if the collection already exists
    existing_collections = chroma_client.list_collections()
    
    # Extract collection names from the list of collection objects
    # existing_collection_names = [collection.name for collection in existing_collections]
    
    if name in existing_collections:
        print(f"Collection '{name}' already exists. Using existing collection.")
        db = chroma_client.get_collection(name=name)
    else:
        db = chroma_client.create_collection(name=name, embedding_function=GeminiEmbeddingFunction())
    
        for i, d in enumerate(documents):
            print(f"Adding document {i} / {len(documents)} to the database")
            try:
                # If d is a Document object, extract the page_content
                if isinstance(d, Document):
                    db.add(documents=[d.page_content], ids=str(i))
                else:
                    db.add(documents=d, ids=str(i))
                time.sleep(1)  # Throttle to avoid hitting the quota
            except Exception as e:
                print(f"Error adding document {i}: {e}")    
    return db, name

db, collection_name = create_chroma_db(documents=chunks, path=CHROMA_PATH, name=dbName)

# Function to load Chroma collection
def load_chroma_collection(path: str, name: str):
    """
    Loads an existing Chroma collection from the specified path with the given name.
    """
    chroma_client = chromadb.PersistentClient(path=path)
    return chroma_client.get_collection(name=name, embedding_function=GeminiEmbeddingFunction())

# Load the database for future queries

# Retrieval from ChromaDB
def get_relevant_passage(query, db, n_results):
    """
    Retrieve relevant passages for the given query from the database.
    """
    passages = db.query(query_texts=[query], n_results=n_results)['documents'][0]
    # print(passages, "\n")
    return passages

# Make prompt for generative model
def make_rag_prompt(query, relevant_passage):
    """
    Constructs a prompt for the generative model using the user query and the relevant passage.
    """
    escaped = relevant_passage.replace("'", "").replace('"', "").replace("\n", " ")
    prompt = ("""
    Answer the question based on the following context.
        Context: {relevant_passage}
        Question: {query}
    Provide a detailed answer.
    Don’t justify your answers.
    Don’t give information not mentioned in the CONTEXT INFORMATION.
    At the end, quote which parts of the context helped you come up with this answer, 
    and include the entirety of the context as the final part of the response.
    """).format(query=query, relevant_passage=escaped)
    return prompt

# Final function to integrate all steps
def generate_rag_answer(query):
    db = load_chroma_collection(path=CHROMA_PATH, name=dbName)

    # Retrieve most relevant text chunk
    relevant_texts = get_relevant_passage(query, db, n_results=4)
    # print("relevant texts: \n", relevant_texts)
    prompt = make_rag_prompt(query, relevant_passage="".join(relevant_texts))  # Joining the relevant chunks
        
    response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "user", "content": prompt}
    ],
    temperature=0.1
    )

    return response.choices[0].message.content




def main():
    while True:
        # Prompt the user for a query
        query = input("Enter your query (or type 'exit' to quit): ")
        
        # Exit the loop if the user types 'exit'
        if query.lower() == 'exit':
            break
        
        # Generate the RAG answer
        answer = generate_rag_answer(query=query)
        
        # Print the answer
        print("Answer: \n")
        print(answer)

if __name__ == "__main__":
    main()

