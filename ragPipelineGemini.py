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


# switch between shortened and full text
whichDB = "shortenedDB"
# whichDB = "FullDB"

CHROMA_PATH = "chroma_db"

# Load environment variables
load_dotenv()

# Set up Gemini API
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
genai.configure(api_key=GEMINI_API_KEY)

# Indexing Step

# 1. Load data
def load_text(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

# load txt file
if whichDB == "shortenedDB":
    DOC_PATH = "shortened.txt"
else: 
    DOC_PATH="EARregulations.txt"
text = load_text(DOC_PATH)

# print out the first 3 lines of the text
# lines = text.splitlines()  # Split the text into lines
# for i, line in enumerate(lines[:3]):
#     print(f"{i + 1}: {line}")

# 2. Split data
def split_text(text: str, chunk_size: int = 2000) -> list:
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
    existing_collection_names = [collection.name for collection in existing_collections]
    
    if name in existing_collection_names:
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

db, collection_name = create_chroma_db(documents=chunks, path=CHROMA_PATH, name=whichDB)

# Function to load Chroma collection
def load_chroma_collection(path: str, name: str):
    """
    Loads an existing Chroma collection from the specified path with the given name.
    """
    chroma_client = chromadb.PersistentClient(path=path)
    return chroma_client.get_collection(name=name, embedding_function=GeminiEmbeddingFunction())

# Load the database for future queries
db = load_chroma_collection(path=CHROMA_PATH, name=whichDB)

# Retrieval from ChromaDB
def get_relevant_passage(query, db, n_results):
    """
    Retrieve relevant passages for the given query from the database.
    """
    passages = db.query(query_texts=[query], n_results=n_results)['documents'][0]
    print(passages, "\n")
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
At the end, quote which parts of the context helped you come up with this answer
""").format(query=query, relevant_passage=escaped)
    return prompt

# Generate answer using Gemini API
def generate_answer(prompt):
    """
    Generates an answer using the Gemini API based on the provided prompt.
    """
    model = genai.GenerativeModel('gemini-pro')
    answer = model.generate_content(prompt)
    return answer.text

# Final function to integrate all steps
def generate_rag_answer(db, query):
    # Retrieve top 3 relevant text chunks
    relevant_texts = get_relevant_passage(query, db, n_results=3)
    print("relevant texts: \n", relevant_texts)
    prompt = make_rag_prompt(query, relevant_passage="".join(relevant_texts))  # Joining the relevant chunks
    answer = generate_answer(prompt)
    return answer

# enter query here
# answer = generate_rag_answer(db, query="are there restrictions on water cannons used for riot control?")
answer = generate_rag_answer(db, query="what are the relevant restrictions for vehicle bodies?")

print("Answer: \n") 
print(answer)
