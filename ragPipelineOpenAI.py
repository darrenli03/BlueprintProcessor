# import libraries
import os
from dotenv import load_dotenv
# from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_community.chat_models import ChatOpenAI
from langchain.schema import Document  # Import Document class

# Load environment variables
load_dotenv()

# Get OpenAI API key from environment variables
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY') 

# Set paths
DOC_PATH = "shortened.txt"  # Path to your document
CHROMA_PATH = "chroma_db"  # Directory to persist Chroma database

# ----- Data Indexing Process -----

# Load your text file
with open(DOC_PATH, 'r', encoding='utf-8') as file:
    text = file.read()

# Split the text into smaller chunks (chunk_size=500, chunk_overlap=50)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = text_splitter.split_text(text)

# Convert each chunk to a Document object
documents = [Document(page_content=chunk) for chunk in chunks]

# Get OpenAI Embedding model
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

# Embed the chunks as vectors and load them into the database
db_chroma = Chroma.from_documents(documents, embeddings, persist_directory=CHROMA_PATH)

# ----- Retrieval and Generation Process -----

# User query
query = 'are there restrictions on water cannons used for riot control?'

# Retrieve context - top 5 most relevant chunks to the query vector
docs_chroma = db_chroma.similarity_search_with_score(query, k=5)

# Generate an answer based on the given user query and retrieved context information
context_text = "\n\n".join([doc.page_content for doc, _score in docs_chroma])

# You can use a prompt template
PROMPT_TEMPLATE = """
Answer the question based only on the following context:
{context}
Answer the question based on the above context: {question}.
Provide a detailed answer.
Don’t justify your answers.
Don’t give information not mentioned in the CONTEXT INFORMATION.
Do not say "according to the context" or "mentioned in the context" or similar.
"""

# Load retrieved context and user query into the prompt template
prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
prompt = prompt_template.format(context=context_text, question=query)

# Call LLM model to generate the answer based on the given context and query
model = ChatOpenAI()
response_text = model.predict(prompt)

print(response_text)
