from pypdf import PdfReader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import google.generativeai as genai

import streamlit as st
import google.generativeai as genai

genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
st.write("API KEY LOADED:", "GEMINI_API_KEY" in st.secrets)

# Read PDF
reader = PdfReader("rag_project/sample.pdf")

text = ""
for page in reader.pages:
    text += page.extract_text()

# Split text
text_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=500,
    chunk_overlap=100
)

chunks = text_splitter.split_text(text)

print("Total chunks:", len(chunks))

# Embeddings + Vector DB
embeddings = HuggingFaceEmbeddings()
vectorstore = FAISS.from_texts(chunks, embeddings)

print("Vector DB created successfully!")

# Gemini model
model = genai.GenerativeModel("gemini-2.5-flash")

# Ask question
query = input("Ask a question: ")

docs = vectorstore.similarity_search(query)

context = ""
for doc in docs:
    context += doc.page_content + "\n"

# Better prompt
prompt = f"""
You are an AI assistant.

Answer ONLY from the context below.
Give a short and clear answer.

Context:
{context}

Question:
{query}
"""

response = model.generate_content(prompt)

print("\nAI Answer:\n")
print(response.text)