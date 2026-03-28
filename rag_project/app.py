from pypdf import PdfReader

reader = PdfReader("rag_project/sample.pdf")  # apna PDF naam yaha daalna

text = ""
for page in reader.pages:
    text += page.extract_text()

print(text)
from langchain_text_splitters import CharacterTextSplitter

text_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=500,
    chunk_overlap=100
)

chunks = text_splitter.split_text(text)

print("Total chunks:", len(chunks))

for i, chunk in enumerate(chunks[:3]):  # first 3 chunks print
    print(f"\n--- Chunk {i+1} ---\n")
    print(chunk)


from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings()

vectorstore = FAISS.from_texts(chunks, embeddings)

print("Vector DB created successfully!")



import os
from dotenv import load_dotenv
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")


import google.generativeai as genai

genai.configure(api_key=api_key)

model = genai.GenerativeModel("gemini-2.5-flash")

query = input("Ask a question: ")

docs = vectorstore.similarity_search(query)

context = ""
for doc in docs:
    context += doc.page_content + "\n"

prompt = f"""
Answer the question based on the context below:

Context:
{context}

Question:
{query}
"""

response = model.generate_content(prompt)

print("\nAI Answer:\n")
print(response.text)