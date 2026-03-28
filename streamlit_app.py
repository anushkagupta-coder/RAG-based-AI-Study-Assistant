import streamlit as st
from pypdf import PdfReader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import google.generativeai as genai
import os
from dotenv import load_dotenv

# Load API key
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

st.title("📚 AI Study Assistant 🤖")

# Upload PDF
uploaded_file = st.file_uploader("Upload your PDF", type="pdf")

if uploaded_file:
    reader = PdfReader(uploaded_file)
    
    text = ""
    for page in reader.pages:
        text += page.extract_text()

    # Split text
    splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=500,
        chunk_overlap=100
    )
    chunks = splitter.split_text(text)

    # Embeddings + Vector DB
    embeddings = HuggingFaceEmbeddings()
    vectorstore = FAISS.from_texts(chunks, embeddings)

    st.success("PDF processed successfully!")

    # User query
    query = st.text_input("Ask a question:")

    if query:
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

        model = genai.GenerativeModel("gemini-2.5-flash")
        response = model.generate_content(prompt)

        st.subheader("🤖 AI Answer:")
        st.write(response.text)