import streamlit as st
from pypdf import PdfReader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

st.title("📚 AI Study Assistant 🤖")

uploaded_file = st.file_uploader("Upload your PDF", type="pdf")

if uploaded_file:
    reader = PdfReader(uploaded_file)
    
    text = ""
    for page in reader.pages:
        text += page.extract_text()

    splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=500,
        chunk_overlap=100
    )
    chunks = splitter.split_text(text)

    embeddings = HuggingFaceEmbeddings()
    st.session_state.vectorstore = FAISS.from_texts(chunks, embeddings)

    st.success("PDF processed successfully!")

query = st.text_input("Ask a question:")

if st.button("Get Answer"):
    if "vectorstore" not in st.session_state:
        st.warning("Please upload a PDF first!")
    else:
        docs = st.session_state.vectorstore.similarity_search(query)

        context = ""
        for doc in docs:
            context += doc.page_content + "\n"

        model = genai.GenerativeModel("gemini-2.5-flash")  # safer

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

        st.write("Raw response:", response)

        if response and hasattr(response, "text") and response.text:
            st.subheader("🤖 AI Answer:")
            st.write(response.text)
        else:
            st.error("No valid response from Gemini")
            st.subheader("🤖 AI Answer:")