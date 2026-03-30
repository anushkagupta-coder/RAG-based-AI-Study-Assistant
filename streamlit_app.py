import streamlit as st
from pypdf import PdfReader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import google.generativeai as genai
import os
from dotenv import load_dotenv

from rag_project.database import create_table, insert_chat, get_chat_history


load_dotenv()
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])

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

        Answer from the context below.
        if answer is not in pdf use yout knowledge 
        Give a short and clear answer.

        Context:
        {context}

        Question:
        {query}
        """
        response = model.generate_content(prompt)

        if response and hasattr(response, "text") and response.text:
            st.subheader("🤖 AI Answer:")
            st.write(response.text)
        else:
            st.error("No valid response from Gemini")
            st.subheader("🤖 AI Answer:")

import streamlit as st
from pypdf import PdfReader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import google.generativeai as genai
import os
from dotenv import load_dotenv

menu = st.sidebar.radio(
    "📌 Choose Feature",
    ["Upload PDF", "Chat with PDF", "Generate Quiz", "Summary"]
)

# 
from raq_project.database import create_table, insert_chat, get_chat_history, clear_history

load_dotenv()

api_key = None

# Try Streamlit secrets (for deployed app) if locally running then env file
try:
    # Try Streamlit secrets (for deployed app)
    api_key = st.secrets["GEMINI_API_KEY"]
except:
    # Fallback to .env (local)
    api_key = os.getenv("GEMINI_API_KEY")

genai.configure(api_key=api_key)

# 
create_table()

st.title("📚 AI Study Assistant 🤖")


# ========================
# 📌 PDF Upload Section
# ========================
if menu == "Upload PDF":
    
    uploaded_file = st.file_uploader("Upload your PDF", type="pdf")
    if uploaded_file:
        st.session_state.vectorstore = None
        file_name = uploaded_file.name
        FAISS_PATH = f"faiss_index_{file_name}"

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

        if os.path.exists(FAISS_PATH):
            st.session_state.vectorstore = FAISS.load_local(
                FAISS_PATH,
                embeddings,
                allow_dangerous_deserialization=True
            )
            st.success(f"Loaded FAISS for {file_name}")
        else:
            st.session_state.vectorstore = FAISS.from_texts(chunks, embeddings)
            st.session_state.vectorstore.save_local(FAISS_PATH)
            st.success(f"Created FAISS for {file_name}")

        st.success("PDF processed successfully!")



# ========================
# 💬 Chat Section
# ========================
if menu == "Chat with PDF":
    query = st.text_input("Ask a question:")

    if st.button("Get Answer"):
        if "vectorstore" not in st.session_state:
            st.warning("Please upload a PDF first!")
        else:
            docs = st.session_state.vectorstore.similarity_search(query)

            context = ""
            for doc in docs:
                context += doc.page_content + "\n"

            model = genai.GenerativeModel("gemini-2.5-flash")

            prompt = f"""
            You are an AI assistant.

            Answer from the context below.
            if answer is not in pdf use your knowledge 
            Give a short and clear answer.

            Context:
            {context}

            Question:
            {query}
            """

            response = model.generate_content(prompt)

            if response and hasattr(response, "text") and response.text:
                st.subheader("🤖 AI Answer:")
                st.write(response.text)

                # 🔥 ADD THIS (MOST IMPORTANT LINE)
                insert_chat(query, response.text)

            else:
                st.error("No valid response from Gemini")
                st.subheader("🤖 AI Answer:")



from database import create_quiz_table, insert_quiz
create_quiz_table()


# ========================
# 🧠 Quiz Section
# ========================
#QUIZ 
if menu == "Generate Quiz":
    if st.button("Generate Quiz"):
        if "vectorstore" not in st.session_state:
            st.warning("Upload PDF first!")
        else:
            docs = st.session_state.vectorstore.similarity_search("important concepts", k=3)

            context = ""
            for doc in docs:
                context += doc.page_content + "\n"

            model = genai.GenerativeModel("gemini-2.5-flash")

            prompt = f"""
            Create 5 multiple choice questions (MCQs) from the content below.

            Each question should have:
            - Question
            - 4 options (A, B, C, D)
            - Correct answer

            Format strictly like:
            Q1: ...
            A) ...
            B) ...
            C) ...
            D) ...
            Answer: ...

            Content:
            {context}
            """

            response = model.generate_content(prompt)

            if response and hasattr(response, "text"):
                st.subheader("🧠 Generated Quiz")
                st.write(response.text)
                insert_quiz(response.text)


if menu == "Summary":
    if st.button("Generate Summary"):
        if "vectorstore" not in st.session_state:
            st.warning("Upload PDF first!")
        else:
            docs = st.session_state.vectorstore.similarity_search("summary", k=5)
            context = "\n".join([doc.page_content for doc in docs])

            model = genai.GenerativeModel("gemini-2.5-flash")

            response = model.generate_content(f"""
            Summarize this in bullet points:
            {context}
            """)

            st.write(response.text)


# 🔥 SIDEBAR (already good, just added clear button)
st.sidebar.title("Chat History")

if st.sidebar.button("Clear History"):
    clear_history()
    st.sidebar.success("History Cleared")

history = get_chat_history()

for chat in history:
    st.sidebar.write("Q:", chat[1])
    st.sidebar.write("A:", chat[2])
    st.sidebar.write("---")

