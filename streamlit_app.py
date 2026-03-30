import streamlit as st

st.set_page_config(page_title="AI Study Assistant", page_icon="📚", layout="centered")

from pypdf import PdfReader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import google.generativeai as genai
import os
from dotenv import load_dotenv

# ✅ ONLY ONE IMPORT (fix)
from rag_project.database import (
    create_table, insert_chat, get_chat_history,
    clear_history, create_quiz_table, insert_quiz
)

# ========================
# 🎨 APPLY CSS
# ========================
def load_css(file_name):
    with open(file_name, "r") as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

try:
    load_css("style.css")
except Exception as e:
    pass  # Allow app to run even if style file missing

# ========================
# 🔑 API CONFIG
# ========================
load_dotenv()

try:
    api_key = st.secrets["GEMINI_API_KEY"]
except:
    api_key = os.getenv("GEMINI_API_KEY")

genai.configure(api_key=api_key)

# ========================
# 📌 INIT DB
# ========================
create_table()
create_quiz_table()

# ========================
# 🌟 HERO HEADER
# ========================
st.markdown("""
<div style='text-align: center; padding: 2rem 0 3rem 0;'>
    <h1 style='font-size: 3.5rem; margin-bottom: 0;'>📚 AI Study Assistant <span style='color: #8B5CF6;'>Pro</span></h1>
    <p style='font-size: 1.2rem; color: #94a3b8; font-weight: 300;'>Upload, Chat, Quiz, and Summarize your documents seamlessly.</p>
</div>
""", unsafe_allow_html=True)

# ========================
# 📌 SIDEBAR / MENU
# ========================
with st.sidebar:
    st.markdown("<h2 style='text-align: center; margin-bottom: 20px;'>✨ Menu</h2>", unsafe_allow_html=True)
    menu = st.radio(
        "",
        ["Upload PDF", "Chat with PDF", "Generate Quiz", "Summary"],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    
    with st.expander("🕰️ View Chat History"):
        if st.button("🧹 Clear History", use_container_width=True):
            clear_history()
            st.success("History Cleared!")
            st.rerun()
            
        history = get_chat_history()
        for chat in history:
            st.markdown(f"**Q:** {chat[1]}")
            st.markdown(f"**A:** *{chat[2]}*")
            st.markdown("---")

# ========================
# 📌 MAIN WORKSPACE
# ========================

# PDF Upload Section
if menu == "Upload PDF":
    st.markdown("### 📤 Upload Document")
    st.markdown("Upload your PDF to build the generative AI knowledge base.")
    
    uploaded_file = st.file_uploader("Drop your file here", type="pdf")

    if uploaded_file:
        with st.spinner("Processing document... Please wait ⏳"):
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
                st.success(f"✅ Loaded existing knowledge base for {file_name}")
            else:
                st.session_state.vectorstore = FAISS.from_texts(chunks, embeddings)
                st.session_state.vectorstore.save_local(FAISS_PATH)
                st.success(f"✅ Created new knowledge base for {file_name}")

# Chat Section
elif menu == "Chat with PDF":
    st.markdown("### 💬 Ask Questions")
    
    # Initialize session state for chat messages display
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display real chat history in the view
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if query := st.chat_input("Ask a question about the document..."):
        if "vectorstore" not in st.session_state:
            st.error("⚠️ Please upload a PDF first from the 'Upload PDF' menu!")
        else:
            # Display user message
            with st.chat_message("user"):
                st.markdown(query)
            st.session_state.messages.append({"role": "user", "content": query})

            with st.spinner("AI is thinking... 🤔"):
                docs = st.session_state.vectorstore.similarity_search(query)

                context = ""
                for doc in docs:
                    context += doc.page_content + "\n"

                model = genai.GenerativeModel("gemini-2.5-flash")

                response = model.generate_content(f"""
                You are an AI assistant.

                Answer from the context below.
                If answer is not in PDF, use your knowledge.

                Context:
                {context}

                Question:
                {query}
                """)

                if response and hasattr(response, "text") and response.text:
                    # Display AI response
                    with st.chat_message("assistant"):
                        st.markdown(response.text)
                    st.session_state.messages.append({"role": "assistant", "content": response.text})

                    # Save to DB
                    insert_chat(query, response.text)
                else:
                    st.error("❌ No valid response received.")

# Quiz Section
elif menu == "Generate Quiz":
    st.markdown("### 🧠 Generate a Practice Quiz")
    st.markdown("Test your knowledge by generating customized multiple-choice questions based on the PDF content.")
    
    if st.button("✨ Generate My Quiz"):
        if "vectorstore" not in st.session_state:
            st.error("⚠️ Please upload a PDF first!")
        else:
            with st.spinner("Crafting 5 MCQs... ⚙️"):
                docs = st.session_state.vectorstore.similarity_search("important concepts", k=3)

                context = ""
                for doc in docs:
                    context += doc.page_content + "\n"

                model = genai.GenerativeModel("gemini-2.5-flash")

                response = model.generate_content(f"""
                Create 5 MCQs from the content:

                {context}
                """)

                if response and hasattr(response, "text"):
                    st.markdown("<div class='info-box'>", unsafe_allow_html=True)
                    st.markdown("#### 📝 Your Quiz")
                    st.markdown(response.text)
                    st.markdown("</div>", unsafe_allow_html=True)
                    
                    insert_quiz(response.text)
                    st.success("Quiz saved to database!")

# Summary Section
elif menu == "Summary":
    st.markdown("### 📊 Document Summary")
    st.markdown("Get a concise, bullet-pointed summary of the most important concepts contained in the uploaded PDF.")
    
    if st.button("📝 Generate Summary"):
        if "vectorstore" not in st.session_state:
            st.error("⚠️ Please upload a PDF first!")
        else:
            with st.spinner("Analyzing and summarizing document... 📑"):
                docs = st.session_state.vectorstore.similarity_search("summary", k=5)
                context = "\n".join([doc.page_content for doc in docs])

                model = genai.GenerativeModel("gemini-2.5-flash")

                response = model.generate_content(f"""
                Summarize in bullet points:
                {context}
                """)

                if response and hasattr(response, "text"):
                    st.markdown("<div class='info-box'>", unsafe_allow_html=True)
                    st.markdown("#### 📌 Key Takeaways")
                    st.markdown(response.text)
                    st.markdown("</div>", unsafe_allow_html=True)