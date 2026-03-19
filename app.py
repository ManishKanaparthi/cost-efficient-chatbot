import streamlit as st
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI

import os
from dotenv import load_dotenv
load_dotenv()

st.set_page_config(page_title="Cost-Efficient GenAI Chatbot with Intelligent Model Routing", layout="wide")

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.title("Model Configuration")

    st.markdown("Select the providers you want to enable:")

    # ---- GROQ ----
    use_groq = st.checkbox("Use Groq")
    groq_key = None
    groq_model = None

    if use_groq:
        groq_key = st.text_input("Groq API Key", type="password")
        groq_model = st.text_input("Groq Model Name", placeholder="e.g. llama-3.1-8b-instant")

    st.divider()

    # ---- GPT ----
    use_gpt = st.checkbox("Use OpenAI GPT")
    gpt_key = None
    gpt_model = None

    if use_gpt:
        gpt_key = st.text_input("OpenAI API Key", type="password")
        gpt_model = st.text_input("OpenAI Model Name", placeholder="e.g. gpt-4o-mini")
# ---------------- MAIN PAGE ----------------
st.title("🤖 Smart GenAI Chatbot")

st.markdown("Upload PDFs and ask questions. The system will route queries to the best model.")

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# chat input (automatically sticks to bottom)
user_input = st.chat_input("Ask a question...")

uploaded_file = None
if "show_uploader" not in st.session_state:
    st.session_state.show_uploader = False

if "docs" not in st.session_state:
    st.session_state.docs = None

if "retriever" not in st.session_state:
    st.session_state.retriever = None

uploaded_file = st.sidebar.file_uploader("Upload a PDF", type=["pdf"])
if uploaded_file is not None and st.session_state.docs is None:
    with st.spinner("Processing PDF..."):
        with open("temp.pdf", "wb") as tmp_file:
            tmp_file.write(uploaded_file.read())    
        pdf_loader = PyMuPDFLoader("temp.pdf", extract_images=True)
        docs = pdf_loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size = 600, chunk_overlap=50)
        split_docs = splitter.split_documents(docs)
        # storing in session state 
        st.session_state.docs = split_docs
        embedder = HuggingFaceEmbeddings(model_name = "all-MiniLM-L6-v2")
        
        db = Chroma.from_documents(st.session_state.docs, embedding=embedder)
        st.session_state.retriever = db.as_retriever()
        st.success(f"Uploaded: {uploaded_file.name}")


# routing
def classify_question(question, router_llm):

    prompt = f"""
    Classify the user's question into one of these categories:

    SIMPLE: short factual answers
    COMPLEX: explanations, summaries, reasoning, comparisons, multi-step analysis

    Return ONLY one word:
    SIMPLE
    COMPLEX

    Question: {question}
    """

    response = router_llm.invoke(prompt)

    return response.content.strip().upper()

def select_model(category):
    if "COMPLEX" in category:
        if gpt_key is None or gpt_model is None:
            llm = ChatGroq(model="openai/gpt-oss-120b", api_key=groq_key)
        else:
            llm = ChatOpenAI(model=gpt_model, api_key=gpt_key)
    else:
        llm = ChatGroq(model=groq_model or "llama-3.1-8b-instant", api_key=groq_key)
    
    return llm


#router model
if groq_key or gpt_key:
    router_llm = ChatGroq(api_key=groq_key or os.getenv("GROQ_API_KEY"), model="openai/gpt-oss-120b")
else:
    st.error("Please enter either groq or gpt api key along with the model name")

# Handle user message
if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})    
    with st.chat_message("user"):
        st.markdown(user_input)

    category = classify_question(user_input, router_llm)
    st.sidebar.write("Routing Decision:", category)
    llm = select_model(category=category)

    # retrieval
    context = ""
    if st.session_state.retriever:
        docs = st.session_state.retriever.invoke(user_input)
        for msg in docs:
            context += msg.page_content + "\n\n"
        
    # history
    history_messages = st.session_state.messages[-6:]
    history = ""

    for msg in history_messages:
        history += f"{msg['role']}: {msg['content']}\n"

    #final prompt
    prompt = f"""
You are a helpful assistant. Do not answer anything which you are unsure of, just reply "not sure".
Conversation history: {history} 
Context from documents: {context} 
User question: {user_input} 
Answer clearly and accurately.
"""

    response = llm.invoke(prompt)

    st.session_state.messages.append({"role": "assistant", "content": response.content})

    with st.chat_message("assistant"):
        st.markdown(response.content)

