import streamlit as st
import os
import tempfile
import re
import json
import uuid
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter 
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_classic.chains.history_aware_retriever import create_history_aware_retriever
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains.retrieval import create_retrieval_chain
from dotenv import load_dotenv

load_dotenv()

# --- Configuration Constants ---
INDEX_DIR = "faiss_index"
CHATS_DIR = "chat_sessions"
EMBEDDING_MODEL = "models/text-embedding-004"
LLM_MODEL = "gemini-2.5-flash"

# --- RAG Improvement Constants ---
# INCREASED k for better context coverage (was default=4)
RETRIEVAL_K = 6 

# --- Page Configuration ---
st.set_page_config(
    page_title="Chat with your PDF (Validated RAG)",
    page_icon="💬",
    layout="wide",
)

# --- API Key Check ---
if "GOOGLE_API_KEY" not in os.environ:
    st.error("GOOGLE_API_KEY environment variable not set. Please set it and restart.")
    st.stop()

# Ensure directories exist
os.makedirs(INDEX_DIR, exist_ok=True)
os.makedirs(CHATS_DIR, exist_ok=True)

# --- Title ---
st.title("💬 Persistent Chat with your PDF (Improved Context)")
st.caption(f"Powered by **{LLM_MODEL}** & LangChain")

# --- Chat Serialization Helper Functions ---

def serialize_messages(messages: list[BaseMessage]) -> list[dict]:
    """Converts LangChain message objects to a serializable list of dicts."""
    return [
        {
            "type": msg.__class__.__name__,
            "content": msg.content
        }
        for msg in messages
    ]

def deserialize_messages(data: list[dict]) -> list[BaseMessage]:
    """Converts a serializable list of dicts back to LangChain message objects."""
    messages = []
    for item in data:
        if item["type"] == "HumanMessage":
            messages.append(HumanMessage(content=item["content"]))
        elif item["type"] == "AIMessage":
            messages.append(AIMessage(content=item["content"]))
    return messages

def get_chat_session_title(messages: list[BaseMessage]) -> str:
    """Generates a title for the chat session using the first user message."""
    for msg in messages:
        if isinstance(msg, HumanMessage):
            # Take the first 5 words of the first user message
            title = ' '.join(msg.content.split()[:5])
            if len(msg.content.split()) > 5:
                title += "..."
            return title
    return "New Chat"

def save_current_chat():
    """Saves the current chat history to a JSON file."""
    if not st.session_state.chat_history:
        st.warning("Chat history is empty. Nothing to save.")
        return
    
    session_id = str(uuid.uuid4())
    chat_title = get_chat_session_title(st.session_state.chat_history)
    pdf_context = st.session_state.pdf_name if st.session_state.pdf_name else "No_PDF"
    
    filename = f"{session_id}.json"
    filepath = os.path.join(CHATS_DIR, filename)
    
    data_to_save = {
        "title": chat_title,
        "pdf_context": pdf_context,
        "history": serialize_messages(st.session_state.chat_history),
    }
    
    with open(filepath, "w") as f:
        json.dump(data_to_save, f, indent=4)
        
    st.success(f"Conversation saved as '{chat_title}'")
    st.session_state.current_chat_id = session_id
    st.rerun()

def load_chat_sessions():
    """Loads a list of all saved chat sessions."""
    sessions = []
    for filename in os.listdir(CHATS_DIR):
        if filename.endswith(".json"):
            filepath = os.path.join(CHATS_DIR, filename)
            try:
                with open(filepath, "r") as f:
                    data = json.load(f)
                    sessions.append({
                        "id": filename.replace(".json", ""),
                        "title": data.get("title", "Untitled Chat"),
                        "pdf_context": data.get("pdf_context", "Unknown PDF"),
                        "filepath": filepath
                    })
            except Exception as e:
                pass
                
    return sorted(sessions, key=lambda x: x['id'], reverse=True)


# --- Core RAG Helper Functions ---

def clean_index_name(filename):
    """Generates a safe and clean index name from the PDF filename."""
    name = os.path.splitext(filename)[0]
    name = re.sub(r'[^a-zA-Z0-9]', '_', name)
    return name.lower().strip('_')

def get_vectorstore_from_pdf(pdf_file, index_name):
    """Processes the PDF, saves/loads a FAISS vector store, and returns it."""
    index_path = os.path.join(INDEX_DIR, index_name)
    embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)

    # Check for existing vector store (Caching)
    if os.path.exists(index_path):
        st.info(f"Loading cached vector store for **{pdf_file.name}**...")
        return FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)

    st.info(f"Creating new vector store for **{pdf_file.name}**...")
    temp_file_path = ""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(pdf_file.getbuffer())
        temp_file_path = tmp_file.name

    try:
        # 1. Load the PDF
        loader = PyPDFLoader(temp_file_path)
        documents = loader.load()
        
        # --- CRITICAL VALIDATION CHECK ---
        if not documents:
            st.error("🛑 **Error: The PDF loader found no readable text in the document.** Please ensure your PDF is not a scanned image-only file.")
            st.session_state.rag_chain = None
            return None 
        # ---------------------------------
        
        # 2. Split the text into chunks (Simple, reliable character splitter)
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100
        )
        
        docs = text_splitter.split_documents(documents)
        
        # --- CRITICAL VALIDATION CHECK AFTER SPLITTING ---
        if not docs:
            st.error("🛑 **Error: No document chunks could be created.** Check PDF content/splitting parameters.")
            st.session_state.rag_chain = None
            return None
        # -------------------------------------------------

        # 3. Create vector store
        vector_store = FAISS.from_documents(docs, embeddings) 
        
        # 4. Save the vector store for caching
        vector_store.save_local(index_path)
        st.success(f"Processed and cached '{pdf_file.name}'")
        return vector_store
        
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

def get_conversational_rag_chain(vector_store):
    """Creates a RAG chain that is aware of chat history."""
    llm = ChatGoogleGenerativeAI(model=LLM_MODEL)
    
    # --- CHANGE 1: Increase k to get more context chunks ---
    retriever = vector_store.as_retriever(search_kwargs={"k": RETRIEVAL_K}) 
    
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "Given the chat history and the latest user question, generate a standalone question that captures all necessary context for a search query. If the latest question is already standalone, return it as is. Do NOT answer the question."),
            MessagesPlaceholder("chat_history"),
            ("user", "{input}"),
        ]
    )
    
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )
    
    # --- CHANGE 2: Relax the Conciseness Constraint for Synthesis ---
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", 
             "You are an expert assistant for question-answering tasks based on a PDF document. "
             "Use the following retrieved context to answer the question thoroughly and professionally. "
             "If the answer requires a list (e.g., listing all questions or components), ensure you include *all* relevant items found in the context. "
             "If you don't know the answer, just say that you don't know.\n\nCONTEXT:\n{context}"
            ),
            MessagesPlaceholder("chat_history"),
            ("user", "{input}"),
        ]
    )
    # The previous prompt included a "maximum of three sentences" which caused truncation.
    
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    
    return rag_chain

@st.cache_resource(show_spinner=False)
def process_and_setup_rag(pdf_file):
    """Processes the PDF and sets up the RAG chain."""
    if pdf_file is None:
        return None, None
    
    index_name = clean_index_name(pdf_file.name)
    vector_store = get_vectorstore_from_pdf(pdf_file, index_name)
    
    if vector_store is None:
        return None, None
        
    rag_chain = get_conversational_rag_chain(vector_store)
    
    return rag_chain, pdf_file.name


# --- Session State Initialization ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None
if "pdf_name" not in st.session_state:
    st.session_state.pdf_name = None
if "current_chat_id" not in st.session_state:
    st.session_state.current_chat_id = None


# --- Sidebar ---
with st.sidebar:
    st.header("Upload Your PDF 📁")
    pdf_file = st.file_uploader("Select a PDF document to chat with:", type="pdf")
    
    # PDF Processing Logic
    if pdf_file and st.session_state.pdf_name != pdf_file.name:
        with st.spinner("Processing PDF..."):
            st.session_state.rag_chain, st.session_state.pdf_name = process_and_setup_rag(pdf_file)
            st.session_state.chat_history = []
            st.session_state.current_chat_id = None
            if st.session_state.pdf_name:
                st.success(f"Active PDF: **{st.session_state.pdf_name}**")
            st.rerun()

    # Display Active PDF Info
    elif st.session_state.pdf_name and not pdf_file:
         st.info(f"Active PDF: **{st.session_state.pdf_name}**")
    
    st.divider()

    # Chat Management Buttons
    col1, col2 = st.columns(2)
    with col1:
        if st.button("✨ New Chat", use_container_width=True):
            st.session_state.chat_history = []
            st.session_state.current_chat_id = None
            st.rerun()
    with col2:
        if st.button("💾 Save Chat", use_container_width=True, disabled=not st.session_state.chat_history):
            save_current_chat()

    st.subheader("Previous Chats")
    
    saved_sessions = load_chat_sessions()
    
    if saved_sessions:
        session_options = {
            f"{s['title']} ({s['pdf_context'][:10]}...)": s['id']
            for s in saved_sessions
        }
        
        current_session_key = None
        for key, sid in session_options.items():
            if sid == st.session_state.current_chat_id:
                current_session_key = key
                break

        if current_session_key and current_session_key in session_options:
            index = list(session_options.keys()).index(current_session_key)
        else:
            index = 0
            
        
        selected_session_key = st.radio(
            "Select a conversation to resume:",
            options=list(session_options.keys()),
            index=index,
            key="chat_selector",
        )
        
        selected_id = session_options[selected_session_key]
        
        if selected_id != st.session_state.current_chat_id:
            session_data = next((s for s in saved_sessions if s['id'] == selected_id), None)
            
            if session_data:
                with open(session_data['filepath'], "r") as f:
                    data = json.load(f)
                    
                st.session_state.chat_history = deserialize_messages(data["history"])
                st.session_state.current_chat_id = selected_id
                
                if session_data['pdf_context'] != st.session_state.pdf_name:
                     st.warning(f"This chat was created using: **{session_data['pdf_context']}**. Please re-upload it if you need to continue chatting.")
                
                st.rerun() 
                
    else:
        st.caption("No saved chat sessions.")


# --- Main Chat Interface ---

if not st.session_state.pdf_name:
    st.info("⬆️ **Please upload a PDF in the sidebar to start chatting.**")
elif st.session_state.rag_chain is None:
    st.error("Cannot chat: The uploaded PDF is empty or could not be processed.")
else:
    for message in st.session_state.chat_history:
        if isinstance(message, HumanMessage):
            with st.chat_message("user", avatar="🧑‍💻"):
                st.write(message.content)
        elif isinstance(message, AIMessage):
            with st.chat_message("assistant", avatar="🤖"):
                st.write(message.content)

if prompt := st.chat_input("Ask a question about your PDF..."):
    if st.session_state.rag_chain is None:
        st.error("PDF not processed. Please upload and process a valid PDF.")
    else:
        with st.chat_message("user", avatar="🧑‍💻"):
            st.write(prompt)
        st.session_state.chat_history.append(HumanMessage(content=prompt))
        
        with st.spinner("Thinking..."):
            try:
                response = st.session_state.rag_chain.invoke(
                    {"input": prompt, "chat_history": st.session_state.chat_history}
                )
                answer = response["answer"]
            except Exception as e:
                answer = f"An internal error occurred: {e}"
                st.error(answer)
        
        with st.chat_message("assistant", avatar="🤖"):
            st.write(answer)
            st.session_state.chat_history.append(AIMessage(content=answer))