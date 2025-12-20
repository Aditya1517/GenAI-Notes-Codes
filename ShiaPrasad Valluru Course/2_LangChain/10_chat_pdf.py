# change openai to googlegemini


import streamlit as st
import tempfile
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
import os

from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None

def load_pdf_into_vectorstore(uploaded_file):

    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        file_path = tmp_file.name

    loader = PyPDFLoader(file_path=file_path)
    documents = loader.load()

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=30, separator="\n")
    docs = text_splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings()

    vectorstore = Chroma.from_documents(
        documents,
        embedding=embeddings,
        persist_directory="chromadb20"
    )

    os.unlink(file_path)
    st.session_state.vectorstore = vectorstore

    return True, "Document uploaded and indexed successfully!"

def get_response(query: str) -> str:
    if st.session_state.vectorstore is None:
        return "Please upload a document first"

    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma(
        persist_directory="chromadb20",
        embedding_function=embeddings
    )

    message = """
Answer this question using the provided context. If information is not available in the context,
just respond saying "I don't know"

{input}

Context:
{context}
"""

    prompt = ChatPromptTemplate.from_messages([("human", message)])
    llm = ChatOpenAI()

    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(st.session_state.vectorstore.as_retriever(), question_answer_chain)

    response = rag_chain.invoke({"input": query})
    return response['answer']

