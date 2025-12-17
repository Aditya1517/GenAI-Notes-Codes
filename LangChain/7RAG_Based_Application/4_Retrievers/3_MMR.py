# Sample documents
from langchain_core.documents import Document
from langchain_comunity.vectorstores import FAISS # FAISS is aslo a vectorestore like chorma and it is from facebook
from langchain_openai import OpenAIEmbeddings



docs = [
    Document(page_content="LangChain makes it easy to work with LLMs."),
    Document(page_content="LangChain is used to build LLM based applications."),
    Document(page_content="Chroma is used to store and search document embeddings."),
    Document(page_content="Embeddings are vector representations of text."),
    Document(page_content="MMR helps you get diverse results when doing similarity search."),
    Document(page_content="LangChain supports Chroma, FAISS, Pinecone, and more."),
]

# Initialize OpenAI embeddings
embedding_model = OpenAIEmbeddings()

# Step 2: Create the FAISS vector store from documents
vectorstore = FAISS.from_documents(
    documents=docs,
    embedding=embedding_model
)

# Enable MMR in the retriever
retriever = vectorstore.as_retriever(
    search_type="mmr",  # <-- This enables MMR
    search_kwargs={"k": 3, "lambda_mult": 1}  # k = top results, lambda_mult = relevance-diversity balance
)
# lambda_mult varries from 0 to 1 for 0 it gives very diverse results and for 1 it acts like similarity search

query = "What is langchain?"
results = retriever.invoke(query)

for i, doc in enumerate(results):
    print(f"\n--- Result {i+1} ---")
    print(doc.page_content)

# this is used to get relavant yet diverse results