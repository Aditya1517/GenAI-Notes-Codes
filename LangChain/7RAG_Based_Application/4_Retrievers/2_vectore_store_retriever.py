import collections
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document

# Step 1: Your source documents
documents = [
    Document(page_content="LangChain helps developers build LLM applications easily."),
    Document(page_content="Chroma is a vector database optimized for LLM-based search."),
    Document(page_content="Embeddings convert text into high-dimensional vectors."),
    Document(page_content="OpenAI provides powerful embedding models."),
]

# step 2: initialize embedding model
embedding_model = OpenAIEmbeddings()

# step 3: create chroma vector store in memory
vectorstore = Chroma.from_documents(
    documents=documents,
    embedding = embedding_model,
    collection_name = 'my_collection'
)

# step 4: convert vectore store into retriever
retriever = vectorstore.as_retriever(search_kwargs={"k":2})
# k=2 will give two relavant results

query = "What is chroma used for?"
results = retriever.invoke(query)

for i, doc in enumerate(results):
    print(f"\n---Result{i+1}---")
    print(doc.page_content)