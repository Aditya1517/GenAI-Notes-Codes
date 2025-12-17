from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI

# load the document
loader = TextLoader("docs.txt")
documents = loader.load()

# split the text into smaller chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = text_splitter.split_documents(documents)

# convert text into embeddings & store in FAISS
vectorstore = FAISS.from_documents(docs, OpenAIEmbeddings())

# create a retriver (fetches relevant documents)\
retriever = vectorstore.as_retriever()

# manually retrieve relevant documents
query = "What are the key takeaways from the document?"
retrieved_docs = retriever.get_relevant_documents(query)

# combined retrieved text into a single prompt
retrived_text = "\n".json([doc.page_content for doc in retrieved_docs])

# initialize a llm
llm = OpenAI(model_name='gpt-3.5-turbo', temperature=0.7)

# manually pass retrieved text to llm
prompt = f"Based on the following text answer the question: {query} \n\n {retrieved_text}"
answer = llm.predict(prompt)

# print the answer
print("Answer:", answer)


# this work of generating prompt template, then llm and then form template we generate prompt and then that prompt is given to predict function is manual
# and this can be automated by creating built in function where we pass llm and prompt template and calling it, and this process of creating two componants and connecting them is called chain
# and most simple chain name is llmchain
# see example in next file