from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader('LangChain Notes.pdf')

docs = loader.load()

print(len(docs)) # this will give number of pages in documents

print(docs[0].page_content)
print(docs[0].metadata)