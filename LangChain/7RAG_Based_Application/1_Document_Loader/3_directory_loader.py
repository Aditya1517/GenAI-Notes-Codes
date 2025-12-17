from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader

loader = DirectoryLoader(
    path = 'New folder',
    glob = '*.pdf' # this will load all the pdfs
    loader_cls = PyPDFLoader
)

docs = loader.load()

print(len(docs))
print(docs[0].page_content)
print(docs[0].metadata)

# this laoding takes time and for that we lazy loding method
docs1 = loader.lazy_load()