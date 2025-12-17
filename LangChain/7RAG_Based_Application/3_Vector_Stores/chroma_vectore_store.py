# run it on terminal
# pip install langchain chormadb openai tiktoken pypdf langchain_openai langchain_community

from lagchain_openai import OpenAIEmbeddings
from langchain.vectorestores import chormadb

from langchain.schema import Document

doc1 = Document(
    page_content="Virat Kohli is one of the most successful and consistent batsmen in IPL history. Known for his aggressive batting style and leadership qualities.",
    metadata={"team": "Royal Challengers Bangalore"}
)

doc2 = Document(
    page_content="Rohit Sharma is the most successful captain in IPL history, leading Mumbai Indians to five titles. He's known for his calm leadership and explosive batting.",
    metadata={"team": "Mumbai Indians"}
)

doc3 = Document(
    page_content="MS Dhoni, famously known as Captain Cool, has led Chennai Super Kings to multiple IPL titles. His finishing skills, wicket-keeping, and tactical acumen make him legendary.",
    metadata={"team": "Chennai Super Kings"}
)

doc4 = Document(
    page_content="Jasprit Bumrah is considered one of the best fast bowlers in T20 cricket. Playing for Mumbai Indians, he is known for his yorkers and death bowling skills.",
    metadata={"team": "Mumbai Indians"}
)

doc5 = Document(
    page_content="Ravindra Jadeja is a dynamic all-rounder who contributes with both bat and ball. Representing Chennai Super Kings, his quick fielding and spin bowling are exceptional.",
    metadata={"team": "Chennai Super Kings"}
)


vector_store = Chroma(
    embedding_function = OpenAIEmbeddings(),
    persist_dictonary = 'my_chorma_db',
    collection_name = 'sample'
)

vector_store.add_document(docs)

vector_store.get(include = ['emdeddings', 'documents', 'metadatas'])

vector_store.similarity_search(
    query = 'Who among these are a bolwer?',
    k=2
)

vector_store.similarity_search_with_score(
    query = 'Who among these are a bolwer?',
    k=2
)

vector_store.similarity_search(
    query = '',
    filter = {"team":"Chennai Super Kings"}
)


# update documents
updated_doc1 = Document(
    page_content="Virat Kohli, the former captain of Royal Challengers Bangalore (RCB), is renowned for his aggressive leadership and consister",
    metadata={"team": "Royal Challengers Bangalore"}
)

vector_store.update_document(document_id='09a39dc5-3ba6-4ea7-9f27-fdda591da5e4', document=updated_doc1)


vector_store.delete(ids=['',''])