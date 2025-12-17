from langchain_community.retrievers import WikipediaRetriever

# initialize the retriever (optional: set language and top_k)
retriever = WikipediaRetriever(top_k_results=2, lang='en')

# define the query
query = "the geopolitical history of india and pakistan from the prespective of a chinese"

# get relavant wikipedia documents
docs = retriever.invoke(query)

# print retriever content
for i, doc in enumerate(docs):
    print(f"\n---Result")
    print(f"\n {doc.page_content}...")  #truncate for display


