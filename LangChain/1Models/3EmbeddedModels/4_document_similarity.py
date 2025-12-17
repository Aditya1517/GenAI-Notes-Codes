from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

load_dotenv()

embeddings = OpenAIEmbeddings(model='text-embedding-3-large', dimensions=300)


documents = [
"Virat Kohli is an Indian cricketer known for his aggressive batting and leadership.",
"MS Dhoni is a former Indian captain famous for his calm demeanor and finishing skills.",
"Sachin Tendulkar, also known as the 'God of Cricket', holds many batting records.",
"Rohit Sharma is known for his elegant batting and record-breaking double centuries.",
"Jasprit Bumrah is an Indian fast bowler known for his unorthodox action and yorkers."
]

query = "Tell me about the Virat Kohli"

doc_embeddings = embeddings.embed_documents(documents)  # 5 vectors with 300 dimensions
query_embedding = embeddings.embed_query(query)  # single vector

# now at this point we have 5 document vectors and one query vector; compute similarity
scores = cosine_similarity([query_embedding], doc_embeddings)[0]  # cosine_similarity expects 2D arrays

# get index and score of the most similar document (highest cosine similarity)
index, score = sorted(list(enumerate(scores)), key=lambda x: x[1])[-1]  # will give (index, score)

print(f"Most similar document index: {index}, score: {score}")
print("Document:", documents[index])

# here we are doing symantic search