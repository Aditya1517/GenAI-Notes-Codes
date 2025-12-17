from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

chat = ChatOpenAI(model="gpt-4", temperture=0)

result = chat.invoke("What is the capital of India?")

print(result.content)