from langchain_core import SystemMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI()

messages = [
    SystemMessage(content='You are a very helpful assistant.'),
    HumanMessage(content = "Tell me about langchain")
]

result = model.invoke(messages)

messages.append(AIMessage(content=result.content))

print(messages)

# now integrate this with chatbot.py