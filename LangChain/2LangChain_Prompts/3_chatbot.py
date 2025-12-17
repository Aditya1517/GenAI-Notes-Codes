from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from dotenv import load_dotenv

load_dotenv()
model = ChatOpenAI() # since this a chatbot it will infinitly run unless user types exit

chathistory = [
    SystemMessage(content="You are a helpful AI assistant.")
]

while True:
    user_input = input("You: ")
    chat_history.append(HumanMessage(content = user_input))
    if user_input.lower() == "exit":
        print("Exiting the chatbot. Goodbye!")
        break
    response = model.invoke(user_input)
    chat_history.append(AIMessage(content = response.content))
    print("Bot:", response.content)