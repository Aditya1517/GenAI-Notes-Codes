from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage, HumanMessage

# in chat history we are having a system message and human message

# chat template
chat_template = ChatPromptTemplate([
    ('system', 'You are a helpful support agent.'),
    # here system will not understand the query coz it is not having privioud messages
    # so we will add a placeholder for chat history
    MessagesPlaceholder(variable_name='chat_history'),
    ('human', '{query}') # today's query
])

chat_history = []
# load chat history
with open('chat_history.txt') as f:
    chat_history.extend(f.readlines())
    
print(chat_history)

# create prompt
prompt = chat_template.invoke({'chat_history':chat_history, 'query':'Where is my refund?'})

