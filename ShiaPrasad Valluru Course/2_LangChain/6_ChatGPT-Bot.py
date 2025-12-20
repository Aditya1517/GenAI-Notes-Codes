import streamlit as st
from dotenv import load_dotenv
import uuid # this will generate random user id
from datetime import datetime
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableWithMessageHistory, ConfigurableFieldSpec
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

if "store" not in st.session_state:
    st.session_state.store = {}

if "user_id" not in st.session_state:
    st.session_state.user_id = str(uuid.uuid4())  # generate a unique user id

# Ensure we use a consistent key name `conversations` (dict of conv_id -> data)
if "conversations" not in st.session_state:
    # create an initial conversation
    st.session_state.conversations = {}
    st.session_state.conversation_counter = 1
    initial_conv_id = str(uuid.uuid4())
    st.session_state.conversations[initial_conv_id] = {
        "number": st.session_state.conversation_counter,
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "messages": []
    }
    st.session_state.current_conversation_id = initial_conv_id
else:
    # conversations exists; ensure counters and current id are present
    if "conversation_counter" not in st.session_state:
        st.session_state.conversation_counter = max(
            (v.get("number", 0) for v in st.session_state.conversations.values()),
            default=0,
        )
    if "current_conversation_id" not in st.session_state:
        # pick an existing conversation if available
        if st.session_state.conversations:
            st.session_state.current_conversation_id = next(iter(st.session_state.conversations.keys()))
        else:
            new_conv_id = str(uuid.uuid4())
            st.session_state.conversation_counter += 1
            st.session_state.conversations[new_conv_id] = {
                "number": st.session_state.conversation_counter,
                "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "messages": []
            }
            st.session_state.current_conversation_id = new_conv_id

def get_session_history(user_id: str, conversation_id: str) -> ChatMessageHistory:
    if (user_id, conversation_id) not in st.session_state.store:
        st.session_state.store[(user_id, conversation_id)] = ChatMessageHistory()
    return st.session_state.store[(user_id, conversation_id)]


def create_new_conversation():
    st.session_state.conversation_counter += 1
    new_conv_id = str(uuid.uuid4())
    st.session_state.conversations[new_conv_id] = {
        "number": st.session_state.conversation_counter,
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "messages": []
    }

    st.session_state.current_conversation_id = new_conv_id
    
prompt = ChatPromptTemplate.from_messages([
    (
        "system","You are a helpful assistant. Answer all the question to the best of you ability in {language}.",
    ),
    MessagesPlaceholder(variable_name="my_chat_history"),
    ("human","{input}")
])    

chain = prompt | llm

model_with_message_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="my_chat_history",
    history_factory_config=[
        ConfigurableFieldSpec(
            id="user_id",
            annotation=str,
            name="User ID",
            is_shared=True,
        ),
        ConfigurableFieldSpec(
            id="conversation_id",
            annotation=str,
            name="Conversation ID",
            is_shared=True,
        ),
    ],
)

def handle_chat_input(prompt, config):
    current_conv = st.session_state.conversations[st.session_state.current_conversation_id]
    current_conv["messages"].append({"role": "user", "content": prompt})

    result = model_with_message_history.invoke(
        {"input": prompt, "language": "English"},
        config=config,
    )

    current_conv["messages"].append({"role": "assistant", "content": result.content})
    
    

st.title("chatBot")

with st.sidebar:
    st.subheader("Conversations")
    
    if st.button("New Conversation"):
        create_new_conversation()
        
    conversation_list = [(conv_id, f"Conversation {conv_data['number']}")
                         for conv_id, conv_data in st.session_state.conversations.items()]
    
    print(conversation_list)
    
    selected_conv = st.selectbox(
        "Select Conversation",
        options=[conv[0] for conv in conversation_list],
        format_func = lambda conv_id: next(conv[1] for conv in conversation_list if conv[0] == conv_id),
        index = list(st.session_state.conversations.keys()).index(st.session_state.current_conversation_id) 
    )
    
    if selected_conv != st.session_state.current_conversation_id:
        st.session_state.current_conversation_id = selected_conv
    
    st.sidebar.divider()
    st.sidebar.write(f"Current User ID: {st.session_state.user_id}")
    st.sidebar.write(f"Current Conversation ID: {st.session_state.current_conversation_id}")
    
    current_conv = st.session_state.conversations[st.session_state.current_conversation_id]
    
if prompt := st.chat_input("What's on your mind?"):
    config = {
        "configurable": {
            "user_id": st.session_state.user_id,
            "conversation_id": st.session_state.current_conversation_id
        }
    }
    
    handle_chat_input(prompt, config)
    

for message in current_conv["messages"]:
    with st.chat_message(message["role"]):
        st.write(message["content"])